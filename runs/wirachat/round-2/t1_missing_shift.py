"""Round 2 - Task 1: does the train/test difference in per-column missingness hurt? (evaluation uses simulated masking on validation folds)
Part A measures the damage; part B tests a generic robustness fix (random column masking during training, no test-derived numbers)."""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, fbeta_score
warnings.filterwarnings("ignore")
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load, SEED

tr, te = load(1, "train"), load(1, "test")
lab = tr.dropna(subset=["is_cheater"]).reset_index(drop=True)
y = lab.is_cheater.astype(int).values
feat = [c for c in lab.columns if c not in ("player_id", "is_cheater")]
X = lab[feat]
rt, rte = X.isna().mean(), te[feat].isna().mean()
d = (rte - rt).sort_values(ascending=False)
print("columns where TEST has more missing than train (test - train):", ", ".join(f"{k}={v:+.3f}" for k, v in d[d > 0.005].items()) or "none")
print("columns where TRAIN has more missing than test:", ", ".join(f"{k}={v:+.3f}" for k, v in d.tail(6).items()))
# extra masking probability that would make a (labelled) validation row look like a test row
p_extra = ((rte - rt).clip(lower=0) / (1 - rt)).values
rng = np.random.default_rng(SEED)


def mask_like_test(df, seed):
    r = np.random.default_rng(seed); M = r.random(df.shape) < p_extra[None, :]
    out = df.copy(); out = out.mask(M); return out


def mask_uniform(df, p, seed):
    r = np.random.default_rng(seed); return df.mask(r.random(df.shape) < p)


params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=40, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=5.0, verbose=-1, seed=SEED, n_jobs=12)
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))


def f2(yv, p):
    return max(fbeta_score(yv, p >= t, beta=2) for t in np.linspace(0.08, 0.5, 43))


res = {"baseline": [], "augmented": []}
for k, (a, b) in enumerate(folds):
    Xa, ya, Xb, yb = X.iloc[a], y[a], X.iloc[b], y[b]
    Xb_shift = mask_like_test(Xb, SEED + k)
    for name in ("baseline", "augmented"):
        if name == "baseline":
            Xt, yt = Xa, ya
        else:  # original + two randomly masked copies (uniform 12% column masking: a generic robustness augmentation)
            Xt = pd.concat([Xa, mask_uniform(Xa, 0.12, SEED + 10 + k), mask_uniform(Xa, 0.12, SEED + 20 + k)], ignore_index=True)
            yt = np.concatenate([ya, ya, ya])
        m = lgb.train(params, lgb.Dataset(Xt, yt), 5000, valid_sets=[lgb.Dataset(Xb, yb)], callbacks=[lgb.early_stopping(100, verbose=False)])
        pc, ps = m.predict(Xb, num_iteration=m.best_iteration), m.predict(Xb_shift, num_iteration=m.best_iteration)
        res[name].append((roc_auc_score(yb, pc), roc_auc_score(yb, ps), f2(yb, pc), f2(yb, ps)))
    print(f"fold {k} done", flush=True)
for name, v in res.items():
    v = np.mean(v, axis=0)
    print(f"{name:10s}: clean-validation AUC {v[0]:.4f} / best F2 {v[2]:.4f} | test-like-missingness validation AUC {v[1]:.4f} / best F2 {v[3]:.4f}")
json.dump({k: np.mean(v, axis=0).tolist() for k, v in res.items()}, open(Path(__file__).resolve().parent / "t1_missing_shift.json", "w"))
