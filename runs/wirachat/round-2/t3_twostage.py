"""Round 2 - Task 3: two-stage model (P(y=0) + log-spend for positives) combined into the conditional MEDIAN.
Compares against the round-1 L1 LightGBM blend (CV NMAE 0.7854)."""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load, nmae_score, SEED

OUT = Path(__file__).resolve().parent
tr, te = load(3, "train"), load(3, "test")
y = tr.spending_30d.values
feat = [c for c in tr.columns if c not in ("player_id", "spending_30d")]
X, Xte = tr[feat], te[feat]
strat = (tr.segment.fillna(-1).astype(int).astype(str) + (y > 0).astype(int).astype(str)).values  # same folds as round 1
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, strat))
base = dict(learning_rate=0.05, num_leaves=63, min_child_samples=30, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
            lambda_l2=5.0, verbose=-1, seed=SEED, n_jobs=12)
pz = np.zeros(len(X)); mu = np.full(len(X), np.nan); pz_te = np.zeros(len(Xte)); mu_te = np.zeros(len(Xte))
pos = y > 0
for k, (a, b) in enumerate(folds):
    m1 = lgb.train({**base, "objective": "binary"}, lgb.Dataset(X.iloc[a], (y[a] > 0).astype(int)), 3000,
                   valid_sets=[lgb.Dataset(X.iloc[b], (y[b] > 0).astype(int))], callbacks=[lgb.early_stopping(150, verbose=False)])
    pz[b] = 1 - m1.predict(X.iloc[b], num_iteration=m1.best_iteration)          # P(y == 0)
    pz_te += (1 - m1.predict(Xte, num_iteration=m1.best_iteration)) / 5
    ap, bp = a[pos[a]], b[pos[b]]
    m2 = lgb.train({**base, "objective": "l2"}, lgb.Dataset(X.iloc[ap], np.log(y[ap])), 3000,
                   valid_sets=[lgb.Dataset(X.iloc[bp], np.log(y[bp]))], callbacks=[lgb.early_stopping(150, verbose=False)])
    mu[b] = m2.predict(X.iloc[b], num_iteration=m2.best_iteration)              # E[log y | y>0] for every row
    mu_te += m2.predict(Xte, num_iteration=m2.best_iteration) / 5
    print(f"fold {k}: zero-AUC={roc_auc_score(y[b] > 0, 1 - pz[b]):.4f} | positive log-rmse={np.sqrt(np.mean((np.log(y[bp]) - mu[bp]) ** 2)):.4f}", flush=True)

res = np.log(y[pos]) - mu[pos]
print(f"\nzero process AUC (OOF): {roc_auc_score(y > 0, 1 - pz):.4f} | share zero {(y == 0).mean():.3f}")
print(f"positive part: log-space R2 = {1 - res.var() / np.log(y[pos]).var():.4f}, residual sd = {res.std():.4f}, skew {pd.Series(res).skew():.2f}, kurt {pd.Series(res).kurt():.2f}")
seg = tr.segment.fillna(4).astype(int).values
sig = {s: float(np.std((np.log(y) - mu)[(seg == s) & pos])) for s in range(5)}
print("residual sd by segment:", {s: round(v, 3) for s, v in sig.items()})
sigma = np.array([sig[s] for s in seg])


def median_pred(pz_, mu_, sigma_):
    q = np.clip((0.5 - pz_) / np.maximum(1 - pz_, 1e-9), 1e-4, 1 - 1e-4)       # quantile of the positive part
    p = np.exp(mu_ + sigma_ * norm.ppf(q))
    return np.where(pz_ >= 0.5, 0.0, np.clip(p, 0, 300000))


pred = median_pred(pz, mu, sigma)
print(f"\ntwo-stage (lognormal median) NMAE = {nmae_score(y, pred):.4f}   [round-1 L1 blend 0.7854]")
base_oof = np.median([np.load(R1 / f"task3_oof_{n}.npy") for n in ["l1_raw", "A_lr08_l63", "B_lr08_l127", "C_lr08_l31", "D_lr05_l255", "E_lr08_l15"]], axis=0)
for w in (0.25, 0.5, 0.75):
    print(f"  blend {w:.2f}*two-stage + {1-w:.2f}*L1 blend: NMAE = {nmae_score(y, w * pred + (1 - w) * base_oof):.4f}")
# where does the error sit now?
err = np.abs(y - pred)
print("abs-error share by segment (two-stage):", {s: round(err[seg == s].sum() / err.sum(), 3) for s in range(5)})
np.save(OUT / "t3_twostage_oof.npy", pred)
np.save(OUT / "t3_twostage_test.npy", median_pred(pz_te, mu_te, np.array([sig[s] for s in te.segment.fillna(4).astype(int).values])))
json.dump(dict(nmae=nmae_score(y, pred), sigma=sig), open(OUT / "t3_twostage_summary.json", "w"), indent=1)
