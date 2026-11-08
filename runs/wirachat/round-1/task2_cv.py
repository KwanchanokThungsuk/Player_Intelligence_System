"""Round 1 - Task 2: LightGBM multiclass 5-fold CV, noise columns dropped, class-weight tuning for macro F1."""
import sys, json, itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from r1_common import load, OUT, SEED

DROP_NOISE = "--keep-noise" not in sys.argv
tr, te = load(2, "train"), load(2, "test")
y = tr.segment.astype(int).values
cats = [c for c in tr.select_dtypes(exclude=np.number).columns if c != "player_id"]
feat = [c for c in tr.columns if c not in ("player_id", "segment")]
if DROP_NOISE:
    feat = [c for c in feat if not c.startswith("random_metric")]


def prep(df, ref):
    X = df[feat].copy()
    for c in cats:
        if c in X:
            X[c] = pd.Categorical(X[c], categories=sorted(ref[c].dropna().unique()))
    return X


X, Xte = prep(tr, tr), prep(te, tr)
params = dict(objective="multiclass", num_class=4, learning_rate=0.03, num_leaves=31, min_child_samples=40,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=SEED, n_jobs=16)
oof = np.zeros((len(X), 4)); pte = np.zeros((len(Xte), 4)); iters = []
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
for k, (a, b) in enumerate(skf.split(X, y)):
    m = lgb.train(params, lgb.Dataset(X.iloc[a], y[a]), 3000, valid_sets=[lgb.Dataset(X.iloc[b], y[b])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    oof[b] = m.predict(X.iloc[b], num_iteration=m.best_iteration)
    pte += m.predict(Xte, num_iteration=m.best_iteration) / 5
    iters.append(m.best_iteration)
    print(f"fold {k}: iters={m.best_iteration} macroF1={f1_score(y[b], oof[b].argmax(1), average='macro'):.4f}", flush=True)

f_arg = f1_score(y, oof.argmax(1), average="macro")
# coordinate-ascent search of per-class probability multipliers to maximise macro F1 on OOF
w = np.ones(4)
grid = np.linspace(0.6, 1.8, 25)
for _ in range(3):
    for c in range(4):
        best = (f1_score(y, (oof * w).argmax(1), average="macro"), w[c])
        for g in grid:
            w2 = w.copy(); w2[c] = g
            s = f1_score(y, (oof * w2).argmax(1), average="macro")
            if s > best[0]:
                best = (s, g)
        w[c] = best[1]
f_w = f1_score(y, (oof * w).argmax(1), average="macro")
print(f"drop_noise={DROP_NOISE} argmax macroF1={f_arg:.4f} | weighted macroF1={f_w:.4f} w={np.round(w, 2).tolist()}")
print("per-class F1 (weighted):", np.round(f1_score(y, (oof * w).argmax(1), average=None), 4).tolist())
tag = "clean" if DROP_NOISE else "noise"
np.save(OUT / f"task2_oof_{tag}.npy", oof); np.save(OUT / f"task2_test_prob_{tag}.npy", pte)
json.dump(dict(drop_noise=DROP_NOISE, macro_f1_argmax=f_arg, macro_f1_weighted=f_w, weights=w.tolist(), iters=iters),
          open(OUT / f"task2_metrics_{tag}.json", "w"), indent=1)
