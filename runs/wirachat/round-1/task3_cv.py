"""Round 1 - Task 3: NMAE = 1/(1+sum|y-p|/sum y). Compare L1 on raw scale vs L1 on log1p scale (median-consistent)."""
import sys, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from r1_common import load, nmae_score, OUT, SEED

tr, te = load(3, "train"), load(3, "test")
y = tr.spending_30d.values
feat = [c for c in tr.columns if c not in ("player_id", "spending_30d")]
X, Xte = tr[feat], te[feat]
strat = (tr.segment.fillna(-1).astype(int).astype(str) + (y > 0).astype(int).astype(str)).values
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
base = dict(learning_rate=0.03, num_leaves=63, min_child_samples=30, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=SEED, n_jobs=16)
res = {}
for name, tgt, inv, obj in [("l1_raw", lambda v: v, lambda p: p, "l1"),
                            ("l1_log1p", np.log1p, np.expm1, "l1"),
                            ("l2_log1p", np.log1p, np.expm1, "l2")]:
    oof = np.zeros(len(X)); pte = np.zeros(len(Xte)); iters = []
    for k, (a, b) in enumerate(skf.split(X, strat)):
        m = lgb.train({**base, "objective": obj, "metric": "l1"}, lgb.Dataset(X.iloc[a], tgt(y[a])), 4000,
                      valid_sets=[lgb.Dataset(X.iloc[b], tgt(y[b]))], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[b] = np.clip(inv(m.predict(X.iloc[b], num_iteration=m.best_iteration)), 0, 300000)
        pte += np.clip(inv(m.predict(Xte, num_iteration=m.best_iteration)), 0, 300000) / 5
        iters.append(m.best_iteration)
    s = nmae_score(y, oof)
    res[name] = dict(nmae=s, iters=iters)
    np.save(OUT / f"task3_oof_{name}.npy", oof); np.save(OUT / f"task3_test_{name}.npy", pte)
    print(f"{name}: NMAE={s:.4f}  MAE={np.abs(y - oof).mean():.1f}  iters={iters}", flush=True)
json.dump(res, open(OUT / "task3_metrics_a.json", "w"), indent=1)
