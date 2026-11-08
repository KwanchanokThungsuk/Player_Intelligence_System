"""Round 1 - Task 3: LightGBM L1 (raw scale) configuration search; saves OOF/test predictions per config."""
import sys, json
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from r1_common import load, nmae_score, OUT, SEED

tr, te = load(3, "train"), load(3, "test")
y = tr.spending_30d.values
feat = [c for c in tr.columns if c not in ("player_id", "spending_30d")]
X, Xte = tr[feat], te[feat]
strat = (tr.segment.fillna(-1).astype(int).astype(str) + (y > 0).astype(int).astype(str)).values
skf = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, strat))
configs = {
    "A_lr08_l63": dict(learning_rate=0.08, num_leaves=63, min_child_samples=30, feature_fraction=0.8, lambda_l2=5.0),
    "B_lr08_l127": dict(learning_rate=0.08, num_leaves=127, min_child_samples=20, feature_fraction=0.8, lambda_l2=5.0),
    "C_lr08_l31": dict(learning_rate=0.08, num_leaves=31, min_child_samples=50, feature_fraction=0.6, lambda_l2=5.0),
    "D_lr05_l255": dict(learning_rate=0.05, num_leaves=255, min_child_samples=40, feature_fraction=0.7, lambda_l2=10.0),
    "E_lr08_l15": dict(learning_rate=0.08, num_leaves=15, min_child_samples=80, feature_fraction=0.7, lambda_l2=2.0),
}
res = {}
for name, cfg in configs.items():
    p = dict(objective="l1", metric="l1", bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=SEED, n_jobs=10, **cfg)
    oof = np.zeros(len(X)); pte = np.zeros(len(Xte)); iters = []
    for a, b in skf:
        m = lgb.train(p, lgb.Dataset(X.iloc[a], y[a]), 8000, valid_sets=[lgb.Dataset(X.iloc[b], y[b])],
                      callbacks=[lgb.early_stopping(250, verbose=False)])
        oof[b] = np.clip(m.predict(X.iloc[b], num_iteration=m.best_iteration), 0, 300000)
        pte += np.clip(m.predict(Xte, num_iteration=m.best_iteration), 0, 300000) / 5
        iters.append(m.best_iteration)
    s = nmae_score(y, oof)
    res[name] = dict(nmae=s, iters=iters)
    np.save(OUT / f"task3_oof_{name}.npy", oof); np.save(OUT / f"task3_test_{name}.npy", pte)
    print(f"{name}: NMAE={s:.4f} iters={iters}", flush=True)
json.dump(res, open(OUT / "task3_metrics_tune.json", "w"), indent=1)
