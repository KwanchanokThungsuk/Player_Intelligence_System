"""Round 2 - Task 3: what identifies the ~1% 'big residual' spenders? Read-only."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, export_text
warnings.filterwarnings("ignore")
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load, SEED

OUT = Path(__file__).resolve().parent
t3 = load(3, "train"); pos = (t3.spending_30d > 0).values
P = t3[pos].reset_index(drop=True)
feat = [c for c in P.columns if c not in ("player_id", "spending_30d")]
ly = np.log(P.spending_30d.values)
oof = np.zeros(len(P))
for a, b in KFold(5, shuffle=True, random_state=SEED).split(P):
    m = lgb.train(dict(objective="l2", learning_rate=0.05, num_leaves=63, min_child_samples=30, feature_fraction=0.8, bagging_fraction=0.8,
                       bagging_freq=1, lambda_l2=5, verbose=-1, seed=SEED, n_jobs=12), lgb.Dataset(P[feat].iloc[a], ly[a]), 4000,
                  valid_sets=[lgb.Dataset(P[feat].iloc[b], ly[b])], callbacks=[lgb.early_stopping(150, verbose=False)])
    oof[b] = m.predict(P[feat].iloc[b], num_iteration=m.best_iteration)
r = ly - oof; big = np.abs(r) > 0.6
np.save(OUT / "t3_logres.npy", r)
# importance of a classifier for 'big'
m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=15, min_child_samples=30, verbose=-1, seed=SEED, n_jobs=12),
              lgb.Dataset(P[feat], big.astype(int)), 300)
imp = pd.Series(m.feature_importance("gain"), index=feat).sort_values(ascending=False)
print("classifier gain importance (share):", ", ".join(f"{k}={v / imp.sum():.2f}" for k, v in imp.head(8).items()))
# simple rules: shallow decision tree on the most important features (NaN -> -999)
top = list(imp.head(8).index)
Xt = P[top].fillna(-999)
dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, class_weight="balanced", random_state=SEED).fit(Xt, big)
print(export_text(dt, feature_names=top, max_depth=3))
# distributions of the top features: big vs rest
for c in list(imp.head(4).index):
    v = P[c]
    print(f"{c}: NaN share big {v[big].isna().mean():.3f} vs rest {v[~big].isna().mean():.3f} | median big {v[big].median():.2f} vs rest {v[~big].median():.2f} | q10/q90 big {v[big].quantile(.1):.2f}/{v[big].quantile(.9):.2f} rest {v[~big].quantile(.1):.2f}/{v[~big].quantile(.9):.2f}")
