"""Round 2 - Task 3: structure of the size error among positive spenders (log-space residuals). Read-only."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load, SEED

t3 = load(3, "train"); pos = (t3.spending_30d > 0).values
P = t3[pos].reset_index(drop=True)
feat = [c for c in P.columns if c not in ("player_id", "spending_30d")]
ly = np.log(P.spending_30d.values)
oof = np.zeros(len(P))
for a, b in KFold(5, shuffle=True, random_state=SEED).split(P):
    m = lgb.train(dict(objective="l2", learning_rate=0.05, num_leaves=63, min_child_samples=30, feature_fraction=0.8, bagging_fraction=0.8,
                       bagging_freq=1, lambda_l2=5, verbose=-1, seed=SEED, n_jobs=12),
                  lgb.Dataset(P[feat].iloc[a], ly[a]), 4000, valid_sets=[lgb.Dataset(P[feat].iloc[b], ly[b])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    oof[b] = m.predict(P[feat].iloc[b], num_iteration=m.best_iteration)
r = ly - oof
print(f"positive rows {len(P)} | log residual: sd {r.std():.3f} | quantiles 1/5/25/50/75/95/99: {np.round(np.quantile(r, [.01, .05, .25, .5, .75, .95, .99]), 3).tolist()}")
g = GaussianMixture(2, random_state=SEED, n_init=3).fit(r.reshape(-1, 1))
o = np.argsort(g.weights_)
print("1-D mixture of residuals: weights", np.round(g.weights_[o], 3).tolist(), "means", np.round(g.means_.ravel()[o], 3).tolist(), "sds", np.round(np.sqrt(g.covariances_.ravel()[o]), 3).tolist())
big = np.abs(r) > 0.6
print(f"rows with |residual|>0.6 (a factor of 1.8 off): {int(big.sum())} ({big.mean():.3%}); share of total abs error (raw scale): "
      f"{(np.abs(np.exp(ly) - np.exp(oof))[big].sum() / np.abs(np.exp(ly) - np.exp(oof)).sum()):.3f}")
print("direction of the big residuals: positive", int((r[big] > 0).sum()), "negative", int((r[big] < 0).sum()))
# are the big residuals predictable from the features?
clf_oof = np.zeros(len(P))
for a, b in KFold(5, shuffle=True, random_state=SEED).split(P):
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=15, min_child_samples=30, verbose=-1, seed=SEED, n_jobs=12),
                  lgb.Dataset(P[feat].iloc[a], big[a].astype(int)), 1500, valid_sets=[lgb.Dataset(P[feat].iloc[b], big[b].astype(int))],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    clf_oof[b] = m.predict(P[feat].iloc[b], num_iteration=m.best_iteration)
print(f"AUC for predicting 'big residual' from features: {roc_auc_score(big, clf_oof):.3f}")
# can the residual itself be predicted (second stage)?
r2 = np.zeros(len(P))
for a, b in KFold(5, shuffle=True, random_state=SEED + 1).split(P):
    m = lgb.train(dict(objective="l2", learning_rate=0.05, num_leaves=15, min_child_samples=50, verbose=-1, seed=SEED, n_jobs=12),
                  lgb.Dataset(P[feat].iloc[a], r[a]), 1500, valid_sets=[lgb.Dataset(P[feat].iloc[b], r[b])], callbacks=[lgb.early_stopping(100, verbose=False)])
    r2[b] = m.predict(P[feat].iloc[b], num_iteration=m.best_iteration)
print(f"second-stage R2 on residuals: {1 - ((r - r2) ** 2).sum() / ((r - r.mean()) ** 2).sum():.4f}")
cors = pd.Series({c: np.corrcoef(np.nan_to_num(P[c].values, nan=np.nanmedian(P[c])), np.abs(r))[0, 1] for c in feat}).abs().sort_values(ascending=False)
print("features most correlated with |residual|:", ", ".join(f"{k}={v:.3f}" for k, v in cors.head(6).items()))
# residual by segment and by a few categorical-like columns
print("residual sd by segment:", P.groupby(P.segment.fillna(4)).apply(lambda d: float(np.std(r[d.index]))).round(3).to_dict())
for c in ["owns_limited_edition", "is_premium_member", "vip_status", "primary_game", "platform"]:
    print(f"  residual sd by {c}:", pd.Series(r).groupby(P[c].fillna(-1).values).std().round(3).to_dict())
