"""Round 2 - headroom diagnostics for Task 1 (model capacity / shift) and Task 3 (where the error comes from). Read-only."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, fbeta_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load, nmae_score, SEED

OUT = Path(__file__).resolve().parent
# ============ TASK 1
lab = load(1, "train").dropna(subset=["is_cheater"]).reset_index(drop=True)
y = lab.is_cheater.astype(int).values
feat = [c for c in lab.columns if c not in ("player_id", "is_cheater")]
X = lab[feat]
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
print("=== TASK 1: AUC by model capacity (interaction order)")
for name, leaves in [("depth-1 stumps (additive)", 2), ("4 leaves", 4), ("8 leaves", 8), ("31 leaves", 31)]:
    oof = np.zeros(len(X))
    for a, b in folds:
        m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=leaves, min_child_samples=40, feature_fraction=0.8,
                           bagging_fraction=0.8, bagging_freq=1, lambda_l2=5, verbose=-1, seed=SEED, n_jobs=12),
                      lgb.Dataset(X.iloc[a], y[a]), 4000, valid_sets=[lgb.Dataset(X.iloc[b], y[b])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[b] = m.predict(X.iloc[b], num_iteration=m.best_iteration)
    print(f"  {name:28s} AUC={roc_auc_score(y, oof):.4f}  best plain F2={max(fbeta_score(y, oof >= t, beta=2) for t in np.linspace(.05, .6, 56)):.4f}", flush=True)
q = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.05))
oq = np.zeros(len(X))
for a, b in folds:
    q.fit(X.iloc[a], y[a]); oq[b] = q.predict_proba(X.iloc[b])[:, 1]
print(f"  QDA (class-conditional Gaussians)  AUC={roc_auc_score(y, oq):.4f}")
# train vs test probability shift (round-1 blends)
oof = (np.load(R1 / "task1_oof_raw.npy") + np.load(R1 / "task1_oof_fe.npy")) / 2
te = (np.load(R1 / "task1_test_prob_raw.npy") + np.load(R1 / "task1_test_prob_fe.npy")) / 2
print(f"  mean predicted prob: OOF {oof.mean():.3f} vs test {te.mean():.3f} | share >=0.18: OOF {(oof >= .18).mean():.3f} vs test {(te >= .18).mean():.3f}")
print("  prob quantiles .1/.25/.5/.75/.9  OOF:", np.round(np.quantile(oof, [.1, .25, .5, .75, .9]), 3).tolist(), " test:", np.round(np.quantile(te, [.1, .25, .5, .75, .9]), 3).tolist())

# ============ TASK 3
t3 = load(3, "train"); y3 = t3.spending_30d.values
pred = np.load(OUT / "t3_twostage_oof.npy")
seg = t3.segment.fillna(4).astype(int).values
err = np.abs(y3 - pred); tot = err.sum()
print("\n=== TASK 3: where the absolute error comes from (two-stage OOF, NMAE %.4f)" % nmae_score(y3, pred))
zero_true, zero_pred = y3 == 0, pred == 0
print(f"  y>0 predicted 0 (missed spenders): {err[~zero_true & zero_pred].sum()/tot:.3f} of error")
print(f"  y=0 predicted >0 (phantom spend):  {err[zero_true & ~zero_pred].sum()/tot:.3f}")
print(f"  y>0 predicted >0 (size error):     {err[~zero_true & ~zero_pred].sum()/tot:.3f}")
for s in range(5):
    m = seg == s
    print(f"  segment {s}: rows {m.sum():>6}, zero share {zero_true[m].mean():.3f}, error share {err[m].sum()/tot:.3f}, y=0&pred>0 share of seg error {err[m & zero_true & ~zero_pred].sum()/max(err[m].sum(),1):.3f}")
# is the zero process predictable beyond the segment?
feat3 = [c for c in t3.columns if c not in ("player_id", "spending_30d")]
f3 = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(t3[feat3], seg * 2 + zero_true))
pz = np.zeros(len(t3))
for a, b in f3:
    m = lgb.train(dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=40, verbose=-1, seed=SEED, n_jobs=12),
                  lgb.Dataset(t3[feat3].iloc[a], zero_true[a].astype(int)), 2000, valid_sets=[lgb.Dataset(t3[feat3].iloc[b], zero_true[b].astype(int))],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    pz[b] = m.predict(t3[feat3].iloc[b], num_iteration=m.best_iteration)
seg_rate = pd.Series(zero_true).groupby(seg).transform("mean").values
print(f"  zero AUC: full model {roc_auc_score(zero_true, pz):.4f} vs segment-only {roc_auc_score(zero_true, seg_rate):.4f}")
for s in range(5):
    m = seg == s
    print(f"    within segment {s}: AUC of the full model for y==0 = {roc_auc_score(zero_true[m], pz[m]):.3f}")
# oracle: if the zero status were known, what would NMAE be?
orc = np.where(zero_true, 0.0, np.where(pred > 0, pred, np.median(y3[y3 > 0])))
print(f"  oracle NMAE with the zero status known: {nmae_score(y3, orc):.4f}  (current {nmae_score(y3, pred):.4f})")
g = t3.discount_rate_used < 0.057
print(f"  discount_rate_used<0.057 group: {int(g.sum())} rows, zero share {zero_true[g.values].mean():.3f} vs rest {zero_true[~g.values].mean():.3f}")
