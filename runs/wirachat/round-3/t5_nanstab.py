import sys, os, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
os.environ.setdefault("CPE342_DATA", "C:/Users/Wirachat/Desktop/CPE342-DATA-v2/kaggle/cpe342-kaggle-main")
R3 = Path("C:/Users/Wirachat/Desktop/CPE342-KAGGLE/runs/wirachat/round-3")
sys.path.insert(0, str(R3.parent / "round-1"))
from r1_common import load
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
OUT = R3 / "data2"
te = load(5, "test"); X = te.drop(columns=["player_id"]); n = len(X)
M = pd.read_pickle(OUT / "task5_groups.pkl"); other = M.drop(columns="behavior_stability").any(axis=1).values
bs = X.behavior_stability.values; grp = np.nan_to_num(bs, nan=9) < 0.552
nanb = np.isnan(bs)
# feature set: purchase_frequency_*, login_count_*, account_age_days + everything else (excluding stability)
feats = [c for c in X.columns if c != "behavior_stability"]
Z = X[feats].copy()
Z["pf_mean"] = X[[f"purchase_frequency_{i}" for i in range(1, 5)]].mean(1); Z["lc_mean"] = X[[f"login_count_{i}" for i in range(1, 5)]].mean(1)
tr = (~nanb) & (~other)
y = grp[tr]; Zt = Z[tr].values
oof = np.zeros(tr.sum())
for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Zt, y):
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, random_state=0).fit(Zt[a], y[a]); oof[b] = m.predict_proba(Zt[b])[:, 1]
print("OOF AUC (side effects only):", round(roc_auc_score(y, oof), 4), "| positives", int(y.sum()))
for t in (0.9, 0.5, 0.3, 0.2, 0.1, 0.05):
    s = oof > t; print(f"  OOF p>{t}: rows {int(s.sum())} precision {y[s].mean() if s.any() else float('nan'):.3f} recall {(y & s).sum() / y.sum():.3f}")
m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, random_state=0).fit(Zt, y)
cand = nanb & (~other)
p = np.zeros(n); p[cand] = m.predict_proba(Z[cand].values)[:, 1]
print("NaN-stability unflagged rows:", int(cand.sum()), "| expected group members ~", round(cand.sum() * y.mean(), 1), "| sum of probs", round(p[cand].sum(), 1))
for t in (0.9, 0.5, 0.3, 0.2, 0.1):
    print(f"  NaN rows p>{t}: {int((p > t).sum())}")
np.save(OUT / "task5_nanstab_prob.npy", p)
