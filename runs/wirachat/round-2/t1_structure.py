"""Round 2 - Task 1: structure checks (gaps, linear vs GBM AUC, missingness information). Read-only."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mutual_info_score
warnings.filterwarnings("ignore")
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load, SEED

lab = load(1, "train").dropna(subset=["is_cheater"]).reset_index(drop=True)
y = lab.is_cheater.astype(int).values
feat = [c for c in lab.columns if c not in ("player_id", "is_cheater")]
X = lab[feat]
# gaps
rows = []
for c in feat:
    v = X[c].dropna().values
    h, e = np.histogram(v, bins=100); nz = np.where(h > 0)[0]
    gaps = [(i1 - i0 - 1, (e[i0 + 1] + e[i1]) / 2) for i0, i1 in zip(nz[:-1], nz[1:])]
    if gaps:
        n_empty, cut = max(gaps); small = min((v < cut).sum(), (v > cut).sum())
        if n_empty >= 3 and small >= 30:
            g = ((X[c] < cut) if (v < cut).sum() < (v > cut).sum() else (X[c] > cut)).fillna(False)
            rows.append((c, n_empty, round(cut, 3), int(small), round(y[g.values].mean(), 3)))
print("Task 1 columns with an internal gap (>=3 empty bins of 100):", rows or "none")
# linear vs boosted
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
lr = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), LogisticRegression(max_iter=500))
p = cross_val_predict(lr, X, y, cv=skf, method="predict_proba", n_jobs=2)[:, 1]
print(f"LogReg 5-fold AUC = {roc_auc_score(y, p):.4f}   [LightGBM round 1: 0.8911]")
# does missingness carry label information?
mi = {c: mutual_info_score(y, X[c].isna().astype(int)) for c in feat if X[c].isna().any()}
print("NaN-indicator mutual information with label, top 5:", ", ".join(f"{k}={v:.5f}" for k, v in sorted(mi.items(), key=lambda t: -t[1])[:5]))
nm = X.isna().sum(axis=1)
print("cheater rate by #missing features:", pd.Series(y).groupby(nm.clip(upper=10)).mean().round(3).to_dict())
# univariate separation
auc = {c: roc_auc_score(y[X[c].notna()], X[c].dropna()) for c in feat}
top = sorted(auc.items(), key=lambda t: -abs(t[1] - 0.5))[:8]
print("top univariate AUCs:", ", ".join(f"{k}={v:.3f}" for k, v in top))
