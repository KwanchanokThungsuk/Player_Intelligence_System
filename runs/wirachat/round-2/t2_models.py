"""Round 2 - Task 2: is the GBM ceiling (macro F1 0.7503) the Bayes limit? Compare simple parametric models and look
at missingness / categorical information."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, mutual_info_score
warnings.filterwarnings("ignore")
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load, SEED

tr = load(2, "train"); y = tr.segment.astype(int).values
cats = [c for c in tr.select_dtypes(exclude=np.number).columns if c != "player_id"]
num = [c for c in tr.columns if c not in cats + ["player_id", "segment"] and not c.startswith("random_metric")]
D = pd.get_dummies(tr[cats].fillna("NA"), dtype=float)
Xn = pd.concat([tr[num], D], axis=1)
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
models = {
    "LDA": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearDiscriminantAnalysis()),
    "QDA": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), QuadraticDiscriminantAnalysis(reg_param=0.1)),
    "LogReg": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=500, C=1.0)),
    "GaussianNB": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), GaussianNB()),
}
print("5-fold macro F1 (LightGBM round 2 reference: 0.7503):")
for n, m in models.items():
    p = cross_val_predict(m, Xn, y, cv=skf, method="predict_proba", n_jobs=5)
    print(f"  {n:11s} {f1_score(y, p.argmax(1), average='macro'):.4f}", flush=True)

# information in missingness and in each categorical column
print("\nmutual information (nats) of NaN-indicator vs segment, top 5:")
mi = {c: mutual_info_score(y, tr[c].isna().astype(int)) for c in tr.columns if c not in ("player_id", "segment") and tr[c].isna().any()}
print(", ".join(f"{k}={v:.5f}" for k, v in sorted(mi.items(), key=lambda x: -x[1])[:5]))
print("mutual information of categorical columns vs segment:")
print(", ".join(f"{c}={mutual_info_score(y, tr[c].fillna('NA')):.4f}" for c in cats))
# univariate class separation of numeric features (eta^2 = between-class variance share)
eta = {}
for c in num:
    v = tr[c]; ok = v.notna()
    g = v[ok].groupby(y[ok])
    eta[c] = float(((g.mean() - v[ok].mean()) ** 2 * g.size()).sum() / ((v[ok] - v[ok].mean()) ** 2).sum())
print("\nnumeric features by class-separation (eta^2), top 12:")
print(", ".join(f"{k}={v:.3f}" for k, v in sorted(eta.items(), key=lambda x: -x[1])[:12]))
print("number of numeric features with eta^2 > 0.01:", sum(v > 0.01 for v in eta.values()), "of", len(eta))
