"""Round 3 - adversarial validation: can a GBM tell train rows from test rows? (Tasks 1, 2, 3 on the NEW data folder.) AUC about 0.5 = no shift, OOF scores transfer."""
import sys, os, warnings
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
os.environ.setdefault("CPE342_DATA", "C:/Users/Wirachat/Desktop/CPE342-DATA-v2/kaggle/cpe342-kaggle-main")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "round-1"))
from r1_common import load
for t, tgt in ((1, "is_high_value"), (2, "segment"), (3, "historical_spending")):
    tr, te = load(t, "train"), load(t, "test")
    cols = [c for c in te.columns if c in tr.columns and c != "player_id"]
    X = pd.concat([tr[cols], te[cols]], ignore_index=True)
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]): X[c] = X[c].astype("category")
    y = np.r_[np.zeros(len(tr)), np.ones(len(te))]
    for name, XX in (("all columns", X), ("without missing-indicator info (filled)", X.fillna(X.median(numeric_only=True)))):
        p = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1, n_jobs=12), XX, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        print(f"Task {t} ({len(tr)} train / {len(te)} test): adversarial AUC {roc_auc_score(y, p):.3f}  [{name}]", flush=True)
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1, n_jobs=12).fit(X, y)
    print("   most used:", sorted(zip(m.booster_.feature_importance("gain").round(0), X.columns), reverse=True)[:5])
