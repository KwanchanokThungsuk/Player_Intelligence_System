"""Round 2 - integrity audit of Tasks 1, 2, 3, 5: column order, and an independent minimal re-implementation compared with the submitted predictions."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
warnings.filterwarnings("ignore")
R2 = Path(__file__).resolve().parent; R1 = R2.parent / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load
sub = pd.read_csv(R2 / "submission_v10_t4_convnext.csv")                 # latest file (Tasks 1,2,3,5 identical since V8/V3)
print("submission shape", sub.shape, "| id order equals sample:", (sub.id == pd.read_csv("cpe342-kaggle-main/sample_submission.csv").id).all())
for t, tgt in [(1, "is_cheater"), (2, "segment"), (3, "spending_30d")]:
    tr, te = load(t, "train"), load(t, "test")
    a = [c for c in tr.columns if c not in ("player_id", tgt)]; b = [c for c in te.columns if c != "player_id"]
    print(f"Task {t}: test columns identical in name AND order to train features: {a == b}")
te5 = load(5, "test"); print("Task 5 flagged rows in submission:", int(sub.task5.sum()), "| ids aligned with test.csv rows:", len(sub) == len(te5))

# ---- Task 1: independent HistGradientBoosting (select columns BY NAME), compare with submitted labels
tr, te = load(1, "train"), load(1, "test"); lab = tr.dropna(subset=["is_cheater"]); cols = [c for c in lab.columns if c not in ("player_id", "is_cheater")]
m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=1).fit(lab[cols], lab.is_cheater.astype(int))
p = m.predict_proba(te[cols])[:, 1]
mine = (p >= 0.18).astype(int)
print(f"Task 1: independent HGB flags {mine.mean():.3f} of test (submitted {sub.task1.mean():.3f}); agreement {(mine == sub.task1.values).mean():.3f}; corr of its probability with our blend rank {spearmanr(p, np.load(R1 / 'task1_test_prob_raw.npy')).statistic:.3f}")
# ---- Task 2: independent HGB with native categoricals
tr, te = load(2, "train"), load(2, "test"); cats = [c for c in tr.columns if c not in ("player_id", "segment") and not pd.api.types.is_numeric_dtype(tr[c])]
cols = [c for c in tr.columns if c not in ("player_id", "segment") and not c.startswith("random_metric")]
def prep(d):
    X = d[cols].copy()
    for c in cats:
        if c in X: X[c] = pd.Categorical(X[c], categories=sorted(tr[c].dropna().unique()))
    return X
m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, categorical_features="from_dtype", random_state=1).fit(prep(tr), tr.segment.astype(int))
pr = m.predict(prep(te))
print(f"Task 2: independent HGB vs submitted class agreement {(pr == sub.task2.values).mean():.3f} (two good models of 0.75 macro-F1 typically agree about 0.85-0.9)")
# ---- Task 3: independent HGB L1
tr, te = load(3, "train"), load(3, "test"); cols = [c for c in tr.columns if c not in ("player_id", "spending_30d")]
m = HistGradientBoostingRegressor(loss="absolute_error", max_iter=500, learning_rate=0.08, random_state=1).fit(tr[cols], tr.spending_30d)
q = np.clip(m.predict(te[cols]), 0, None)
print(f"Task 3: Spearman(independent HGB, submitted) = {spearmanr(q, sub.task3.values).statistic:.3f} | medians {np.median(q):.1f} vs {np.median(sub.task3):.1f} | sums {q.sum():.3g} vs {sub.task3.sum():.3g}")
