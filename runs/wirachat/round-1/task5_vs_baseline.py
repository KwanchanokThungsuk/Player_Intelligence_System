import sys
import numpy as np, pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from r1_common import load, ROOT, OUT
te = load(5, "test")
S = pd.read_pickle(OUT / "task5_S.pkl")["S"]
base = pd.read_csv(ROOT / "predictions" / "task5_test_predictions.csv")
assert base.id.tolist() == te.player_id.tolist()
b = base.task5.values
for thr in (110, 130, 180):
    a = (S > thr).fillna(False).values.astype(int)
    print(f"S>{thr}: flagged {a.sum()} ({a.mean():.3%}) | baseline flagged {b.sum()} | overlap {int((a & b).sum())}")
    # pretend our set is truth: how would the baseline score?
    p, r = precision_score(a, b), recall_score(a, b)
    print(f"   baseline vs this set as truth: precision={p:.3f} recall={r:.3f} F3(pos)={fbeta_score(a, b, beta=3):.3f} macro F3={fbeta_score(a, b, beta=3, average='macro'):.3f}")
