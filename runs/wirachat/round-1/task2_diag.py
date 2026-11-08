import sys
import numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from r1_common import load, OUT
tr = load(2, "train"); y = tr.segment.astype(int).values
oof = np.load(OUT / "task2_oof_clean.npy")
pred = oof.argmax(1); conf = oof.max(1)
print("accuracy", round((pred == y).mean(), 4), "| macro F1", round(f1_score(y, pred, average="macro"), 4))
print("confusion (rows=true):\n", confusion_matrix(y, pred))
for lo, hi in [(0, .4), (.4, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)]:
    m = (conf >= lo) & (conf < hi)
    print(f"conf [{lo:.1f},{hi:.1f}): n={int(m.sum()):>6} acc={(pred[m] == y[m]).mean():.3f}")
# does missingness of a feature carry signal?
feat = [c for c in tr.columns if c not in ("player_id", "segment")]
nm = tr[feat].isna().sum(axis=1)
print("accuracy by #missing features:", pd.Series(pred == y).groupby(nm.clip(upper=8)).mean().round(3).to_dict())
print("rows by #missing:", nm.clip(upper=8).value_counts().sort_index().to_dict())
