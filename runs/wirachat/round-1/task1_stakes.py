import sys
import numpy as np
from sklearn.metrics import fbeta_score
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from r1_common import load, macro_fbeta, OUT
y = load(1, "train").dropna(subset=["is_cheater"]).is_cheater.astype(int).values
oof = (np.load(OUT / "task1_oof_raw.npy") + np.load(OUT / "task1_oof_fe.npy")) / 2
print("thr   binaryF2  macroF2  mean(both)")
best = (0, 0)
for t in (0.15, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.40, 0.44):
    b = fbeta_score(y, oof >= t, beta=2); m = macro_fbeta(y, oof >= t, 2)
    print(f"{t:.2f}  {b:.4f}   {m:.4f}   {(b + m) / 2:.4f}")
