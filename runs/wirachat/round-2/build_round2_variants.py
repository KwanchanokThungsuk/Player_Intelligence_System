"""Round 2 - diagnostic variants of the round-1 submission. Each variant changes exactly ONE task column.
Nothing is uploaded; files are local (CSV is excluded from git)."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load, macro_fbeta, OUT as R1_OUT
from sklearn.metrics import fbeta_score

OUT = Path(__file__).resolve().parent
base = pd.read_csv(R1_OUT / "submission.csv")
assert base.shape == (25889, 6)

# ---- Task 1 probabilities (blend of raw + fe LightGBM) and the OOF behaviour of each threshold
te1p = (np.load(R1_OUT / "task1_test_prob_raw.npy") + np.load(R1_OUT / "task1_test_prob_fe.npy")) / 2
y1 = load(1, "train").dropna(subset=["is_cheater"]).is_cheater.astype(int).values
oof1 = (np.load(R1_OUT / "task1_oof_raw.npy") + np.load(R1_OUT / "task1_oof_fe.npy")) / 2
assert ((te1p >= 0.40).astype(int) == base.task1.values).all(), "round-1 Task 1 column not reproduced"

# ---- Task 5 score from round 1
S = pd.read_pickle(R1_OUT / "task5_S.pkl")["S"].values
assert ((np.nan_to_num(S, nan=0) > 130).astype(int) == base.task5.values).all(), "round-1 Task 5 column not reproduced"


def write(name, col, values, note):
    v = base.copy(); v[col] = values
    changed = {c: int((v[c].astype(str) != base[c].astype(str)).sum()) for c in base.columns}
    assert sum(1 for c, n in changed.items() if n > 0) == 1 and changed[col] > 0, changed
    v.to_csv(OUT / name, index=False)
    print(f"{name}: changes only {col} ({changed[col]} rows) | {note}")


# V1: Task 1 threshold 0.18 (best for PLAIN F2) -> tells which metric Kaggle uses for Task 1
t = 0.18
write("submission_v1_t1_thr018.csv", "task1", (te1p >= t).astype(int),
      f"Task1 positive rate {(te1p >= t).mean():.3f} (round 1: {(te1p >= .4).mean():.3f}); OOF binaryF2 {fbeta_score(y1, oof1 >= t, beta=2):.4f}, macroF2 {macro_fbeta(y1, oof1 >= t, 2):.4f}")

# V2: Task 5 looser set (S > 80 instead of 130) -> does the true anomaly set extend below the gap?
loose = (np.nan_to_num(S, nan=0) > 80).astype(int)
write("submission_v2_t5_loose.csv", "task5", loose, f"Task5 flagged {int(loose.sum())} rows ({loose.mean():.2%}) vs {int(base.task5.sum())}")
json.dump(dict(base="round-1 submission.csv (Kaggle 0.54617)", v1="Task1 thr 0.18", v2="Task5 S>80"), open(OUT / "variants.json", "w"), indent=1)
