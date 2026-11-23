"""Round 3 - V14b: V14 with (a) the improved Task 3 blend and (b) two extra Task 5 flags. Tasks 1, 2 are untouched; Task 4 stays as in V14 (rebuilt afterwards by build_round3_t4_variant.py with T4_BASE=submission_v14b_t3fe_t5tail.csv).
Task 3 blend: 0.55 * two-stage (FE features, data2/t3_twostage_fe_test.npy) + 0.25 * two-stage (raw features, round-2/t3_twostage_test.npy) + 0.20 * L1 LightGBM with identity features (round-2/t3_fe2_test.npy), clipped to [0, 300000]. OOF NMAE 0.79071 (V14 blend 0.78962).
Task 5 extra flags: rows not flagged in V14 whose login location jumps by more than 20 degrees between two periods AND whose onset score S34 (mean of periods 3-4 minus mean of periods 1-2, robust z over 29 base features, clipped at 8) exceeds 200,
the only two rows far beyond the smooth tail of everything else (S34 243 and 237; the next row is 188). Both are rows with 13-14 missing values, which is why the period-4 gap rule missed them. Unsupervised, fitted on the Task 5 test features only. Local file only."""
import os, re, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
os.environ.setdefault("CPE342_DATA", "C:/Users/Wirachat/Desktop/CPE342-DATA-v2/kaggle/cpe342-kaggle-main")
R3 = Path(__file__).resolve().parent; R2 = R3.parent / "round-2"
sys.path.insert(0, str(R3.parent / "round-1"))
from r1_common import load
v14 = pd.read_csv(R3 / "submission_v14_full_new_data.csv"); out = v14.copy()
# ---- Task 3
bl = 0.55 * np.load(R3 / "data2/t3_twostage_fe_test.npy") + 0.25 * np.load(R2 / "t3_twostage_test.npy") + 0.20 * np.load(R2 / "t3_fe2_test.npy")
out["task3"] = np.clip(bl, 0, 300000)
print(f"Task 3: mean {out.task3.mean():.1f} (V14 {v14.task3.mean():.1f}) | median abs change vs V14 {np.median(np.abs(out.task3 - v14.task3)):.2f} | rows with 0: {(out.task3 == 0).sum()} (V14 {(v14.task3 == 0).sum()})")
# ---- Task 5
X = load(5, "test").drop(columns=["player_id"]); flags = np.load(R3 / "data2/task5_flags.npy").astype(bool); assert (flags == v14.task5.values.astype(bool)).all()
bases = sorted({re.sub(r"_[1-4]$", "", c) for c in X.columns if re.search(r"_[1-4]$", c)})
def rz(s):
    med, iqr = s.median(), s.quantile(.75) - s.quantile(.25); return (s - med) / (iqr / 1.349 if iqr > 0 else (s.std() or 1))
d = pd.DataFrame({b: (X[f"{b}_3"] + X[f"{b}_4"]) / 2 - (X[f"{b}_1"] + X[f"{b}_2"]) / 2 for b in bases}).apply(rz)
S34 = ((d.clip(-8, 8) ** 2).sum(1, min_count=1) / d.notna().sum(1) * d.shape[1]).fillna(0).values
lat = X[[f"login_lat_{p}" for p in (1, 2, 3, 4)]].values; lon = X[[f"login_lon_{p}" for p in (1, 2, 3, 4)]].values
step = np.nan_to_num(np.column_stack([np.hypot(lat[:, t + 1] - lat[:, t], lon[:, t + 1] - lon[:, t]) for t in range(3)]), nan=0).max(1)
add = np.where(~flags & (step > 20) & (S34 > 200))[0]
print("Task 5 extra flag rows (0-based index):", add.tolist(), "| S34", np.round(S34[add], 1).tolist(), "| max location step (deg)", np.round(step[add], 1).tolist())
assert len(add) == 2, add
out.loc[add, "task5"] = 1
print("Task 5 flags:", int(v14.task5.sum()), "->", int(out.task5.sum()))
assert out.id.tolist() == v14.id.tolist() and np.isfinite(out.task3).all() and (out.task1 == v14.task1).all() and (out.task2 == v14.task2).all() and (out.task4 == v14.task4).all()
out.to_csv(R3 / "submission_v14b_t3fe_t5tail.csv", index=False); print("wrote submission_v14b_t3fe_t5tail.csv")
