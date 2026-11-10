"""Round 2 - V8: V4 (Task 4 = integer ids per datacard, test.csv order) with the improved Task 3 blend
(0.5 two-stage lognormal-median model + 0.5 L1 LightGBM with exact-identity features). Local file only; nothing is uploaded."""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
R2 = Path(__file__).resolve().parent; R1 = R2.parent / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load, nmae_score
y = load(3, "train").spending_30d.values
names = ["l1_raw", "A_lr08_l63", "B_lr08_l127", "C_lr08_l31", "D_lr05_l255", "E_lr08_l15"]
base_oof = np.median([np.load(R1 / f"task3_oof_{n}.npy") for n in names], axis=0); base_te = np.median([np.load(R1 / f"task3_test_{n}.npy") for n in names], axis=0)
two_oof, two_te = np.load(R2 / "t3_twostage_oof.npy"), np.load(R2 / "t3_twostage_test.npy")
fe_oof, fe_te = np.load(R2 / "t3_fe2_oof.npy"), np.load(R2 / "t3_fe2_test.npy")
best = max(((nmae_score(y, wb * base_oof + wt * two_oof + wf * fe_oof), wb, wt, wf) for wb, wt, wf in itertools.product(np.arange(0, 1.01, .25), repeat=3) if abs(wb + wt + wf - 1) < 1e-9))
s, wb, wt, wf = best
print(f"best blend OOF NMAE {s:.4f} with weights base {wb}, two-stage {wt}, FE2 {wf}")
v4 = pd.read_csv(R2 / "submission_v4_t4_int.csv")
v8 = v4.copy(); v8["task3"] = np.clip(wb * base_te + wt * two_te + wf * fe_te, 0, 300000)
assert [c for c in v4.columns if c != "task3" and (v4[c].astype(str) != v8[c].astype(str)).any()] == []
v8.to_csv(R2 / "submission_v8_t3_blend.csv", index=False); print("wrote submission_v8_t3_blend.csv")
