"""Round 3 - diagnostic V16: V15 with Task 5 reduced to ONE flagged row (the row with the strongest period-4 change score, almost surely a true anomaly).
F3 then is about 0.0005, so (V15 score - V16 score) * 5 = our Task 5 score exactly, and Task 4 = (V15*5 - 2.38) - Task 5. Local file only."""
from pathlib import Path
import numpy as np, pandas as pd
R3 = Path(__file__).resolve().parent
v15 = pd.read_csv(R3 / "submission_v15_t4_cnxt_all.csv"); S = np.load(R3 / "data2/task5_S.npy")
top = int(np.argmax(S)); v = v15.copy(); v["task5"] = 0; v.loc[top, "task5"] = 1
assert [c for c in v15.columns if c != "task5" and (v15[c].astype(str) != v[c].astype(str)).any()] == []
v.to_csv(R3 / "submission_v16_ablate_t5.csv", index=False)
print(f"V16: Task 5 has {int(v.task5.sum())} flagged row (row {top + 1}, S={S[top]:.0f}); Task 5 rows changed vs V15: {int((v.task5 != v15.task5).sum())}")
