"""Round 2 - V5: V4 with Task 4 re-encoded by the first search guess (pube <-> roblock swapped vs alphabetical).
Local file only; nothing is uploaded."""
from pathlib import Path
import pandas as pd
OUT = Path(__file__).resolve().parent
v3 = pd.read_csv(OUT / "submission_v3_t5_groups.csv")        # Task 4 as class names
v4 = pd.read_csv(OUT / "submission_v4_t4_int.csv")
g1 = {"7-11k": 0, "fiveN": 1, "freefried": 2, "pube": 4, "roblock": 3}
v5 = v4.copy(); v5["task4"] = v3.task4.map(g1).astype(int)
assert [c for c in v4.columns if c != "task4" and (v4[c].astype(str) != v5[c].astype(str)).any()] == []
diff = int((v4.task4 != v5.task4).sum())
v5.to_csv(OUT / "submission_v5_t4_guess1.csv", index=False)
print("rows", v5.shape, "| only task4 differs from V4:", diff, "rows (pube and roblock predictions swapped) | counts:", v5.task4.value_counts().sort_index().to_dict())
