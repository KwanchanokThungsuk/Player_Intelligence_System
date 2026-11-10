"""Round 2 - V6: second search guess for the Task 4 encoding. Local file only; nothing is uploaded."""
from pathlib import Path
import pandas as pd
OUT = Path(__file__).resolve().parent
v3 = pd.read_csv(OUT / "submission_v3_t5_groups.csv")        # Task 4 as class names
v4 = pd.read_csv(OUT / "submission_v4_t4_int.csv")
g2 = {"7-11k": 1, "fiveN": 3, "freefried": 4, "pube": 0, "roblock": 2}
v6 = v4.copy(); v6["task4"] = v3.task4.map(g2).astype(int)
assert [c for c in v4.columns if c != "task4" and (v4[c].astype(str) != v6[c].astype(str)).any()] == []
v6.to_csv(OUT / "submission_v6_t4_guess2.csv", index=False)
print("rows", v6.shape, "| task4 rows differing from V4:", int((v4.task4 != v6.task4).sum()), "| counts:", v6.task4.value_counts().sort_index().to_dict())
