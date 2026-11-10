"""Round 2 - candidate V4: V3 with Task 4 written as integer class ids (datacard order = alphabetical label order).
Local file only; nothing is uploaded."""
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parent
v3 = pd.read_csv(OUT / "submission_v3_t5_groups.csv")
mapping = {"7-11k": 0, "fiveN": 1, "freefried": 2, "pube": 3, "roblock": 4}   # datacard: 0 7-11 Knights, 1 FiveN, 2 Free Fried, 3 PUBE, 4 Roblock
v4 = v3.copy(); v4["task4"] = v3.task4.map(mapping)
assert v4.task4.notna().all() and set(v4.task4.unique()) <= {0, 1, 2, 3, 4}
v4["task4"] = v4.task4.astype(int)
changed = [c for c in v3.columns if c != "task4" and (v3[c].astype(str) != v4[c].astype(str)).any()]
assert changed == [], changed
assert (v4.task4.map({v: k for k, v in mapping.items()}) == v3.task4).all()
v4.to_csv(OUT / "submission_v4_t4_int.csv", index=False)
print("rows", v4.shape, "| only task4 changed | task4 class counts:", v4.task4.value_counts().sort_index().to_dict())
print(v4.head(3).to_string(index=False))
