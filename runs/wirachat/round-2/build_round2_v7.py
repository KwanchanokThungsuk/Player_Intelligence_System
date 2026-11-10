"""Round 2 - V7: test whether the Task 4 key is in SORTED-id order instead of test.csv order.
Row i of the submission gets the prediction for the i-th id in ascending order (alphabetical encoding kept).
Local file only; nothing is uploaded."""
from pathlib import Path
import pandas as pd
OUT = Path(__file__).resolve().parents[0]
ROOT = OUT.parents[2]
v3 = pd.read_csv(OUT / "submission_v3_t5_groups.csv")            # Task 4 as names, in test.csv order
v4 = pd.read_csv(OUT / "submission_v4_t4_int.csv")
ids = pd.read_csv(ROOT / "cpe342-kaggle-main" / "04_image_classification" / "test.csv").id.astype(str)
assert len(ids) == len(v3) == 25889
m0 = {"7-11k": 0, "fiveN": 1, "freefried": 2, "pube": 3, "roblock": 4}
pred_by_id = dict(zip(ids, v3.task4.map(m0)))
sorted_ids = sorted(ids)
v7 = v4.copy(); v7["task4"] = [pred_by_id[i] for i in sorted_ids]
assert [c for c in v4.columns if c != "task4" and (v4[c].astype(str) != v7[c].astype(str)).any()] == []
assert v7.task4.value_counts().sort_index().to_dict() == v4.task4.value_counts().sort_index().to_dict()
v7.to_csv(OUT / "submission_v7_t4_sorted.csv", index=False)
print("rows", v7.shape, "| task4 rows differing from V4:", int((v4.task4 != v7.task4).sum()), "| class counts unchanged:", v7.task4.value_counts().sort_index().to_dict())
