"""Round 3 - Task 4 row-order variants with the ConvNeXt predictions and the datacard encoding (V10 base). Local files only; nothing is uploaded."""
from pathlib import Path
import numpy as np, pandas as pd
R2 = Path(__file__).resolve().parents[1] / "round-2"; OUT = Path(__file__).resolve().parent
D = Path(__file__).resolve().parents[3] / "cpe342-kaggle-main" / "04_image_classification"
v10 = pd.read_csv(R2 / "submission_v10_t4_convnext.csv"); pred = v10.task4.values; ids = pd.read_csv(D / "test.csv").id.astype(str).values
by_id = dict(zip(ids, pred))
def save(name, arr, note):
    v = v10.copy(); v["task4"] = np.asarray(arr).astype(int)
    assert [c for c in v10.columns if c != "task4" and (v10[c].astype(str) != v[c].astype(str)).any()] == []
    v.to_csv(OUT / f"submission_{name}.csv", index=False); print(f"{name}: {note} | rows differing from V10: {int((v.task4 != v10.task4).sum())}")
save("v13_t4_reversed", pred[::-1], "predictions in reversed test.csv order")
save("v14_t4_sorted_desc", [by_id[i] for i in sorted(ids, reverse=True)], "predictions in descending sorted-id order")
n = len(pred); counts = [n // 5 + (1 if i < n % 5 else 0) for i in range(5)]
save("v15_t4_classblocks", np.repeat(np.arange(5), counts), f"class-sorted blocks with balanced counts {counts} (tests a key sorted by label)")
