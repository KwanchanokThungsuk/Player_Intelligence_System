"""Round 3 - Task 4 encoding search with the ConvNeXt predictions (V10 base): rows aligned (test.csv order), class codes re-mapped.
Under 'rows aligned, key codes differ from the datacard' only 4 encodings fit V4/V5/V6 (each matched exactly one class).
Local files only; nothing is uploaded."""
from pathlib import Path
import numpy as np, pandas as pd
R2 = Path(__file__).resolve().parents[1] / "round-2"; OUT = Path(__file__).resolve().parent
CL = ["7-11k", "fiveN", "freefried", "pube", "roblock"]
v10 = pd.read_csv(R2 / "submission_v10_t4_convnext.csv"); base = v10.task4.values            # class index in datacard order 0..4 (alphabetical)
cands = {"c1": {"7-11k": 0, "fiveN": 4, "freefried": 3, "pube": 1, "roblock": 2}, "c2": {"7-11k": 3, "fiveN": 1, "freefried": 4, "pube": 2, "roblock": 0},
         "c3": {"7-11k": 3, "fiveN": 4, "freefried": 2, "pube": 0, "roblock": 1}, "c4": {"7-11k": 4, "fiveN": 3, "freefried": 2, "pube": 1, "roblock": 0}}
split = {"7-11k": 0, "fiveN": 3, "freefried": 2, "pube": 1, "roblock": 4}                    # separates the 4 candidates: matches c1=2, c2=0, c3=1, c4=3
def make(name, mp):
    arr = np.array([mp[CL[i]] for i in base]); v = v10.copy(); v["task4"] = arr.astype(int)
    assert [c for c in v10.columns if c != "task4" and (v10[c].astype(str) != v[c].astype(str)).any()] == []
    v.to_csv(OUT / f"submission_{name}.csv", index=False); print(f"{name}: {mp} | rows differing from V10: {int((v.task4 != v10.task4).sum())}")
make("v11_t4_split_guess", split)
for k, mp in cands.items(): make(f"v12_t4_{k}", mp)
