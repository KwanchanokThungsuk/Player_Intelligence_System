"""Round 2 - V9 candidate: V8 with Task 4 from the colour-robust multi-view DINOv2-small head (class-balanced), datacard encoding, test.csv order.
Local file only; nothing is uploaded."""
from pathlib import Path
import numpy as np, pandas as pd
R2 = Path(__file__).resolve().parent
v8 = pd.read_csv(R2 / "submission_v8_t3_blend.csv")
P = np.load(R2 / "t4_bal_test_prob.npy")                    # columns follow the alphabetical order = datacard order 0..4
assert P.shape == (len(v8), 5)
v9 = v8.copy(); v9["task4"] = P.argmax(1).astype(int)
assert [c for c in v8.columns if c != "task4" and (v8[c].astype(str) != v9[c].astype(str)).any()] == []
v9.to_csv(R2 / "submission_v9_t4_dino_mv.csv", index=False)
print("rows", v9.shape, "| task4 rows differing from V8:", int((v8.task4 != v9.task4).sum()), "| class shares:", (v9.task4.value_counts(normalize=True).sort_index().round(3)).to_dict())
