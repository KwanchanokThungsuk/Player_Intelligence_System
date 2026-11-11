"""Round 2 - V10 candidate: V8 with Task 4 from the fine-tuned ConvNeXt-Tiny (strong class-independent augmentation), datacard encoding, test.csv order.
Local file only; nothing is uploaded."""
from pathlib import Path
import numpy as np, pandas as pd
R2 = Path(__file__).resolve().parent
v8 = pd.read_csv(R2 / "submission_v8_t3_blend.csv")
P = np.load(R2 / "t4_cnxt_test_prob.npy")                 # columns = alphabetical class order = datacard order 0..4
assert P.shape == (len(v8), 5)
v10 = v8.copy(); v10["task4"] = P.argmax(1).astype(int)
assert [c for c in v8.columns if c != "task4" and (v8[c].astype(str) != v10[c].astype(str)).any()] == []
v10.to_csv(R2 / "submission_v10_t4_convnext.csv", index=False)
prev = pd.read_csv(R2 / "submission_v9_t4_dino_mv.csv")
print("rows", v10.shape, "| task4 rows differing from V8 (ResNet):", int((v8.task4 != v10.task4).sum()), "| from V9 (DINOv2):", int((prev.task4 != v10.task4).sum()))
print("class shares:", v10.task4.value_counts(normalize=True).sort_index().round(3).to_dict())
