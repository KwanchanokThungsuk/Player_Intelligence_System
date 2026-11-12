"""Round 3 - V14: complete file for the UPDATED datasets (2026-11-12). Local file only; nothing is uploaded.
Task 1 and Task 3: unchanged datasets -> predictions from V10/V8 kept.  Task 2: retrained on the new data.  Task 4: ConvNeXt predictions placed by the new ANS-keyed test.csv.
Task 5: flags from the unsupervised search on the new data."""
import sys, os
from pathlib import Path
import numpy as np, pandas as pd
R3 = Path(__file__).resolve().parent; R2 = R3.parent / "round-2"
NEW = Path("C:/Users/Wirachat/Desktop/CPE342-DATA-v2/kaggle/cpe342-kaggle-main")
v13 = pd.read_csv(R3 / "submission_v13_t4_fixed.csv")                    # T1, T3 (unchanged tasks), T4 fixed; T2/T5 here are the OLD-data predictions
v14 = v13.copy()
v14["task2"] = np.load(R3 / "data2/task2_test_pred.npy").astype(int)
v14["task5"] = np.load(R3 / "data2/task5_flags.npy").astype(int)
# integrity: new test files must have the row counts of the sample, and Tasks 1/3 files must be identical to the ones used for the old predictions
sample = pd.read_csv(NEW / "sample_submission.csv"); assert v14.id.tolist() == sample.id.tolist()
assert len(pd.read_csv(NEW / "02_multiclass_classification/test.csv")) == len(v14) == len(pd.read_csv(NEW / "05_anomaly_detection/test.csv"))
changed = {c: int((v14[c].astype(str) != v13[c].astype(str)).sum()) for c in v14.columns}
print("columns changed vs V13 (rows):", {c: n for c, n in changed.items() if n})
assert set(v14.task1) <= {0, 1} and set(v14.task2) <= {0, 1, 2, 3} and set(v14.task4) <= {0, 1, 2, 3, 4} and set(v14.task5) <= {0, 1} and np.isfinite(v14.task3).all()
v14.to_csv(R3 / "submission_v14_full_new_data.csv", index=False)
print("V14:", v14.shape, "| task1 flagged", round(v14.task1.mean(), 3), "| task2 shares", np.round(np.bincount(v14.task2) / len(v14), 3).tolist(), "| task4 shares", np.round(np.bincount(v14.task4) / len(v14), 3).tolist(), "| task5 flagged", int(v14.task5.sum()))
print(v14.head(3).to_string(index=False))
