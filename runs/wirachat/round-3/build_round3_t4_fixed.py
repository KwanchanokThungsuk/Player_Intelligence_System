"""Round 3 - V13: Task 4 predictions placed by the NEW test.csv (ANS id -> file_name), using the existing ConvNeXt-Tiny predictions (images are byte-identical).
Local file only; nothing is uploaded. Usage: python build_round3_t4_fixed.py <path-to-new-test.csv>"""
import sys, io, zipfile
from pathlib import Path
import numpy as np, pandas as pd
R2 = Path(__file__).resolve().parents[1] / "round-2"; OUT = Path(__file__).resolve().parent
OLD_TEST = Path(__file__).resolve().parents[3] / "cpe342-kaggle-main/04_image_classification/test.csv"
new = pd.read_csv(sys.argv[1]); old_ids = pd.read_csv(OLD_TEST).id.astype(str).tolist()
P = np.load(R2 / "t4_cnxt_test_prob.npy"); assert len(P) == len(old_ids)
pred_by_img = dict(zip(old_ids, P.argmax(1)))                    # keyed by image id (old test.csv rows)
stems = new.file_name.str.replace(".jpg", "", regex=False).tolist()
missing = [s for s in stems if s not in pred_by_img]
print("new test.csv rows", len(new), "| image names not found among our predictions:", len(missing))
sample = pd.read_csv(Path(__file__).resolve().parents[3] / "cpe342-kaggle-main/sample_submission.csv")
print("new test.csv id order equals the sample_submission ANS order:", new.id.tolist() == sample.id.tolist())
v10 = pd.read_csv(R2 / "submission_v10_t4_convnext.csv"); assert v10.id.tolist() == new.id.tolist()
v = v10.copy(); v["task4"] = np.array([pred_by_img[s] for s in stems], dtype=int)
assert [c for c in v10.columns if c != "task4" and (v10[c].astype(str) != v[c].astype(str)).any()] == []
v.to_csv(OUT / "submission_v13_t4_fixed.csv", index=False)
print("rows", v.shape, "| task4 rows differing from V10:", int((v.task4 != v10.task4).sum()), "| class shares:", v.task4.value_counts(normalize=True).sort_index().round(3).to_dict())
