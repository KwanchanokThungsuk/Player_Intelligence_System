"""Round 3 - build a V14-style file with a different Task 4 (ensemble of one or more probability files in OLD test.csv row order).
Tasks 1,2,3,5 stay exactly as in V14. Usage: python build_round3_t4_variant.py <out_name> <probfile[:weight]> [<probfile[:weight]> ...]   (prob files are in runs/wirachat/round-2 or round-3)
Local file only; nothing is uploaded."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
R3 = Path(__file__).resolve().parent; R2 = R3.parent / "round-2"
OLD_TEST = Path(__file__).resolve().parents[3] / "cpe342-kaggle-main/04_image_classification/test.csv"
NEW_TEST = Path("C:/Users/Wirachat/Desktop/CPE342-DATA-v2/task4_new_csv/test.csv")
out = sys.argv[1]; specs = sys.argv[2:]
old_ids = pd.read_csv(OLD_TEST).id.astype(str).tolist(); new = pd.read_csv(NEW_TEST)
stems = new.file_name.str.replace(".jpg", "", regex=False).tolist()
P = 0; wsum = 0
for s in specs:
    name, _, w = s.partition(":"); w = float(w or 1)
    f = (R3 / name) if (R3 / name).exists() else (R2 / name)
    p = np.load(f); assert p.shape == (len(old_ids), 5), (f, p.shape); P = P + w * p; wsum += w
P = P / wsum
by_img = dict(zip(old_ids, P.argmax(1)))
v14 = pd.read_csv(R3 / "submission_v14_full_new_data.csv"); assert v14.id.tolist() == new.id.tolist()
v = v14.copy(); v["task4"] = np.array([by_img[s] for s in stems], dtype=int)
assert [c for c in v14.columns if c != "task4" and (v14[c].astype(str) != v[c].astype(str)).any()] == []
v.to_csv(R3 / f"submission_{out}.csv", index=False)
print(f"{out}: Task 4 rows differing from V14: {int((v.task4 != v14.task4).sum())} ({(v.task4 != v14.task4).mean():.1%}) | shares {np.round(np.bincount(v.task4, minlength=5) / len(v), 3).tolist()}")
