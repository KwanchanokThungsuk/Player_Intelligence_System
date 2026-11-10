"""Round 2 - Task 4: how do TEST images differ from TRAIN images? (image statistics only; no hand labelling, no predictions)"""
import os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
warnings.filterwarnings("ignore")
D = Path(__file__).resolve().parents[3] / "cpe342-kaggle-main" / "04_image_classification"
tr = pd.read_csv(D / "train.csv"); te = pd.read_csv(D / "test.csv")
rng = np.random.default_rng(1)


def stats(split, ids, n=3000):
    ids = rng.choice(np.asarray(ids), size=min(n, len(ids)), replace=False)
    rows = []
    for s in ids:
        with Image.open(D / split / f"{s}.jpg") as im:
            a = np.asarray(im.convert("RGB"), dtype=np.float32)
        g = a.mean(2)
        lap = (g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
        mx = a.max(2); mn = a.min(2)
        rows.append(dict(id=s, bright=g.mean(), contrast=g.std(), p1=np.percentile(g, 1), p99=np.percentile(g, 99), sharp=lap.var(),
                         sat=float((mx - mn).mean()), dark=float((g < 40).mean()), top=g[:20].mean(), bottom=g[-20:].mean(),
                         left=g[:, :20].mean(), right=g[:, -20:].mean()))
    return pd.DataFrame(rows)


A = stats("train", tr.id.astype(str)).merge(tr.assign(id=tr.id.astype(str)), on="id")
B = stats("test", te.id.astype(str))
cols = ["bright", "contrast", "p1", "p99", "sharp", "sat", "dark", "top", "bottom", "left", "right"]
print("image statistics (mean | p10 / p50 / p90):")
for c in cols:
    print(f"  {c:9s} train {A[c].mean():8.2f} | {A[c].quantile(.1):7.2f} / {A[c].quantile(.5):7.2f} / {A[c].quantile(.9):7.2f}    test {B[c].mean():8.2f} | {B[c].quantile(.1):7.2f} / {B[c].quantile(.5):7.2f} / {B[c].quantile(.9):7.2f}")
print("\ntrain brightness by class:", A.groupby("class").bright.mean().round(1).to_dict())
print("train sharpness by class:", A.groupby("class").sharp.mean().round(0).to_dict())
print("share of test images darker than the train 10th percentile of brightness:", round(float((B.bright < A.bright.quantile(.1)).mean()), 3))
# is there a bimodal split in test (e.g., a darkened subset)?
h, e = np.histogram(B.bright, bins=24); hA, _ = np.histogram(A.bright, bins=e)
print("brightness histogram test:", ", ".join(f"{a:.0f}:{c}" for a, c in zip(e[:-1], h)))
print("brightness histogram train:", ", ".join(f"{a:.0f}:{c}" for a, c in zip(e[:-1], hA)))
