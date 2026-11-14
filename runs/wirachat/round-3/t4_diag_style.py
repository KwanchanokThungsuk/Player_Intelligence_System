"""Round 3 - Task 4 diagnosis: low-level STYLE statistics of every train/test image (no content): black-border fraction (rotation), brightness, saturation, noise energy,
clipping, colourfulness, hue histogram. Then: (a) how well does style alone predict the train class? (b) how different are train and test styles (adversarial AUC)?"""
import sys, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
from PIL import Image
warnings.filterwarnings("ignore")
R3 = Path(__file__).resolve().parent; D = R3.parents[2] / "cpe342-kaggle-main" / "04_image_classification"
def feats(path):
    im = Image.open(path).convert("RGB"); a = np.asarray(im).astype(np.float32); g = a.mean(2)
    hsv = np.asarray(im.convert("HSV")).astype(np.float32)
    black = (a.max(2) < 8); bf = black.mean()
    # fraction of black connected to the 4 border strips (rotation corners) vs interior black
    border = np.zeros_like(black); border[:6, :] = True; border[-6:, :] = True; border[:, :6] = True; border[:, -6:] = True
    hp = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    core = ~black[1:-1, 1:-1]; noise = np.abs(hp[core]).mean() if core.sum() > 50 else 0.0
    sat = hsv[..., 1][~black]; val = hsv[..., 2][~black]; hue = hsv[..., 0][~black]
    rg = a[..., 0] - a[..., 1]; yb = 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]; colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    ch = [a[..., c][~black] for c in range(3)] if (~black).sum() > 50 else [np.zeros(1)] * 3
    hh = np.histogram(hue, bins=12, range=(0, 256))[0] / max(1, hue.size) if hue.size else np.zeros(12)
    corners = [black[:8, :8].mean(), black[:8, -8:].mean(), black[-8:, :8].mean(), black[-8:, -8:].mean()]
    return [bf, black[border].mean() if border.any() else 0, *corners, g[~black].mean() if (~black).any() else 0, g[~black].std() if (~black).any() else 0,
            sat.mean() if sat.size else 0, sat.std() if sat.size else 0, val.mean() if val.size else 0, noise, colorful,
            (a >= 250).mean(), (a <= 5).mean(), *[c.mean() for c in ch], *[c.std() for c in ch], *hh]
if __name__ == "__main__":
    tr = pd.read_csv(D / "train.csv"); te = pd.read_csv(D / "test.csv")
    paths = [D / "train" / f"{i}.jpg" for i in tr.id.astype(str)] + [D / "test" / f"{i}.jpg" for i in te.id.astype(str)]
    with ProcessPoolExecutor(12) as ex: F = np.array(list(ex.map(feats, paths, chunksize=256)), dtype=np.float32)
    np.save(R3 / "data3" / "t4_style_feats.npy", F); print("features", F.shape)
