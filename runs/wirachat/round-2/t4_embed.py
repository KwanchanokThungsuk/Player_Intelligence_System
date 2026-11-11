"""Round 2 - Task 4: frozen pretrained embeddings (inference only; no training on test images).
Usage: python t4_embed.py --model facebook/dinov2-small [--variant orig|gray|hue] [--subset 6000]
Output: runs/wirachat/round-2/emb/<model>__<variant>__{train,test}.npy  (CLS + mean patch token, averaged over horizontal flip)"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader

OUT = Path(__file__).resolve().parent
D = OUT.parents[2] / "cpe342-kaggle-main" / "04_image_classification"
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
SIZE = (294, 154)  # (W, H): multiples of 14, keeps the 230x120 aspect ratio


def hue_shift(im, seed):
    r = np.random.default_rng(seed)
    h, s, v = im.convert("HSV").split()
    sh = int(r.integers(30, 225))
    h = Image.fromarray(((np.asarray(h, dtype=np.int32) + sh) % 256).astype(np.uint8))
    return Image.merge("HSV", (h, s, v)).convert("RGB")


class Imgs(Dataset):
    def __init__(self, split, ids, variant):
        self.split, self.ids, self.variant = split, list(ids), variant
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        with Image.open(D / self.split / f"{self.ids[i]}.jpg") as im:
            im = im.convert("RGB")
            if self.variant == "gray": im = im.convert("L").convert("RGB")
            elif self.variant == "hue": im = hue_shift(im, i)
            elif self.variant in ("jit", "jit2"):           # strong random colour jitter: hue shift + saturation + brightness
                r = np.random.default_rng(i + (0 if self.variant == "jit" else 777777))
                im = hue_shift(im, int(r.integers(0, 10**6)))
                im = ImageEnhance.Color(im).enhance(float(r.uniform(0.15, 2.4)))
                im = ImageEnhance.Brightness(im).enhance(float(r.uniform(0.7, 1.3)))
            a = np.asarray(im.resize(SIZE, Image.BICUBIC), dtype=np.float32) / 255.0
        a = (a - MEAN) / STD
        return torch.from_numpy(a.transpose(2, 0, 1))


@torch.no_grad()
def embed(model, ds, bs=128, workers=6):
    dl = DataLoader(ds, batch_size=bs, num_workers=workers, shuffle=False, pin_memory=True)
    outs = []
    for x in dl:
        x = x.cuda(non_blocking=True)
        feats = []
        for xx in (x, torch.flip(x, dims=[3])):
            with torch.autocast("cuda", dtype=torch.float16):
                h = model(pixel_values=xx).last_hidden_state
            feats.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1).float())
        outs.append(((feats[0] + feats[1]) / 2).cpu().numpy())
    return np.concatenate(outs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/dinov2-small"); ap.add_argument("--variant", default="orig", choices=["orig", "gray", "hue", "jit", "jit2"])
    ap.add_argument("--subset", type=int, default=0, help="only embed a random subset of train (for robustness variants); skips test")
    a = ap.parse_args()
    from transformers import AutoModel
    model = AutoModel.from_pretrained(a.model).cuda().eval()
    tag = a.model.replace("/", "_")
    (OUT / "emb").mkdir(exist_ok=True)
    tr = pd.read_csv(D / "train.csv"); te = pd.read_csv(D / "test.csv")
    t0 = time.time()
    if a.subset:
        idx = np.random.default_rng(0).choice(len(tr), a.subset, replace=False); np.save(OUT / "emb" / "subset_idx.npy", idx)
        E = embed(model, Imgs("train", tr.id.astype(str).values[idx], a.variant))
        np.save(OUT / "emb" / f"{tag}__{a.variant}__trainsubset.npy", E); print("subset", E.shape, f"{time.time()-t0:.0f}s")
    else:
        E = embed(model, Imgs("train", tr.id.astype(str).values, a.variant)); np.save(OUT / "emb" / f"{tag}__{a.variant}__train.npy", E); print("train", E.shape, f"{time.time()-t0:.0f}s", flush=True)
        E = embed(model, Imgs("test", te.id.astype(str).values, a.variant)); np.save(OUT / "emb" / f"{tag}__{a.variant}__test.npy", E); print("test", E.shape, f"{time.time()-t0:.0f}s")
