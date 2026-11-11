"""Round 2 - Task 4: grayscale-only head (no colour information) and a colour-style check of the test predictions.
Training images only for fitting; test images are only predicted (inference) and described by simple colour statistics."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent
D = OUT.parents[2] / "cpe342-kaggle-main" / "04_image_classification"
TAG = "facebook_dinov2-small"
tr = pd.read_csv(D / "train.csv"); te = pd.read_csv(D / "test.csv")
classes = sorted(tr["class"].unique()); y = tr["class"].map({c: i for i, c in enumerate(classes)}).values
G_tr, G_te = np.load(OUT / "emb" / f"{TAG}__gray__train.npy"), np.load(OUT / "emb" / f"{TAG}__gray__test.npy")
O_tr = np.load(OUT / "emb" / f"{TAG}__orig__train.npy")
dev = torch.device("cuda")


def fit_linear(X, yy, wd=1e-3, steps=400, lr=0.02):
    X = torch.tensor(X, device=dev, dtype=torch.float32); yt = torch.tensor(yy, device=dev)
    W = torch.zeros(X.shape[1], 5, device=dev, requires_grad=True); b = torch.zeros(5, device=dev, requires_grad=True)
    cnt = torch.bincount(yt, minlength=5).float(); w = cnt.sum() / (5 * cnt)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad(); (torch.nn.functional.cross_entropy(X @ W + b, yt, weight=w) + wd * (W ** 2).sum()).backward(); opt.step()
    return W.detach(), b.detach()


def proba(m, X):
    with torch.no_grad():
        return torch.softmax(torch.tensor(X, device=dev, dtype=torch.float32) @ m[0] + m[1], 1).cpu().numpy()


mu, sd = G_tr.mean(0), G_tr.std(0) + 1e-6
Ztr, Zte = (G_tr - mu) / sd, (G_te - mu) / sd
Xn = O_tr / np.linalg.norm(O_tr, axis=1, keepdims=True)
groups = MiniBatchKMeans(300, random_state=1, n_init=3, batch_size=4096).fit(Xn).labels_
oof = np.zeros(len(y), int)
for a, b in GroupKFold(5).split(y, y, groups):
    oof[b] = proba(fit_linear(Ztr[a], y[a]), Ztr[b]).argmax(1)
print(f"gray-only head, held-out clusters: accuracy {accuracy_score(y, oof):.4f} | macro F1 {f1_score(y, oof, average='macro'):.4f} | per-class F1 {np.round(f1_score(y, oof, average=None), 3).tolist()}")
m = fit_linear(Ztr, y); P = proba(m, Zte); pred = P.argmax(1); np.save(OUT / "t4_gray_test_prob.npy", P)
print("gray-only test class shares:", dict(zip(classes, np.round(np.bincount(pred, minlength=5) / len(pred), 3))), "| mean max-prob", round(float(P.max(1).mean()), 3))
resnet = pd.read_csv(OUT / "submission_v3_t5_groups.csv").task4.map({c: i for i, c in enumerate(classes)}).values
dino = np.load(OUT / "t4_dino_small_test_prob.npy").argmax(1); mv = np.load(OUT / "t4_bal_test_prob.npy").argmax(1)
print(f"agreement of the GRAY-only predictions with: ResNet {(pred == resnet).mean():.3f} | DINOv2 original {(pred == dino).mean():.3f} | DINOv2 multi-view {(pred == mv).mean():.3f}")

# colour style per class: mean saturation of every train image (by TRUE label) vs every test image (by predicted label)
def sat(split, ids):
    out = np.empty(len(ids), np.float32)
    for k, s in enumerate(ids):
        with Image.open(D / split / f"{s}.jpg") as im:
            a = np.asarray(im.convert("RGB"), dtype=np.float32)
        out[k] = (a.max(2) - a.min(2)).mean()
    return out
cache = OUT / "t4_sat.npz"
if cache.exists():
    z = np.load(cache); s_tr, s_te = z["tr"], z["te"]
else:
    s_tr, s_te = sat("train", tr.id.astype(str).values), sat("test", te.id.astype(str).values); np.savez(cache, tr=s_tr, te=s_te)
print("\nmean saturation per class: TRAIN (true label) vs TEST (predicted label by each model)")
print(f"{'class':10s} {'train':>7s} | " + " | ".join(f"{n:>11s}" for n in ["gray-only", "ResNet", "DINO orig", "DINO mv"]))
for c in range(5):
    row = [s_te[p == c].mean() for p in (pred, resnet, dino, mv)]
    print(f"{classes[c]:10s} {s_tr[y == c].mean():7.1f} | " + " | ".join(f"{v:11.1f}" for v in row))
print("overall mean saturation: train", round(float(s_tr.mean()), 1), "| test", round(float(s_te.mean()), 1))
