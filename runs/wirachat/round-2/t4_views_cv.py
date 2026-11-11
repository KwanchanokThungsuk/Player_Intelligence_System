"""Round 2 - Task 4: colour-invariant head. Train on the union of original / gray / jitter views, validate on held-out CLUSTERS of training images
under each view, then predict test (inference only; no fitting on test images)."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent
D = OUT.parents[2] / "cpe342-kaggle-main" / "04_image_classification"
TAG = "facebook_dinov2-small"; VIEWS = ["orig", "gray", "jit", "jit2"]
tr = pd.read_csv(D / "train.csv"); classes = sorted(tr["class"].unique()); y = tr["class"].map({c: i for i, c in enumerate(classes)}).values
Etr = {v: np.load(OUT / "emb" / f"{TAG}__{v}__train.npy") for v in VIEWS}
Ete = {v: np.load(OUT / "emb" / f"{TAG}__{v}__test.npy") for v in VIEWS}
dev = torch.device("cuda")


def fit_linear(X, yy, balanced=False, wd=1e-3, steps=400, lr=0.02):
    X = torch.tensor(X, device=dev, dtype=torch.float32); yt = torch.tensor(yy, device=dev)
    W = torch.zeros(X.shape[1], 5, device=dev, requires_grad=True); b = torch.zeros(5, device=dev, requires_grad=True)
    w = None
    if balanced:
        cnt = torch.bincount(yt, minlength=5).float(); w = (cnt.sum() / (5 * cnt))
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad(); loss = torch.nn.functional.cross_entropy(X @ W + b, yt, weight=w) + wd * (W ** 2).sum(); loss.backward(); opt.step()
    return W.detach(), b.detach()


def proba(Wb, X):
    W, b = Wb
    with torch.no_grad():
        return torch.softmax(torch.tensor(X, device=dev, dtype=torch.float32) @ W + b, 1).cpu().numpy()


allX = np.concatenate([Etr[v] for v in VIEWS]); mu, sd = allX.mean(0), allX.std(0) + 1e-6
Z = lambda X: (X - mu) / sd
Ztr = {v: Z(Etr[v]) for v in VIEWS}; Zte = {v: Z(Ete[v]) for v in VIEWS}
Xn = Etr["orig"] / np.linalg.norm(Etr["orig"], axis=1, keepdims=True)
groups = MiniBatchKMeans(300, random_state=1, n_init=3, batch_size=4096).fit(Xn).labels_

res = {}
for name, views, bal in [("orig-only", ["orig"], False), ("multi-view", VIEWS, False), ("multi-view balanced", VIEWS, True)]:
    sc = {v: [] for v in VIEWS}
    for a, b in GroupKFold(5).split(y, y, groups):
        Xt = np.concatenate([Ztr[v][a] for v in views]); yt = np.concatenate([y[a]] * len(views))
        m = fit_linear(Xt, yt, balanced=bal)
        for v in VIEWS:
            p = proba(m, Ztr[v][b]).argmax(1); sc[v].append((accuracy_score(y[b], p), f1_score(y[b], p, average="macro")))
    res[name] = {v: np.mean(s, axis=0) for v, s in sc.items()}
    print(f"{name:20s} (held-out clusters) " + " | ".join(f"{v}: acc {r[0]:.3f} F1 {r[1]:.3f}" for v, r in res[name].items()), flush=True)

# final: multi-view head on ALL training images, predict test as the average over the 4 colour views
for name, bal in [("multi-view", False), ("multi-view balanced", True)]:
    m = fit_linear(np.concatenate([Ztr[v] for v in VIEWS]), np.concatenate([y] * len(VIEWS)), balanced=bal)
    P = np.mean([proba(m, Zte[v]) for v in VIEWS], axis=0)
    tag = "bal" if bal else "mv"
    np.save(OUT / f"t4_{tag}_test_prob.npy", P)
    pred = P.argmax(1)
    resnet = pd.read_csv(OUT / "submission_v3_t5_groups.csv").task4.map({c: i for i, c in enumerate(classes)}).values
    dino = np.load(OUT / "t4_dino_small_test_prob.npy").argmax(1)
    print(f"[{name}] test class shares {dict(zip(classes, np.round(np.bincount(pred, minlength=5) / len(pred), 3)))} | mean max-prob {P.max(1).mean():.3f} | agreement with ResNet {(pred == resnet).mean():.3f}, with DINOv2-orig {(pred == dino).mean():.3f}")
    per_view = [proba(m, Zte[v]).argmax(1) for v in VIEWS]
    print(f"   agreement between the test views' own predictions: orig-vs-gray {(per_view[0] == per_view[1]).mean():.3f}, orig-vs-jit {(per_view[0] == per_view[2]).mean():.3f}, jit-vs-jit2 {(per_view[2] == per_view[3]).mean():.3f}")
