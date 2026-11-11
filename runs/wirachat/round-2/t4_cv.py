"""Round 2 - Task 4: honest validation of embedding classifiers.
(a) random 5-fold, (b) cluster-grouped 5-fold (whole clusters of similar training images held out = unseen content),
(c) robustness to grayscale / hue-shifted images, (d) near-duplicate structure of the training set. Training images only."""
import sys, argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent
D = OUT.parents[2] / "cpe342-kaggle-main" / "04_image_classification"
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="facebook/dinov2-small"); ap.add_argument("--C", type=float, default=1.0)
a = ap.parse_args(); tag = a.model.replace("/", "_")
tr = pd.read_csv(D / "train.csv"); classes = sorted(tr["class"].unique()); y = tr["class"].map({c: i for i, c in enumerate(classes)}).values
X = np.load(OUT / "emb" / f"{tag}__orig__train.npy"); print("embeddings", X.shape)
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)


def head(C=a.C):
    return LogisticRegression(C=C, max_iter=300, n_jobs=8)


def cv(groups=None, label=""):
    oof = np.zeros(len(y), int)
    splits = StratifiedKFold(5, shuffle=True, random_state=1).split(X, y) if groups is None else GroupKFold(5).split(X, y, groups)
    for tri, tei in splits:
        sc = StandardScaler().fit(X[tri]); m = head().fit(sc.transform(X[tri]), y[tri]); oof[tei] = m.predict(sc.transform(X[tei]))
    print(f"{label:34s} accuracy {accuracy_score(y, oof):.4f} | macro F1 {f1_score(y, oof, average='macro'):.4f} | per-class F1 {np.round(f1_score(y, oof, average=None), 3).tolist()}", flush=True)


cv(None, "random 5-fold")
km = MiniBatchKMeans(300, random_state=1, n_init=3, batch_size=4096).fit(Xn); groups = km.labels_
print("cluster sizes: min/median/max", np.min(np.bincount(groups)), int(np.median(np.bincount(groups))), np.max(np.bincount(groups)))
cv(groups, "cluster-grouped 5-fold (k=300)")
km2 = MiniBatchKMeans(60, random_state=1, n_init=3, batch_size=4096).fit(Xn)
cv(km2.labels_, "cluster-grouped 5-fold (k=60, harder)")

# near-duplicate structure: similarity of each image to its nearest OTHER training image
rng = np.random.default_rng(0); idx = rng.choice(len(y), 4000, replace=False)
S = Xn[idx] @ Xn.T; S[np.arange(len(idx)), idx] = -1
nn = S.argmax(1); sim = S.max(1)
print(f"nearest-other-image cosine similarity quantiles .1/.5/.9/.99: {np.round(np.quantile(sim, [.1, .5, .9, .99]), 3).tolist()} | same label as nearest neighbour: {(y[nn] == y[idx]).mean():.3f}")

# robustness: gray / hue variants of a subset (head trained on the rest)
sub_p = OUT / "emb" / "subset_idx.npy"
if sub_p.exists():
    sidx = np.load(sub_p); rest = np.setdiff1d(np.arange(len(y)), sidx)
    sc = StandardScaler().fit(X[rest]); m = head().fit(sc.transform(X[rest]), y[rest])
    for v in ("orig", "gray", "hue"):
        p = OUT / "emb" / f"{tag}__{v}__trainsubset.npy"
        if p.exists():
            Xs = np.load(p); pr = m.predict(sc.transform(Xs))
            print(f"robustness [{v:4s}] on {len(sidx)} held-out train images: accuracy {accuracy_score(y[sidx], pr):.4f} | macro F1 {f1_score(y[sidx], pr, average='macro'):.4f}")
