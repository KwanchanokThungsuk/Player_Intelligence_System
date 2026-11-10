"""Round 2 - Task 4 sanity check: do the baseline ResNet's test predictions agree with an independent method
(nearest neighbours on tiny image thumbnails)? Read-only."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
D = ROOT / "cpe342-kaggle-main" / "04_image_classification"
tr = pd.read_csv(D / "train.csv"); te = pd.read_csv(D / "test.csv")


def thumbs(split, ids):
    X = np.empty((len(ids), 32 * 16 * 3), dtype=np.float32)
    for i, s in enumerate(ids):
        with Image.open(D / split / f"{s}.jpg") as im:
            X[i] = np.asarray(im.convert("RGB").resize((32, 16), Image.BILINEAR), dtype=np.float32).ravel() / 255.0
    return X


cache = OUT / "t4_thumbs.npz"
if cache.exists():
    z = np.load(cache); Xtr, Xte = z["tr"], z["te"]
else:
    Xtr, Xte = thumbs("train", tr.id.astype(str)), thumbs("test", te.id.astype(str))
    np.savez_compressed(cache, tr=Xtr, te=Xte)
print("thumbnails:", Xtr.shape, Xte.shape, flush=True)
ytr = tr["class"].values
# 1) is a tiny-thumbnail kNN informative at all? (5-fold on a 8000-row subsample of train)
rng = np.random.default_rng(1); idx = rng.choice(len(Xtr), 8000, replace=False)
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=8)
p = cross_val_predict(knn, Xtr[idx], ytr[idx], cv=StratifiedKFold(5, shuffle=True, random_state=1))
print(f"kNN-on-thumbnails train CV accuracy (8000 rows): {accuracy_score(ytr[idx], p):.3f}")
# 2) agreement between kNN labels for the test images and the baseline ResNet predictions, in test.csv order
knn.fit(Xtr, ytr); pk = knn.predict(Xte)
v3 = pd.read_csv(OUT / "submission_v3_t5_groups.csv")
resnet = v3.task4.values                                   # names, in test.csv order
print(f"agreement between kNN(test) and the submitted ResNet predictions: {(pk == resnet).mean():.3f}")
ids = te.id.astype(str).values
order = np.argsort(ids)
print(f"agreement if the ResNet predictions were in SORTED-id order instead: {(pk[order] == resnet[:len(order)]).mean():.3f}")
print("kNN-predicted test class shares:", pd.Series(pk).value_counts(normalize=True).round(3).to_dict())
print("ResNet predicted test class shares:", pd.Series(resnet).value_counts(normalize=True).round(3).to_dict())
# 3) the same agreement measured inside TRAIN (what we would expect if everything is aligned)
