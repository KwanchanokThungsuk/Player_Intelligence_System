"""Round 2 - Task 4: independent hand-crafted-feature classifier (colour / layout / edges) to cross-check the ResNet test predictions."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent
D = OUT.parents[2] / "cpe342-kaggle-main" / "04_image_classification"
tr = pd.read_csv(D / "train.csv"); te = pd.read_csv(D / "test.csv")


def feats(path):
    with Image.open(path) as im:
        im = im.convert("RGB"); a = np.asarray(im, dtype=np.float32)
        hsv = np.asarray(im.convert("HSV"), dtype=np.float32)
    g = a.mean(2); H, W = g.shape
    out = []
    gh, gw = 4, 8
    for i in range(gh):
        for j in range(gw):
            blk = a[i * H // gh:(i + 1) * H // gh, j * W // gw:(j + 1) * W // gw]
            out += list(blk.mean((0, 1))) + [blk.std()]
            gb = g[i * H // gh:(i + 1) * H // gh, j * W // gw:(j + 1) * W // gw]
            out.append(np.abs(np.diff(gb, axis=1)).mean() + np.abs(np.diff(gb, axis=0)).mean())
    out += list(np.histogram(hsv[..., 0], bins=16, range=(0, 255))[0] / (H * W))
    out += list(np.histogram(hsv[..., 1], bins=8, range=(0, 255))[0] / (H * W))
    out += list(np.histogram(hsv[..., 2], bins=8, range=(0, 255))[0] / (H * W))
    out += list(np.percentile(g, [1, 5, 25, 50, 75, 95, 99]))
    return np.array(out, dtype=np.float32)


cache = OUT / "t4_handfeats.npz"
if cache.exists():
    z = np.load(cache); Ftr, Fte = z["tr"], z["te"]
else:
    Ftr = np.stack([feats(D / "train" / f"{s}.jpg") for s in tr.id.astype(str)])
    Fte = np.stack([feats(D / "test" / f"{s}.jpg") for s in te.id.astype(str)])
    np.savez_compressed(cache, tr=Ftr, te=Fte)
print("features", Ftr.shape, Fte.shape, flush=True)
classes = sorted(tr["class"].unique()); ytr = tr["class"].map({c: i for i, c in enumerate(classes)}).values
p = dict(objective="multiclass", num_class=5, learning_rate=0.08, num_leaves=63, min_child_samples=20, feature_fraction=0.6, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=2.0, verbose=-1, seed=342, n_jobs=12)
oof = np.zeros((len(Ftr), 5)); pte = np.zeros((len(Fte), 5))
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=342).split(Ftr, ytr)):
    m = lgb.train(p, lgb.Dataset(Ftr[a], ytr[a]), 3000, valid_sets=[lgb.Dataset(Ftr[b], ytr[b])], callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[b] = m.predict(Ftr[b], num_iteration=m.best_iteration); pte += m.predict(Fte, num_iteration=m.best_iteration) / 5
print(f"hand-crafted-feature model, 5-fold on train: accuracy {accuracy_score(ytr, oof.argmax(1)):.3f}, macro F1 {f1_score(ytr, oof.argmax(1), average='macro'):.3f}")
v3 = pd.read_csv(OUT / "submission_v3_t5_groups.csv"); resnet = v3.task4.map({c: i for i, c in enumerate(classes)}).values
pk = pte.argmax(1)
print(f"agreement with the submitted ResNet predictions on TEST (test.csv order): {(pk == resnet).mean():.3f}")
conf = pte.max(1); hi = conf > 0.8
print(f"   on the {int(hi.sum())} test images where the feature model is >80% confident: agreement {(pk[hi] == resnet[hi]).mean():.3f}")
print("   agreement if ResNet predictions were shuffled (chance level):", round(float((pk == np.random.default_rng(1).permutation(resnet)).mean()), 3))
print("feature-model test class shares:", dict(zip(classes, np.round(np.bincount(pk, minlength=5) / len(pk), 3))))
print("ResNet test class shares:      ", dict(zip(classes, np.round(np.bincount(resnet, minlength=5) / len(resnet), 3))))
print("confusion (rows = ResNet, cols = feature model) on TEST:\n", confusion_matrix(resnet, pk))
np.save(OUT / "t4_feat_test_prob.npy", pte)
