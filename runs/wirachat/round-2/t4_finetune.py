"""Round 2 - Task 4: end-to-end fine-tuning of a pretrained ConvNeXt-Tiny (torchvision, ImageNet weights) with strong CLASS-INDEPENDENT augmentation
(colour jitter, grayscale, blur, noise, rotation, crops) to break the class-dependent augmentation style present in the training images.
Validation = held-out CLUSTERS of training images, scored as-is and under heavy colour stress. Test images are only predicted (inference + TTA)."""
import argparse, time, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from sklearn.metrics import f1_score, accuracy_score
warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent
D = OUT.parents[2] / "cpe342-kaggle-main" / "04_image_classification"
H, W = 160, 320
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class AddNoise:
    def __init__(self, p=0.5, std=0.05): self.p, self.std = p, std
    def __call__(self, x):
        return x + torch.randn_like(x) * self.std * float(np.random.rand()) if np.random.rand() < self.p else x


train_tf = T.Compose([
    T.RandomResizedCrop((H, W), scale=(0.55, 1.0), ratio=(1.7, 2.3)),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.RandomRotation(25)], p=0.5),
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=(0.0, 2.2), hue=0.5),
    T.RandomGrayscale(p=0.3),
    T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.3),
    T.ToTensor(), AddNoise(), T.RandomErasing(p=0.25, scale=(0.02, 0.15)), T.Normalize(MEAN, STD)])
plain_tf = T.Compose([T.Resize((H, W)), T.ToTensor(), T.Normalize(MEAN, STD)])
stress_tf = T.Compose([T.Resize((H, W)), T.ColorJitter(brightness=0.4, contrast=0.4, saturation=(0.0, 2.2), hue=0.5), T.RandomGrayscale(p=0.3), T.ToTensor(), T.Normalize(MEAN, STD)])


class DS(Dataset):
    def __init__(self, split, ids, labels, tf, seed=0):
        self.split, self.ids, self.labels, self.tf, self.seed = split, list(ids), labels, tf, seed
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        np.random.seed((self.seed * 1000003 + i) % (2**31)); torch.manual_seed((self.seed * 1000003 + i) % (2**31))  # reproducible stress views
        with Image.open(D / self.split / f"{self.ids[i]}.jpg") as im:
            x = self.tf(im.convert("RGB"))
        return x, (-1 if self.labels is None else int(self.labels[i]))


@torch.no_grad()
def predict(model, ds, bs=128, tta=("id", "flip")):
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=8)
    P = []
    for x, _ in dl:
        x = x.cuda(non_blocking=True); acc = 0
        for t in tta:
            xx = torch.flip(x, dims=[3]) if t == "flip" else x
            with torch.autocast("cuda", dtype=torch.float16):
                acc = acc + torch.softmax(model(xx).float(), 1)
        P.append((acc / len(tta)).cpu().numpy())
    return np.concatenate(P)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--tag", default="cnxt"); ap.add_argument("--all", action="store_true", help="train on ALL training images (no held-out clusters)"); ap.add_argument("--arch", default="convnext_tiny", choices=["convnext_tiny", "convnext_small"])
    a = ap.parse_args()
    torch.manual_seed(342); np.random.seed(342)
    tr = pd.read_csv(D / "train.csv"); te = pd.read_csv(D / "test.csv")
    classes = sorted(tr["class"].unique()); y = tr["class"].map({c: i for i, c in enumerate(classes)}).values; ids = tr.id.astype(str).values
    # held-out clusters (from DINOv2 embeddings of TRAIN images): ~20% of the clusters
    from sklearn.cluster import MiniBatchKMeans
    E = np.load(OUT / "emb" / "facebook_dinov2-small__orig__train.npy"); E = E / np.linalg.norm(E, axis=1, keepdims=True)
    g = MiniBatchKMeans(300, random_state=1, n_init=3, batch_size=4096).fit(E).labels_
    rng = np.random.default_rng(1); held = rng.choice(300, 60, replace=False); val = np.isin(g, held)
    tri = np.arange(len(y)) if a.all else np.where(~val)[0]; vai = np.where(val)[0]
    print(f"train {len(tri)} / held-out-cluster val {len(vai)} images", flush=True)
    model = (torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1) if a.arch == "convnext_tiny"
             else torchvision.models.convnext_small(weights=torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1))
    model.classifier[2] = nn.Linear(768, 5); model = model.cuda().to(memory_format=torch.channels_last)
    cnt = np.bincount(y[tri], minlength=5); w = torch.tensor(cnt.sum() / (5 * cnt), dtype=torch.float32).cuda()
    dl = DataLoader(DS("train", ids[tri], y[tri], train_tf), batch_size=a.bs, shuffle=True, num_workers=10, drop_last=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    steps = a.epochs * len(dl); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(); crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); tot = 0
        for x, yy in dl:
            x, yy = x.cuda(non_blocking=True).to(memory_format=torch.channels_last), yy.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16): loss = crit(model(x), yy)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
        msg = f"epoch {ep+1}/{a.epochs} loss {tot/len(dl):.4f} ({time.time()-t0:.0f}s)"
        if not a.all and (ep % 3 == 2 or ep == a.epochs - 1):
            model.eval(); yv = y[vai]
            p1 = predict(model, DS("train", ids[vai], None, plain_tf), tta=("id",)).argmax(1)
            p2 = predict(model, DS("train", ids[vai], None, stress_tf, seed=7), tta=("id",)).argmax(1)
            msg += f" | held-out clusters: plain macroF1 {f1_score(yv, p1, average='macro'):.4f} acc {accuracy_score(yv, p1):.4f} | colour-stress macroF1 {f1_score(yv, p2, average='macro'):.4f}"
        print(msg, flush=True)
    torch.save(model.state_dict(), OUT / "emb" / f"t4_{a.tag}.pt")
    model.eval()
    P = predict(model, DS("test", te.id.astype(str).values, None, plain_tf), tta=("id", "flip"))
    np.save(OUT / f"t4_{a.tag}_test_prob.npy", P)
    pred = P.argmax(1)
    print("test class shares:", dict(zip(classes, np.round(np.bincount(pred, minlength=5) / len(pred), 3))), "| mean max-prob", round(float(P.max(1).mean()), 3))
    resnet = pd.read_csv(OUT / "submission_v3_t5_groups.csv").task4.map({c: i for i, c in enumerate(classes)}).values
    mv = np.load(OUT / "t4_bal_test_prob.npy").argmax(1)
    print(f"agreement with ResNet {(pred == resnet).mean():.3f} | with DINOv2 multi-view {(pred == mv).mean():.3f}")
