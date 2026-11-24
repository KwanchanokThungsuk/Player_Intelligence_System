"""Task 4 - fine-tune ONE pretrained backbone end to end on all training images and predict the test images (the building block of the final ensemble).
Pretrained weights (allowed for Task 4) come from torchvision (ConvNeXt-S/T) or the Hugging Face hub through timm (ConvNeXt-B 22k, DINOv2-B/L).
Training: 224x448 input, strong class-independent augmentation drawn fresh every epoch (--fresh-aug), class-balanced cross-entropy with label smoothing 0.1, AdamW + one-cycle lr, fp16.
New recipe (best members): --llrd (layer-wise lr decay), --ema (weight EMA used for the final predictions), --mix (CutMix/MixUp on a share of the batches).
Prediction: multi-scale (--scales) + horizontal-flip test-time augmentation; saves softmax probabilities in the row order of test.csv.
Run guide and the exact command of every ensemble member: README.md and train_all_members.sh in this folder."""
import argparse, os, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from sklearn.metrics import f1_score, accuracy_score
warnings.filterwarnings("ignore")
# data: Kaggle download of the corrected release. CSVs (id, file_name, label) and the image folders train/ and test/ may live in different places.
DATA_T4 = Path(os.environ.get("CPE342_DATA", ".")) / "04_image_classification"
CSV = Path(os.environ.get("CPE342_T4_CSV_DIR", DATA_T4))      # folder with train.csv / test.csv
IMG = Path(os.environ.get("CPE342_T4_IMAGES", DATA_T4))       # folder with train/ and test/ (jpg files named <file_name>)
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class AddNoise:
    def __init__(self, p=0.6, std=0.10): self.p, self.std = p, std
    def __call__(self, x): return x + torch.randn_like(x) * self.std * float(np.random.rand()) if np.random.rand() < self.p else x


def channel_shuffle(im): return Image.merge("RGB", [im.split()[i] for i in np.random.permutation(3)])
def down_up(im):
    w, h = im.size; s = np.random.uniform(0.4, 0.9); return im.resize((max(8, int(w * s)), max(8, int(h * s))), Image.BILINEAR).resize((w, h), Image.BILINEAR)


def make_tf(H, W):
    train = T.Compose([
        T.RandomResizedCrop((H, W), scale=(0.45, 1.0), ratio=(1.6, 2.4)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.RandomRotation(45)], p=0.6),
        T.RandomApply([T.Lambda(channel_shuffle)], p=0.3),
        T.ColorJitter(brightness=0.6, contrast=0.6, saturation=(0.0, 2.5), hue=0.5),
        T.RandomApply([T.RandomEqualize(1.0)], p=0.12), T.RandomApply([T.RandomAutocontrast(1.0)], p=0.15),
        T.RandomApply([T.RandomPosterize(3, 1.0)], p=0.08), T.RandomApply([T.RandomSolarize(150, 1.0)], p=0.08),
        T.RandomGrayscale(p=0.25),
        T.RandomApply([T.Lambda(down_up)], p=0.35),
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.5))], p=0.3),
        T.ToTensor(), AddNoise(), T.RandomErasing(p=0.3, scale=(0.02, 0.2)), T.Normalize(MEAN, STD)])
    stress = T.Compose([T.Resize((H, W)), T.ColorJitter(brightness=0.4, contrast=0.4, saturation=(0.0, 2.2), hue=0.5), T.RandomGrayscale(p=0.3), T.ToTensor(), T.Normalize(MEAN, STD)])
    return train, stress


def plain(H, W): return T.Compose([T.Resize((H, W)), T.ToTensor(), T.Normalize(MEAN, STD)])


class DS(Dataset):
    def __init__(self, split, ids, labels, tf, seed=0): self.split, self.ids, self.labels, self.tf, self.seed = split, list(ids), labels, tf, seed
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        # seed=None (training with --fresh-aug): leave the worker RNG streams alone so every epoch draws NEW augmentations.
        # (v1/v2 runs seeded by image index, i.e. every image got the SAME augmentation in every epoch.)
        if self.seed is not None: np.random.seed((self.seed * 1000003 + i) % (2**31)); torch.manual_seed((self.seed * 1000003 + i) % (2**31))
        with Image.open(IMG / self.split / f"{self.ids[i]}.jpg") as im: x = self.tf(im.convert("RGB"))
        return x, (-1 if self.labels is None else int(self.labels[i]))


@torch.no_grad()
def predict(model, ds, bs=64, flip=True):
    P = []
    for x, _ in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2):
        x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last); acc = 0; views = [x] + ([torch.flip(x, dims=[3])] if flip else [])
        for v in views:
            with torch.autocast("cuda", dtype=torch.float16): acc = acc + torch.softmax(model(v).float(), 1)
        P.append((acc / len(views)).cpu().numpy())
    return np.concatenate(P)


def build(arch, H=224, W=448):
    if arch.startswith("timm:"):       # timm backbones (e.g. timm:convnext_base.fb_in22k_ft_in1k, timm:vit_base_patch14_dinov2.lvd142m)
        import timm
        kw = dict(img_size=(H, W)) if any(k in arch for k in ("vit", "eva", "deit", "dinov2")) else {}
        from huggingface_hub import hf_hub_download     # timm 1.0.15's own hub import is incompatible with huggingface_hub 2.x: fetch the weights ourselves
        f = hf_hub_download(f"timm/{arch[5:]}", "model.safetensors")
        return timm.create_model(arch[5:], pretrained=True, pretrained_cfg_overlay=dict(file=f), num_classes=5, drop_path_rate=0.1, **kw)
    if arch.startswith("convnext"):
        m = getattr(torchvision.models, arch)(weights="IMAGENET1K_V1"); m.classifier[2] = nn.Linear(m.classifier[2].in_features, 5)
    else:
        m = getattr(torchvision.models, arch)(weights="IMAGENET1K_V1"); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 5)
    return m


def mix_batch(x, y):
    """CutMix (rectangle with the image's own aspect ratio) or MixUp, 50/50. Returns mixed x, labels y, partner labels y2 and the weight lam of y."""
    idx = torch.randperm(x.size(0), device=x.device); y2 = y[idx]
    if np.random.rand() < 0.5:
        lam = float(np.random.beta(1.0, 1.0)); Hh, Ww = x.shape[2], x.shape[3]; r = (1 - lam) ** 0.5
        ch, cw = int(Hh * r), int(Ww * r); cy, cx = np.random.randint(Hh), np.random.randint(Ww)
        y1, y2_, x1, x2_ = max(cy - ch // 2, 0), min(cy + ch // 2, Hh), max(cx - cw // 2, 0), min(cx + cw // 2, Ww)
        x = x.clone(); x[:, :, y1:y2_, x1:x2_] = x[idx][:, :, y1:y2_, x1:x2_]; lam = 1 - (y2_ - y1) * (x2_ - x1) / float(Hh * Ww)
    else:
        lam = float(np.random.beta(0.8, 0.8)); x = lam * x + (1 - lam) * x[idx]
    return x, y, y2, lam


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, p in zip(ema.parameters(), model.parameters()): pe.mul_(decay).add_(p.detach().to(pe.device), alpha=1 - decay)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--arch", default="convnext_tiny"); ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--h", type=int, default=224); ap.add_argument("--w", type=int, default=448)
    ap.add_argument("--tag", required=True); ap.add_argument("--scales", default="0.8,1.0,1.25")
    ap.add_argument("--all", action="store_true", help="kept for compatibility with the original commands: the final models always train on ALL training images")
    ap.add_argument("--seed", type=int, default=342); ap.add_argument("--fresh-aug", dest="fresh_aug", action="store_true", help="new random augmentations every epoch (the final models all use this)")
    ap.add_argument("--gc", action="store_true", help="gradient (activation) checkpointing for timm models (needed for ViT-Large on 8 GB)")
    ap.add_argument("--workers", type=int, default=10, help="training DataLoader workers")
    ap.add_argument("--llrd", type=float, default=1.0, help="layer-wise lr decay for timm ViTs (1.0 = off; 0.8 for ViT-B, 0.9 for ViT-L; --lr is then the lr of the top layer)")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights (0 = off, e.g. 0.999); the final predictions use the EMA weights")
    ap.add_argument("--ema-cpu", dest="ema_cpu", action="store_true", help="keep the EMA copy on the CPU (updated every --ema-every steps) to save GPU memory (ViT-Large)")
    ap.add_argument("--ema-every", dest="ema_every", type=int, default=1, help="update the EMA every N optimizer steps")
    ap.add_argument("--mix", type=float, default=0.0, help="probability per batch of applying CutMix or MixUp (0 = off)")
    ap.add_argument("--out", default=os.environ.get("CPE342_T4_OUT", "../outputs/task4_probs"), help="folder for the probability file and the weights")
    ap.add_argument("--smoke", action="store_true", help="quick test: 640 training images, 1 epoch, 64 test images")
    a = ap.parse_args(); H, W = a.h, a.w
    OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tr = pd.read_csv(CSV / "train.csv"); te = pd.read_csv(CSV / "test.csv")
    y = tr["label"].astype(int).values; ids = tr.file_name.str.replace(".jpg", "", regex=False).values; tid = te.file_name.str.replace(".jpg", "", regex=False).values
    classes = ["7-11 Knights", "FiveN", "Free Fried", "PUBE", "Roblock"]
    if a.smoke:
        sub = np.random.RandomState(0).choice(len(y), 640, replace=False); ids, y, tid, a.epochs = ids[sub], y[sub], tid[:64], 1
    tri = np.arange(len(y))
    print(f"{a.arch} {H}x{W} | training images {len(tri)} | test images {len(tid)}", flush=True)
    train_tf, _ = make_tf(H, W)
    model = build(a.arch, H, W)
    if a.gc: model.set_grad_checkpointing(True)      # activation checkpointing: lets ViT-Large fit in 8 GB
    model = model.cuda().to(memory_format=torch.channels_last)
    cnt = np.bincount(y[tri], minlength=5); w = torch.tensor(cnt.sum() / (5 * cnt), dtype=torch.float32).cuda()      # class-balanced loss weights
    dl = DataLoader(DS("train", ids[tri], y[tri], train_tf, seed=None if a.fresh_aug else 0), batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True, persistent_workers=True)
    if a.llrd < 1.0:        # layer-wise lr decay (timm builds one param group per ViT block with an lr_scale; OneCycleLR needs the per-group max lr)
        import timm.optim
        opt = timm.optim.create_optimizer_v2(model, opt="adamw", lr=a.lr, weight_decay=0.05, layer_decay=a.llrd)
        max_lr = [a.lr * g.get("lr_scale", 1.0) for g in opt.param_groups]
        print(f"layer-wise lr decay {a.llrd}: {len(max_lr)} param groups, lr from {min(max_lr):.2e} to {max(max_lr):.2e}", flush=True)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05); max_lr = a.lr
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=a.epochs * len(dl), pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(); crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1); t0 = time.time()
    ema = None
    if a.ema > 0:
        import copy
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        if a.ema_cpu: ema = ema.cpu()
    step = 0
    for ep in range(a.epochs):
        model.train(); tot = 0
        for x, yy in dl:
            x, yy = x.cuda(non_blocking=True).to(memory_format=torch.channels_last), yy.cuda(non_blocking=True)
            if a.mix > 0 and np.random.rand() < a.mix:
                x, ya, yb, lam = mix_batch(x, yy)
                with torch.autocast("cuda", dtype=torch.float16): out = model(x); loss = lam * crit(out, ya) + (1 - lam) * crit(out, yb)
            else:
                with torch.autocast("cuda", dtype=torch.float16): loss = crit(model(x), yy)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tot += loss.item(); step += 1
            if ema is not None and step % a.ema_every == 0: ema_update(ema, model, min(a.ema, (1 + step) / (10 + step)) ** a.ema_every)
        print(f"epoch {ep+1}/{a.epochs} loss {tot/len(dl):.4f} ({time.time()-t0:.0f}s)", flush=True)
    if ema is not None: model = ema.cuda() if a.ema_cpu else ema; del opt; torch.cuda.empty_cache()    # final predictions use the EMA weights
    torch.save(model.state_dict(), OUT / f"t4_{a.tag}.pt"); model.eval()
    Ps = {}
    for s in [float(v) for v in a.scales.split(",")]:
        h, w_ = int(round(H * s / 32) * 32), int(round(W * s / 32) * 32)
        Ps[s] = predict(model, DS("test", tid, None, plain(h, w_))); print(f"scale {s} -> {h}x{w_}: mean max-prob {Ps[s].max(1).mean():.3f}", flush=True)
    P = np.mean(list(Ps.values()), 0); np.save(OUT / f"t4_{a.tag}_test_prob.npy", P)         # rows follow test.csv; columns = classes 0..4
    print("multi-scale test class shares:", dict(zip(classes, np.round(np.bincount(P.argmax(1), minlength=5) / len(P), 3))), "| mean max-prob", round(float(P.max(1).mean()), 3))
