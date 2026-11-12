"""Round 3 - Tasks 1 and 2: GPU MLP (from scratch) vs the boosted trees. Same 5 folds as round 1."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, fbeta_score, f1_score
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "round-1"))
from r1_common import load, SEED
dev = torch.device("cuda")


def design(df, cols, cats=()):
    parts = []
    for c in cols:
        if c in cats:
            d = pd.get_dummies(df[c].fillna("NA"), dtype=np.float32); parts.append(d.values)
        else:
            v = df[c].values.astype(np.float64); med = np.nanmedian(v); isn = np.isnan(v)
            parts.append(np.where(isn, med, v)[:, None])
            if isn.any(): parts.append(isn[:, None].astype(np.float64))
    F = np.hstack(parts).astype(np.float32)
    return (F - F.mean(0)) / (F.std(0) + 1e-6)


def mlp(d_in, d_out, w=256, p=0.15):
    return nn.Sequential(nn.Linear(d_in, w), nn.SiLU(), nn.Dropout(p), nn.Linear(w, w), nn.SiLU(), nn.Dropout(p), nn.Linear(w, w // 2), nn.SiLU(), nn.Linear(w // 2, d_out)).to(dev)


def train(F, y, a, b, d_out, loss_fn, epochs=40, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    net = mlp(F.shape[1], d_out); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    Xa, ya = torch.tensor(F[a], device=dev), torch.tensor(y[a], device=dev); Xb = torch.tensor(F[b], device=dev)
    steps = epochs * (len(a) // 512 + 1); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(a), device=dev)
        for i in range(0, len(a), 512):
            idx = perm[i:i + 512]; loss = loss_fn(net(Xa[idx]), ya[idx]); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad(): return net(Xb).cpu().numpy()


# ---------------- Task 1
lab = load(1, "train").dropna(subset=["is_cheater"]).reset_index(drop=True); y = lab.is_cheater.astype(int).values
cols = [c for c in lab.columns if c not in ("player_id", "is_cheater")]; F = design(lab, cols)
oof = np.zeros(len(y)); skf = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(F, y))
for k, (a, b) in enumerate(skf):
    out = train(F, y.astype(np.float32), a, b, 1, lambda o, t: nn.functional.binary_cross_entropy_with_logits(o.squeeze(1), t))
    oof[b] = 1 / (1 + np.exp(-out.squeeze(1)))
gbm = (np.load(Path(__file__).resolve().parents[1] / "round-1/task1_oof_raw.npy") + np.load(Path(__file__).resolve().parents[1] / "round-1/task1_oof_fe.npy")) / 2
f2 = lambda p: max(fbeta_score(y, p >= t, beta=2) for t in np.linspace(.08, .5, 43))
print(f"Task 1: MLP AUC {roc_auc_score(y, oof):.4f} best plain F2 {f2(oof):.4f} | LightGBM AUC {roc_auc_score(y, gbm):.4f} F2 {f2(gbm):.4f} | 50/50 blend AUC {roc_auc_score(y, (oof + gbm) / 2):.4f} F2 {f2((oof + gbm) / 2):.4f}", flush=True)
# ---------------- Task 2
t2 = load(2, "train"); y2 = t2.segment.astype(int).values
cats = [c for c in t2.columns if c not in ("player_id", "segment") and not pd.api.types.is_numeric_dtype(t2[c])]
cols = [c for c in t2.columns if c not in ("player_id", "segment") and not c.startswith("random_metric")]; F2 = design(t2, cols, cats)
oof2 = np.zeros((len(y2), 4)); sk2 = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(F2, y2))
for k, (a, b) in enumerate(sk2):
    out = train(F2, y2, a, b, 4, lambda o, t: nn.functional.cross_entropy(o, t.long()), epochs=30)
    e = np.exp(out - out.max(1, keepdims=True)); oof2[b] = e / e.sum(1, keepdims=True)
gb2 = np.load(Path(__file__).resolve().parents[1] / "round-1/task2_oof_clean.npy")
mf = lambda p: f1_score(y2, p.argmax(1), average="macro")
print(f"Task 2: MLP macro F1 {mf(oof2):.4f} | LightGBM {mf(gb2):.4f} | 50/50 blend {mf((oof2 + gb2) / 2):.4f}")
