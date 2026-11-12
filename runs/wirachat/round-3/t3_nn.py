"""Round 3 - Task 3: can a smooth model (log-linear / MLP trained on the GPU) explain the POSITIVE part of spending better than the trees?
Trees: 5-fold residual sd of log(y) = 0.225 (log-space R2 0.9895). Trained from scratch on the provided training data only."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "round-1"))
from r1_common import load, SEED
tr = load(3, "train"); P = tr[tr.spending_30d > 0].reset_index(drop=True)
ly = np.log(P.spending_30d.values)
feat = [c for c in P.columns if c not in ("player_id", "spending_30d")]
tx = P.total_transactions * P.avg_transaction_value
X = P[feat].copy(); X["hist_f"] = P.historical_spending.fillna(tx).fillna(P.prev_month_spending * np.exp(1.05)); X["prev_f"] = P.prev_month_spending.fillna(X.hist_f * np.exp(-1.05))
cols = list(X.columns)
M = X.isna().values.astype(np.float32)                                  # missing indicators
parts = []
for c in cols:
    v = X[c].values.astype(np.float64); med = np.nanmedian(v); v = np.where(np.isnan(v), med, v)
    parts.append(v)
    if (X[c].dropna() > 0).all() and X[c].dropna().max() / max(X[c].dropna().min(), 1e-9) > 50: parts.append(np.log(np.maximum(v, 1e-6)))   # log copy of heavy-tailed positive columns
F = np.column_stack(parts + [M[:, M.any(0)]]).astype(np.float32)
F = (F - F.mean(0)) / (F.std(0) + 1e-6)
print(f"positive rows {len(P)}, design matrix {F.shape}")
res_ols, res_mlp = [], []
dev = torch.device("cuda")
for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(F)):
    A = np.c_[F[a], np.ones(len(a))]; beta, *_ = np.linalg.lstsq(A, ly[a], rcond=None)
    res_ols.append(ly[b] - np.c_[F[b], np.ones(len(b))] @ beta)
    torch.manual_seed(SEED + k)
    net = nn.Sequential(nn.Linear(F.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 1)).to(dev)
    Xa, ya = torch.tensor(F[a], device=dev), torch.tensor(ly[a], dtype=torch.float32, device=dev)
    Xb, yb = torch.tensor(F[b], device=dev), torch.tensor(ly[b], dtype=torch.float32, device=dev)
    mu = ya.mean(); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4); steps = 60 * (len(a) // 512 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=steps); it = 0; best = 9
    for ep in range(60):
        perm = torch.randperm(len(a), device=dev); net.train()
        for i in range(0, len(a), 512):
            idx = perm[i:i + 512]; loss = nn.functional.smooth_l1_loss(net(Xa[idx]).squeeze(1) + mu, ya[idx], beta=0.3)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); it += 1
        if ep % 10 == 9 or ep == 59:
            net.eval()
            with torch.no_grad(): r = (yb - (net(Xb).squeeze(1) + mu)).cpu().numpy()
            best = min(best, float(np.sqrt((r ** 2).mean())))
    res_mlp.append(r)
    print(f"fold {k}: OLS residual sd {res_ols[-1].std():.4f} | MLP residual sd {r.std():.4f} (best seen rmse {best:.4f})", flush=True)
ro, rm = np.concatenate(res_ols), np.concatenate(res_mlp)
print(f"\nlog-space residual sd: log-linear {ro.std():.4f} | MLP {rm.std():.4f} | trees (round 2) 0.225")
print(f"median |residual|: log-linear {np.median(np.abs(ro)):.4f} | MLP {np.median(np.abs(rm)):.4f}")
