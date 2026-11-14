"""Re-run multi-scale test prediction from a saved fine-tuned state dict (used when the original run crashed in the prediction stage on Windows: DataLoader shared-memory error)."""
import sys, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
import t4_finetune2 as F
arch, tag, scales = sys.argv[1], sys.argv[2], [float(s) for s in sys.argv[3].split(",")]
H, W = 224, 448
te = pd.read_csv(F.D / "test.csv"); tid = te.id.astype(str).values
m = F.build(arch, H, W); m.load_state_dict(torch.load(F.R2 / "emb" / f"t4_{tag}.pt", map_location="cpu")); m = m.cuda().to(memory_format=torch.channels_last).eval()
def predict(model, ds, bs=32):
    P = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=3):
            x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last); acc = 0
            for v in (x, torch.flip(x, dims=[3])):
                with torch.autocast("cuda", dtype=torch.float16): acc = acc + torch.softmax(model(v).float(), 1)
            P.append((acc / 2).cpu().numpy())
    return np.concatenate(P)
if __name__ == "__main__":
    Ps = {}
    for s in scales:
        h, w = int(round(H * s / 32) * 32), int(round(W * s / 32) * 32)
        Ps[s] = predict(m, F.DS("test", tid, None, F.plain(h, w))); print(f"scale {s} -> {h}x{w}: mean max-prob {Ps[s].max(1).mean():.3f}", flush=True)
    np.save(F.R3 / f"t4_{tag}_test_prob_native.npy", Ps[1.0]); P = np.mean(list(Ps.values()), 0); np.save(F.R3 / f"t4_{tag}_test_prob.npy", P)
    print("shares", np.round(np.bincount(P.argmax(1), minlength=5) / len(P), 3), "mean max-prob", round(float(P.max(1).mean()), 3))
