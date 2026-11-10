"""Round 2 - Task 5: are the low tails of the economic features just a pile of exact zeros (clipping)? (read-only)"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load

OUT = Path(__file__).resolve().parent
te = load(5, "test"); X = te.drop(columns=["player_id"])
G = pd.read_pickle(OUT / "task5_groups.pkl"); rest = ~G.any(axis=1).values
feats = ["purchase_frequency", "purchase_amount_std", "purchase_amount_mean", "currency_transfers", "location_changes"]
for f in feats:
    M = X.loc[rest, [f"{f}_{p}" for p in (1, 2, 3, 4)]]
    zero = (M <= 0)
    print(f"[{f}] share of exact/clipped <=0 per period: {np.round(zero.mean().values, 4).tolist()} | rows <=0 in all 4 periods: {int(zero.all(axis=1).sum())} | in >=3 periods: {int((zero.sum(axis=1) >= 3).sum())}")
    mean = M.mean(axis=1).dropna()
    h, e = np.histogram(mean, bins=30)
    print("   row-mean histogram:", ", ".join(f"{a:.1f}:{c}" for a, c in zip(e[:-1], h)))
    # a continuous bump separated from the bulk would show as an isolated low block: compare lowest bins with a smooth tail
    pos = M[(M > 0).all(axis=1)].mean(axis=1)
    print(f"   rows with all 4 periods >0: {len(pos)} | mean of positive rows {pos.mean():.2f} sd {pos.std():.2f}; rows with row-mean < 0.3 sd... skip")
