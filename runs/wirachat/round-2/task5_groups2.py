"""Round 2 - Task 5: are the gap sub-populations persistent groups (same rows) or random per period? (read-only)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load, OUT as R1_OUT

te = load(5, "test"); X = te.drop(columns=["player_id"])
S = pd.read_pickle(R1_OUT / "task5_S.pkl")["S"].fillna(0)
A = (S > 130).values


def gap_indicator(col):
    v = X.loc[~A, col].dropna().values
    h, e = np.histogram(v, bins=80)
    nz = np.where(h > 0)[0]
    best = max(((i1 - i0 - 1, (e[i0 + 1] + e[i1]) / 2) for i0, i1 in zip(nz[:-1], nz[1:])), key=lambda x: x[0])
    cut = best[1]
    small_is_low = (v < cut).sum() < (v > cut).sum()
    ind = (X[col] < cut) if small_is_low else (X[col] > cut)
    return ind.fillna(False), cut, "low" if small_is_low else "high"


fam = {"ip": "ip_hash_entropy", "ltv": "login_time_variance", "comm": "communication_frequency", "act": "activity_level"}
I = {}
for k, base in fam.items():
    for p in (1, 2, 3, 4):
        ind, cut, side = gap_indicator(f"{base}_{p}")
        I[f"{k}{p}"] = ind.values
        if p == 1: print(f"{base}: odd side = {side} than {cut:.3f}")
I["bs"] = ((X.behavior_stability < 0.55).fillna(False)).values
I = pd.DataFrame(I)
print("\nodd counts per column:", I.sum().to_dict())
print("in obvious group A:", {c: int((I[c] & A).sum()) for c in I.columns if I[c].sum() and (I[c] & A).sum() > 0} or "none")

for k in ("ip", "ltv", "comm", "act"):
    cols = [f"{k}{p}" for p in (1, 2, 3, 4)]
    n = I[cols].sum(axis=1)
    exp = 1 - (1 - I[cols].values.mean()) ** 4
    print(f"[{k}] rows odd in 0/1/2/3/4 periods: {n.value_counts().sort_index().to_dict()} | union={int((n>0).sum())} | expected union if independent per period: {exp*len(I):.0f}")

print("\nsame-period overlaps (period 4): rows odd in both")
for a in ("ip", "ltv", "comm", "act"):
    print("  ", a, {b: int((I[f'{a}4'] & I[f'{b}4']).sum()) for b in ("ip", "ltv", "comm", "act") if b != a}, "| with bs:", int((I[f'{a}4'] & I['bs']).sum()))
anyodd = I.drop(columns="bs").any(axis=1)
print("\nany odd indicator (excluding bs):", int(anyodd.sum()), "| plus bs:", int((anyodd | I.bs).sum()))
tot = I.sum(axis=1)
print("number of odd indicators per row:", tot.value_counts().sort_index().to_dict())
