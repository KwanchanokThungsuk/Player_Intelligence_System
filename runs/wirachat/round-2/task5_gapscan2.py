"""Round 2 - Task 5: second gap scan after removing the known groups; also builds the group indicators."""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load, OUT as R1_OUT

OUT = Path(__file__).resolve().parent
te = load(5, "test"); X = te.drop(columns=["player_id"])
S = pd.read_pickle(R1_OUT / "task5_S.pkl")["S"].fillna(0)
A = (S > 130).values


def gap_cut(v, bins=80):
    h, e = np.histogram(v, bins=bins)
    nz = np.where(h > 0)[0]
    return max(((i1 - i0 - 1, (e[i0 + 1] + e[i1]) / 2) for i0, i1 in zip(nz[:-1], nz[1:])), key=lambda x: x[0])


def family_members(base, periods=(1, 2, 3, 4)):
    """rows odd (on the small side of the largest gap) in >=2 periods"""
    cnt = np.zeros(len(X), int)
    for p in periods:
        c = f"{base}_{p}" if p else base
        v = X.loc[~A, c].dropna().values
        n_empty, cut = gap_cut(v)
        low = (v < cut).sum() < (v > cut).sum()
        cnt += ((X[c] < cut) if low else (X[c] > cut)).fillna(False).values.astype(int)
    return cnt >= 2


G = {}
G["ip_ltv"] = family_members("ip_hash_entropy") | family_members("login_time_variance")
G["comm"] = family_members("communication_frequency")
G["act"] = family_members("activity_level")
G["bs"] = (X.behavior_stability < 0.55).fillna(False).values
known = A.copy()
for k, v in G.items():
    print(f"group {k}: {int(v.sum())} rows | overlap with A: {int((v & A).sum())}")
    known |= v
print("known union:", int(known.sum()), f"({known.mean():.2%})  | A alone {int(A.sum())}")
pd.DataFrame(G).assign(A=A).to_pickle(OUT / "task5_groups.pkl")

# second scan on the remaining rows
rest = ~known
rows = []
for c in X.columns:
    v = X.loc[rest, c].dropna().values
    if len(np.unique(v)) < 25: continue
    n_empty, cut = gap_cut(v, 100)
    small = min((v < cut).sum(), (v > cut).sum())
    if n_empty >= 3 and small >= 40:
        rows.append((c, n_empty, round(cut, 3), int(small), round(small / len(v), 4)))
res = pd.DataFrame(rows, columns=["column", "empty_bins(of 100)", "cut", "n_small_side", "frac"]).sort_values("empty_bins(of 100)", ascending=False)
print(f"\nremaining columns with a gap (>=3 empty bins of 100, small side >= 40 rows): {len(res)}")
print(res.head(12).to_string(index=False))
