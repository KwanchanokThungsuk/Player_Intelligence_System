"""Round 2 - Task 5: scan every column for internal gaps / separate sub-populations (unsupervised, read-only)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load, OUT as R1_OUT

te = load(5, "test"); X = te.drop(columns=["player_id"])
S = pd.read_pickle(R1_OUT / "task5_S.pkl")["S"].fillna(0)
A = (S > 130).values
rows = []
for c in X.columns:
    v = X.loc[~A, c].dropna().values
    if len(np.unique(v)) < 12: continue
    lo, hi = v.min(), v.max()
    h, e = np.histogram(v, bins=80)
    # internal empty run strictly between non-empty bins
    nz = np.where(h > 0)[0]
    gaps = []
    for i0, i1 in zip(nz[:-1], nz[1:]):
        if i1 - i0 > 3: gaps.append((e[i0 + 1], e[i1], i1 - i0 - 1))
    small_side = 0
    if gaps:
        g = max(gaps, key=lambda x: x[2])
        small_side = min((v < g[0]).sum(), (v > g[1]).sum())
        rows.append((c, g[2], round(g[0], 3), round(g[1], 3), int(small_side), round(small_side / len(v), 4)))
res = pd.DataFrame(rows, columns=["column", "empty_bins", "gap_from", "gap_to", "n_in_small_side", "frac_small_side"]).sort_values("empty_bins", ascending=False)
print(f"columns with an internal gap (>=4 empty bins of 80), outside obvious group A: {len(res)}")
print(res.head(15).to_string(index=False))
