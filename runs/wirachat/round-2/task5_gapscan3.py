"""Round 2 - Task 5: gap scan on DERIVED per-row statistics, known groups removed (read-only)."""
import sys, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load

OUT = Path(__file__).resolve().parent
te = load(5, "test"); X = te.drop(columns=["player_id"])
G = pd.read_pickle(OUT / "task5_groups.pkl"); rest = ~G.any(axis=1).values
bases = sorted({re.sub(r"_[1-4]$", "", c) for c in X.columns if re.search(r"_[1-4]$", c)})
D = {}
for b in bases:
    M = X[[f"{b}_{i}" for i in (1, 2, 3, 4)]].values.astype(float)
    ok = (~np.isnan(M)).sum(1) >= 3
    D[f"mean_{b}"] = np.where(ok, np.nanmean(M, 1), np.nan)
    D[f"spread_{b}"] = np.where(ok, np.nanstd(M, 1), np.nan)
    D[f"min_{b}"] = np.where(ok, np.nanmin(M, 1), np.nan)
    D[f"max_{b}"] = np.where(ok, np.nanmax(M, 1), np.nan)
    D[f"d4_{b}"] = np.where(~np.isnan(M[:, 3]) & ok, M[:, 3] - np.nanmean(M[:, :3], 1), np.nan)
D = pd.DataFrame(D, index=X.index)
rows = []
for c in D.columns:
    v = D.loc[rest, c].dropna().values
    if len(np.unique(v)) < 40: continue
    h, e = np.histogram(v, bins=120)
    nz = np.where(h > 0)[0]
    gaps = [(i1 - i0 - 1, (e[i0 + 1] + e[i1]) / 2) for i0, i1 in zip(nz[:-1], nz[1:])]
    if not gaps: continue
    n_empty, cut = max(gaps)
    small = min((v < cut).sum(), (v > cut).sum())
    if n_empty >= 4 and small >= 40:
        rows.append((c, n_empty, round(cut, 3), int(small)))
r = pd.DataFrame(rows, columns=["derived_feature", "empty_bins(of 120)", "cut", "n_small_side"]).sort_values("empty_bins(of 120)", ascending=False)
print(f"derived features with an internal gap (>=4 empty bins of 120, small side>=40): {len(r)}")
print(r.head(12).to_string(index=False))
