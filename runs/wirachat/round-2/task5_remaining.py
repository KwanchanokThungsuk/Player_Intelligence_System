"""Round 2 - Task 5: search for the remaining (unflagged) anomalies (unsupervised, read-only)."""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load, SEED

OUT = Path(__file__).resolve().parent
te = load(5, "test"); X = te.drop(columns=["player_id"])
G = pd.read_pickle(OUT / "task5_groups.pkl")
known = G.any(axis=1).values
rest = ~known
print(f"rows: total {len(X)}, known-anomaly {int(known.sum())}, rest {int(rest.sum())}")

# 1) low-cardinality columns (skipped by the gap scan)
print("\nlow-cardinality columns (<25 unique values):")
for c in X.columns:
    u = X[c].dropna().unique()
    if len(u) < 25:
        vc = X.loc[rest, c].value_counts(normalize=True).sort_index().head(8).round(3).to_dict()
        print(f"  {c}: {len(u)} values; rest distribution {vc}")

# 2) persistent-shift scan using a Gaussian copula: observed vs expected count of rows extreme in ALL 4 periods
bases = sorted({re.sub(r"_[1-4]$", "", c) for c in X.columns if re.search(r"_[1-4]$", c)})
rng = np.random.default_rng(SEED)
res = []
for b in bases:
    cols = [f"{b}_{i}" for i in (1, 2, 3, 4)]
    M = X.loc[rest, cols]
    U = M.rank(pct=True, na_option="keep")
    n = M.notna().sum()
    Zs = pd.DataFrame(norm.ppf((U * M.notna().sum(0).values / (M.notna().sum(0).values + 1)).clip(1e-6, 1 - 1e-6)), index=M.index, columns=cols)
    cc = Zs.corr().values
    rho = cc[np.triu_indices(4, 1)].mean()
    full = Zs.dropna()
    if len(full) < 1000: continue
    for tail, k in (("high", 2.0), ("low", -2.0)):
        obs = ((full > k).all(axis=1) if tail == "high" else (full < k).all(axis=1)).sum() / len(full)
        # simulate equicorrelated Gaussian
        cov = np.full((4, 4), rho); np.fill_diagonal(cov, 1.0)
        sim = rng.multivariate_normal(np.zeros(4), cov, size=400000)
        ex = ((sim > k).all(axis=1) if tail == "high" else (sim < k).all(axis=1)).mean()
        res.append((b, tail, round(rho, 2), int(obs * len(full)), round(ex * len(full), 1), round((obs - ex) * len(full), 1)))
r = pd.DataFrame(res, columns=["base", "tail", "rho", "observed", "expected", "excess"]).sort_values("excess", ascending=False)
print("\nrows extreme (beyond +/-2 sd of rank-normal score) in all 4 periods: observed vs expected under equicorrelated Gaussian")
print(r.head(12).to_string(index=False))
