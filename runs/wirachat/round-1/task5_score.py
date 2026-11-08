"""Round 1 - Task 5: unsupervised statistical score on period-change features; look for a natural gap."""
import sys
import numpy as np, pandas as pd
from scipy.stats import chi2
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from r1_common import load, OUT

te = load(5, "test"); X = te.drop(columns=["player_id"])
Dz = pd.read_pickle(OUT / "task5_dz.pkl")
d4 = Dz.filter(like="d4_")
n_av = d4.notna().sum(axis=1)
print("available d4 features per row: min", int(n_av.min()), "median", int(n_av.median()))
# are normal-state d4 features ~independent N(0,1)?  use rows with no |z|>3
calm = (d4.abs() <= 3).all(axis=1) | d4.isna().any(axis=1) & (d4.abs().max(axis=1) <= 3)
Cm = d4[calm].corr().values
off = np.abs(Cm[np.triu_indices_from(Cm, 1)])
print("calm rows:", int(calm.sum()), "| mean |offdiag corr|:", round(np.nanmean(off), 3), "max:", round(np.nanmax(off), 3))
print("calm rows per-feature z std (should be ~<1 due to truncation):", round(d4[calm].std().mean(), 3))

cap = 64.0
S = (d4.clip(-np.sqrt(cap), np.sqrt(cap)) ** 2).sum(axis=1, min_count=1)
Sn = S / n_av * d4.shape[1]            # rescale rows with missing features to 29 features
print("\nscore S (sum of capped z^2, rescaled to 29 features): quantiles")
print(Sn.quantile([.5, .9, .95, .97, .975, .98, .99, .999]).round(1).to_dict())
print("\nthreshold  observed_flagged  expected_if_all_normal(chi2 df=29)")
for t in (45, 50, 60, 70, 80, 100, 150, 250, 500):
    print(f"{t:>8}  {int((Sn > t).sum()):>16}  {len(Sn) * chi2.sf(t, d4.shape[1]):>10.1f}")
h, edges = np.histogram(np.log10(Sn.dropna()), bins=24)
print("\nlog10(S) histogram (edge: count):", ", ".join(f"{10**e:.0f}:{c}" for e, c in zip(edges[:-1], h)))
Sn.to_frame("S").to_pickle(OUT / "task5_S.pkl")
