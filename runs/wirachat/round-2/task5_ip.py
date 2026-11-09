"""Round 2 - Task 5: is the ip_hash_entropy tail a separate anomaly population? (unsupervised, read-only)"""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load, OUT as R1_OUT

te = load(5, "test"); X = te.drop(columns=["player_id"])
flags = ["password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"]
S = pd.read_pickle(R1_OUT / "task5_S.pkl")["S"].fillna(0)
A = (S > 130).values

# 1) continuous raw columns with excess kurtosis outside the obvious group
cont = [c for c in X.columns if X[c].nunique() > 25 and c not in flags]
ku = pd.Series({c: X.loc[~A, c].kurt() for c in cont}).sort_values(ascending=False)
print("highest excess kurtosis (continuous cols, outside obvious group):")
print(ku.head(10).round(2).to_string())

# 2) ip entropy structure
ip = X[[f"ip_hash_entropy_{i}" for i in (1, 2, 3, 4)]]
print("\nip_hash_entropy per period quantiles (outside A): 1% / 50% / 99% / 99.9%")
for c in ip.columns:
    print(f"  {c}: {ip.loc[~A, c].quantile([.01, .5, .99, .999]).round(3).tolist()}")
print("cross-period corr:\n", ip.corr().round(3).to_string())
m = ip.mean(axis=1)
z = (m - m[~A].median()) / (1.4826 * (m[~A] - m[~A].median()).abs().median())
hi = (z > 3.5) & ~A
print(f"\nrows with mean ip entropy robust z>3.5 (outside A): {int(hi.sum())} | also in A: {int(((z > 3.5) & A).sum())}")
print("distribution of mean-ip-entropy z outside A (quantiles .5/.9/.99/.999/max):", z[~A].quantile([.5, .9, .99, .999, 1]).round(2).tolist())
npk = (ip.loc[hi] > ip.loc[~A].quantile(.99)).sum(axis=1)
print("among hi rows: number of periods above the 99th pct:", npk.value_counts().sort_index().to_dict())

# 3) what else is different about these rows?
rest = ~A & ~hi
print("\nflag rates hi vs rest:", {f: (round(X.loc[hi, f].mean(), 3), round(X.loc[rest, f].mean(), 3)) for f in flags})
diff = {}
for c in X.columns:
    if c in flags or c.startswith("ip_hash"): continue
    a, b = X.loc[hi, c].dropna(), X.loc[rest, c].dropna()
    sd = X.loc[rest, c].std()
    if sd and len(a) > 20: diff[c] = (a.mean() - b.mean()) / sd
d = pd.Series(diff); d = d.reindex(d.abs().sort_values(ascending=False).index)
print("largest standardised mean differences (hi - rest):", ", ".join(f"{k}={v:+.2f}" for k, v in d.head(8).items()))
