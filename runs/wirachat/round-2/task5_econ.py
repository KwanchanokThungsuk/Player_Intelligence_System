"""Round 2 - Task 5: is the 'low economic activity in all periods' excess one persistent group? (read-only)"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
warnings.filterwarnings("ignore")
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load, SEED

OUT = Path(__file__).resolve().parent
te = load(5, "test"); X = te.drop(columns=["player_id"])
G = pd.read_pickle(OUT / "task5_groups.pkl")
known = G.any(axis=1).values
rest = ~known
feats = ["purchase_frequency", "purchase_amount_std", "purchase_amount_mean", "currency_transfers", "location_changes"]


def rank_normal(s):
    r = s.rank(pct=True, na_option="keep"); n = s.notna().sum()
    return pd.Series(norm.ppf((r * n / (n + 1)).clip(1e-6, 1 - 1e-6)), index=s.index)


Z = {}
for f in feats:
    for p in (1, 2, 3, 4):
        Z[f"{f}_{p}"] = rank_normal(X.loc[rest, f"{f}_{p}"])
Z = pd.DataFrame(Z)
low4 = {f: (Z[[f"{f}_{p}" for p in (1, 2, 3, 4)]] < -2).all(axis=1) for f in feats}
print("rows low (<-2sd) in all 4 periods, per feature:", {f: int(v.sum()) for f, v in low4.items()})
inter = pd.DataFrame({f: low4[f] for f in feats})
print("number of features for which a row is low in all 4 periods:", inter.sum(axis=1).value_counts().sort_index().to_dict())
print("pairwise overlap (rows low-in-all-periods for both):")
print(pd.DataFrame({a: {b: int((low4[a] & low4[b]).sum()) for b in feats} for a in feats}).to_string())

# combined score: mean rank-normal z across the 5 features x 4 periods (NaN-aware)
m = Z.mean(axis=1, skipna=True)
h, e = np.histogram(m.dropna(), bins=40)
print("\ncombined score histogram (edge:count):", ", ".join(f"{a:.2f}:{c}" for a, c in zip(e[:-1], h)))
g = GaussianMixture(2, random_state=SEED, n_init=3).fit(m.dropna().values.reshape(-1, 1))
lo = int(np.argmin(g.means_.ravel()))
print("1-D GMM: component means", np.round(g.means_.ravel(), 3).tolist(), "weights", np.round(g.weights_, 4).tolist(), "sds", np.round(np.sqrt(g.covariances_.ravel()), 3).tolist())
post = pd.Series(np.nan, index=m.index); post[m.notna()] = g.predict_proba(m.dropna().values.reshape(-1, 1))[:, lo]
for t in (0.5, 0.9, 0.99):
    print(f"rows with posterior(low component) > {t}: {int((post > t).sum())}")
grp = post > 0.5
print("group size (posterior>0.5):", int(grp.sum()), "| rows with NaN score:", int(m.isna().sum()))
# what do these rows look like elsewhere?
rows_idx = grp[grp].index
flags = ["password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"]
print("flag rates group vs rest:", {f: (round(X.loc[rows_idx, f].mean(), 3), round(X.loc[rest & ~X.index.isin(rows_idx), f].mean(), 3)) for f in flags})
other = {}
for c in X.columns:
    if c in flags or any(c.startswith(f) for f in feats): continue
    a, b = X.loc[rows_idx, c].dropna(), X.loc[rest & ~X.index.isin(rows_idx), c].dropna()
    sd = b.std()
    if sd and len(a) > 20: other[c] = (a.mean() - b.mean()) / sd
o = pd.Series(other); o = o.reindex(o.abs().sort_values(ascending=False).index)
print("largest standardised mean differences on OTHER columns:", ", ".join(f"{k}={v:+.2f}" for k, v in o.head(6).items()))
pd.Series(grp.reindex(X.index, fill_value=False).values, index=X.index).to_pickle(OUT / "task5_econ_group.pkl")
