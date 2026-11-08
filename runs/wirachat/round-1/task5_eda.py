"""Round 1 - Task 5 EDA: structure of the unlabeled data (read-only)."""
import sys, re
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from r1_common import load, SEED

te = load(5, "test"); X = te.drop(columns=["player_id"])
flags = ["password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"]
bases = sorted({re.sub(r"_[1-4]$", "", c) for c in X.columns if re.search(r"_[1-4]$", c)})
static = [c for c in X.columns if not re.search(r"_[1-4]$", c)]
print("base measures:", len(bases), "| static cols:", static)
Z = (X - X.median()) / (X.quantile(.75) - X.quantile(.25)).replace(0, np.nan)  # robust z
print("\nheavy tails: fraction |robust z|>4 for top features")
print((Z.abs() > 4).mean().sort_values(ascending=False).head(12).round(3).to_string())

# period structure: cross-period correlation of the same measure; and period-to-period change size
print("\ncross-period correlation of base measures (mean corr between period 1 and 4):")
cc = {b: X[f"{b}_1"].corr(X[f"{b}_4"]) for b in bases}
print(pd.Series(cc).sort_values().round(2).to_string())

# association of flags with other features
anyflag = (X[flags].fillna(0).sum(axis=1) >= 1)
print("\nrows with any flag:", anyflag.mean().round(3))
d = []
for c in X.columns:
    if c in flags: continue
    a, b = X.loc[anyflag, c].dropna(), X.loc[~anyflag, c].dropna()
    sd = X[c].std()
    if sd > 0: d.append((c, (a.mean() - b.mean()) / sd))
d = pd.Series(dict(d)).sort_values()
print("largest standardized mean differences (flagged - unflagged):")
print(pd.concat([d.head(6), d.tail(6)]).round(3).to_string())
print("\nflags pairwise corr:\n", X[flags].corr().round(3).to_string())

# latent cluster check: PCA on robust z, imputed 0
Zi = Z.fillna(0).clip(-6, 6)
pca = PCA(10, random_state=SEED).fit(Zi)
print("\nPCA explained var ratio (10):", np.round(pca.explained_variance_ratio_, 3).tolist())
P = pca.transform(Zi)
for k in range(3):
    s = pd.Series(P[:, k]); print(f"PC{k+1}: skew {s.skew():.2f} kurt {s.kurt():.2f} | quantiles 0.1/1/50/99/99.9: {s.quantile([.001,.01,.5,.99,.999]).round(1).tolist()}")
# GMM on top PCs: does a small second component appear?
for k in (1, 2, 3, 4):
    g = GaussianMixture(k, covariance_type="full", random_state=SEED, n_init=2).fit(P[:, :8])
    print(f"GMM k={k}: BIC={g.bic(P[:, :8]):.0f} weights={np.round(np.sort(g.weights_)[::-1], 3).tolist()}")
