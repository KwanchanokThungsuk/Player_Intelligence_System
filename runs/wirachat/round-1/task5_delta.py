"""Round 1 - Task 5: period-change features and mixture structure (read-only exploration)."""
import sys, re
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from r1_common import load, SEED, OUT

te = load(5, "test"); X = te.drop(columns=["player_id"])
flags = ["password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"]
bases = sorted({re.sub(r"_[1-4]$", "", c) for c in X.columns if re.search(r"_[1-4]$", c)})


def robust_z(s):
    med, iqr = s.median(), s.quantile(.75) - s.quantile(.25)
    return (s - med) / (iqr / 1.349 if iqr > 0 else s.std())


D = pd.DataFrame(index=X.index)
for b in bases:
    prev = X[[f"{b}_1", f"{b}_2", f"{b}_3"]].mean(axis=1)
    D[f"d4_{b}"] = X[f"{b}_4"] - prev                       # last period vs earlier mean
    D[f"d3_{b}"] = X[f"{b}_3"] - X[[f"{b}_1", f"{b}_2"]].mean(axis=1)  # control: period 3 vs 1-2
    D[f"sd_{b}"] = X[[f"{b}_{i}" for i in (1, 2, 3, 4)]].std(axis=1)
Dz = D.apply(robust_z)
heavy4 = (Dz.filter(like="d4_").abs() > 4).mean().sort_values(ascending=False)
heavy3 = (Dz.filter(like="d3_").abs() > 4).mean()
print("fraction |robust z|>4  d4 (period4 vs 1-3)  vs  d3 (control):")
print(pd.DataFrame({"d4": heavy4, "d3_control": heavy3.rename(lambda s: s.replace("d3_", "d4_"))}).head(14).round(4).to_string())

# anomaly-ish evidence: rows with many extreme d4 features
ext = (Dz.filter(like="d4_").abs() > 3.5).sum(axis=1)
print("\nnumber of extreme d4 features per row: ", ext.value_counts().sort_index().head(10).to_dict())
anyflag = X[flags].fillna(0).sum(axis=1) >= 1
print("any-flag rate by #extreme-d4 bucket:", anyflag.groupby(pd.cut(ext, [-1, 0, 1, 2, 4, 40])).mean().round(3).to_dict())
for f in flags:
    print(f"  {f}: rate overall {X[f].mean():.3f}, among rows with >=3 extreme d4: {X.loc[ext >= 3, f].mean():.3f}")

# mixture on the d4 features
cols = [c for c in Dz.columns if c.startswith("d4_")]
M = Dz[cols].fillna(0).clip(-8, 8)
P = PCA(12, random_state=SEED).fit_transform(M)
for k in (1, 2, 3, 4, 5):
    g = GaussianMixture(k, covariance_type="full", random_state=SEED, n_init=2).fit(P)
    print(f"GMM(d4 PCA12) k={k}: BIC={g.bic(P):.0f} weights={np.round(np.sort(g.weights_)[::-1], 3).tolist()}")
Dz.to_pickle(OUT / "task5_dz.pkl")
