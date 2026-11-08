"""Round 1 - Task 5: characterise the extreme-change group and look for other subtypes (read-only)."""
import sys
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from r1_common import load, SEED, OUT

te = load(5, "test"); X = te.drop(columns=["player_id"])
flags = ["password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"]
Dz = pd.read_pickle(OUT / "task5_dz.pkl")
d4 = Dz.filter(like="d4_")
ext = (d4.abs() > 3.5).sum(axis=1)
A = ext >= 4
print("group A (>=4 extreme d4):", int(A.sum()), f"({A.mean():.3%})")
prof = pd.DataFrame({"meanZ_A": d4[A].mean(), "meanZ_rest": d4[~A].mean(), "frac_ext_A": (d4[A].abs() > 3.5).mean()})
print("\nd4 feature profile of group A (sorted by |mean z|):")
print(prof.reindex(prof.meanZ_A.abs().sort_values(ascending=False).index).head(14).round(2).to_string())
print("\nflag rates A vs rest:", {f: (round(X.loc[A, f].mean(), 3), round(X.loc[~A, f].mean(), 3)) for f in flags})
print("static: account_age mean A/rest:", round(X.loc[A, "account_age_days"].mean()), round(X.loc[~A, "account_age_days"].mean()),
      "| behavior_stability A/rest:", round(X.loc[A, "behavior_stability"].mean(), 3), round(X.loc[~A, "behavior_stability"].mean(), 3))

# split group A in two types?
M = d4.fillna(0).clip(-8, 8)
cl = GaussianMixture(2, random_state=SEED, n_init=3).fit(M[A])
lab = cl.predict(M[A])
for k in (0, 1):
    idx = A[A].index[lab == k]
    top = d4.loc[idx].mean().abs().sort_values(ascending=False).head(5)
    print(f"A-type {k}: n={len(idx)}; top mean-z features: {', '.join(f'{i}={d4.loc[idx, i].mean():+.1f}' for i in top.index)}; flags: " +
          ", ".join(f"{f}={X.loc[idx, f].mean():.2f}" for f in flags))

# is there another subtype outside A?  look at sd_ (variability) and static/flags among rest
rest = Dz[~A]
sd = rest.filter(like="sd_")
print("\nrest: fraction |z|>4 for sd_ features (variability across periods):")
print((sd.abs() > 4).mean().sort_values(ascending=False).head(6).round(4).to_string())
Zr = pd.concat([rest, ((X.loc[~A, ["account_age_days", "behavior_stability"]] - X[["account_age_days", "behavior_stability"]].median()) / X[["account_age_days", "behavior_stability"]].std())], axis=1).fillna(0).clip(-8, 8)
P = PCA(15, random_state=SEED).fit_transform(Zr)
for k in (1, 2, 3):
    g = GaussianMixture(k, covariance_type="full", random_state=SEED, n_init=2).fit(P)
    print(f"GMM on rest k={k}: BIC={g.bic(P):.0f} weights={np.round(np.sort(g.weights_)[::-1], 3).tolist()}")
for f in flags:
    print(f"rest, {f}=1 rate {X.loc[~A, f].mean():.3f}; behavior_stability mean when flag=1 vs 0: {X.loc[(~A) & (X[f] == 1), 'behavior_stability'].mean():.3f} vs {X.loc[(~A) & (X[f] == 0), 'behavior_stability'].mean():.3f}")
