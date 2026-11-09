"""Round 2 - Task 5: search for SUBTLE anomalies beyond the obvious period-4 group (unsupervised, read-only)."""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load, SEED, OUT as R1_OUT

OUT = Path(__file__).resolve().parent
te = load(5, "test"); X = te.drop(columns=["player_id"])
flags = ["password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"]
bases = sorted({re.sub(r"_[1-4]$", "", c) for c in X.columns if re.search(r"_[1-4]$", c)})
S = pd.read_pickle(R1_OUT / "task5_S.pkl")["S"].fillna(0)
A = (S > 130).values  # obvious group from round 1


def rz(s):
    med, iqr = s.median(), s.quantile(.75) - s.quantile(.25)
    return (s - med) / (iqr / 1.349 if iqr > 0 else (s.std() or 1))


t = np.array([0.0, 1.0, 2.0, 3.0])
F = {}
for b in bases:
    M = X[[f"{b}_{i}" for i in (1, 2, 3, 4)]].values.astype(float)
    ok = ~np.isnan(M)
    cnt = ok.sum(1)
    mean = np.nanmean(M, axis=1)
    # slope with available points (>=3)
    tt = np.where(ok, t, np.nan)
    tm = np.nanmean(tt, axis=1)
    num = np.nansum((tt - tm[:, None]) * (M - mean[:, None]), axis=1); den = np.nansum((tt - tm[:, None]) ** 2, axis=1)
    slope = np.where(cnt >= 3, num / np.where(den > 0, den, np.nan), np.nan)
    spread = np.where(cnt >= 3, np.nanstd(M, axis=1), np.nan)
    F[f"mean_{b}"] = mean; F[f"slope_{b}"] = slope; F[f"spread_{b}"] = spread
    F[f"curv_{b}"] = np.where(cnt == 4, M[:, 0] - M[:, 1] - M[:, 2] + M[:, 3], np.nan)
F = pd.DataFrame(F)
Z = F.apply(rz)
Zr = Z[~A]
print(f"rows outside obvious group: {int((~A).sum())}")
exp = 2 * (1 - 0.99977)  # P(|z|>3.5) under N(0,1) ~ 0.000465
print(f"expected P(|z|>3.5) under Gaussian = {2*chi2.sf(3.5**2,1)/1:.5f}")
tail = (Zr.abs() > 3.5).mean()
for g in ("mean", "slope", "spread", "curv"):
    s = tail[[c for c in Z.columns if c.startswith(g + "_")]].sort_values(ascending=False)
    print(f"[{g}] mean tail frac {s.mean():.5f} | top: " + ", ".join(f"{k.split('_',1)[1]}={v:.4f}" for k, v in s.head(5).items()))

# multivariate whitened distance on groups (outside obvious group)
def t2_report(cols, label, maxpc=60):
    M = Zr[cols].fillna(0).clip(-6, 6).values
    p = PCA(min(maxpc, M.shape[1]), random_state=SEED).fit(M)
    W = p.transform(M) / np.sqrt(p.explained_variance_)
    d = (W ** 2).sum(1); k = W.shape[1]
    q = np.quantile(d, [.5, .9, .99, .999])
    print(f"[{label}] k={k}: T2 quantiles 50/90/99/99.9 = {np.round(q,1).tolist()} | chi2 expected = {np.round(chi2.ppf([.5,.9,.99,.999], k),1).tolist()} | frac above chi2 99.9% = {(d > chi2.ppf(.999, k)).mean():.4f}")
    return d

print()
d_mean = t2_report([c for c in Z.columns if c.startswith("mean_")], "level means")
d_slope = t2_report([c for c in Z.columns if c.startswith("slope_")], "slopes")
d_spread = t2_report([c for c in Z.columns if c.startswith("spread_")], "spreads")
d_all = t2_report(list(Z.columns), "all 116 features", 80)

# mixture structure on the combined space (outside obvious group)
M = Zr.fillna(0).clip(-6, 6).values
P = PCA(25, random_state=SEED).fit_transform(M)
for k in (1, 2, 3):
    g = GaussianMixture(k, covariance_type="diag", random_state=SEED, n_init=2).fit(P)
    print(f"GMM(diag, 25 PCs) k={k}: BIC={g.bic(P):.0f} weights={np.round(np.sort(g.weights_)[::-1], 3).tolist()}")

# are flags related to any of the new feature groups (outside the obvious group)?
print("\nflag vs new features (outside obvious group): top |standardised mean diff| per flag")
for f in flags:
    m = X.loc[~A, f] == 1
    diffs = pd.Series({c: (Zr.loc[m, c].mean() - Zr.loc[~m, c].mean()) for c in Z.columns}).abs().sort_values(ascending=False)
    print(f"  {f}: n={int(m.sum())}; top: " + ", ".join(f"{k}={v:.2f}" for k, v in diffs.head(3).items()))
F.to_pickle(OUT / "task5_subtle_features.pkl"); Z.to_pickle(OUT / "task5_subtle_z.pkl")
