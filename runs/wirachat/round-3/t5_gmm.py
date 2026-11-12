"""Round 3 - Task 5: purely unsupervised mixture model over all features (rank-normal scores, PCA) to look for components beyond the known groups.
No self-labels are used for fitting; the known groups are used only afterwards to describe the components."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
warnings.filterwarnings("ignore")
R2 = Path(__file__).resolve().parents[1] / "round-2"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "round-1"))
from r1_common import load, SEED
te = load(5, "test"); X = te.drop(columns=["player_id"])
G = pd.read_pickle(R2 / "task5_groups.pkl"); known = G.any(axis=1).values
flags = ["password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"]
cols = [c for c in X.columns if c not in flags]
# rank-normal transform of every continuous column (robust to the non-Gaussian marginals), missing -> 0 (the median score)
Z = pd.DataFrame({c: norm.ppf((X[c].rank(pct=True, na_option="keep") * X[c].notna().sum() / (X[c].notna().sum() + 1)).clip(1e-4, 1 - 1e-4)) for c in cols}).fillna(0).values
P = PCA(40, random_state=SEED).fit_transform(Z)
print("PCA explained variance of 40 comps:", round(float(PCA(40, random_state=SEED).fit(Z).explained_variance_ratio_.sum()), 3))
best = None
for k in (4, 6, 8, 10, 12):
    g = GaussianMixture(k, covariance_type="diag", random_state=SEED, n_init=2, max_iter=200).fit(P)
    b = g.bic(P); lab = g.predict(P); w = np.bincount(lab, minlength=k) / len(lab)
    print(f"K={k:2d} BIC={b:,.0f} | component sizes (%): {np.round(np.sort(w)[::-1] * 100, 1).tolist()}")
    if best is None or b < best[0]: best = (b, k, g, lab)
b, k, g, lab = best
print(f"\nbest K by BIC: {k}")
for c in range(k):
    m = lab == c
    print(f"  comp {c}: {int(m.sum()):6d} rows ({m.mean()*100:5.2f}%) | overlap with known anomaly groups: {int((m & known).sum()):5d} ({(m & known).sum()/max(m.sum(),1)*100:5.1f}% of the component) | group A {int((m & G['A'].values).sum())}, ip_ltv {int((m & G['ip_ltv'].values).sum())}, comm {int((m & G['comm'].values).sum())}, act {int((m & G['act'].values).sum())}, bs {int((m & G['bs'].values).sum())}")
small = [c for c in range(k) if (lab == c).mean() < 0.05]
print("\ncomponents below 5% of the rows that are NOT mostly known groups:", [c for c in small if (((lab == c) & known).sum() / max((lab == c).sum(), 1)) < 0.5])
