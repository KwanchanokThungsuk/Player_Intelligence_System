"""Round 3 - Task 3 two-stage model (P(y=0) + lognormal median of the positive part) with the exact-identity features of round-2/t3_fe2.py (fe()).
Same 5-fold split as t3_twostage.py / t3_fe2.py. Trained from scratch on the Task 3 train file only. Writes OOF and test predictions to data2/.
Final Task 3 blend (OOF NMAE 0.79071, previous blend 0.78962): 0.55 * this model + 0.25 * round-2 two-stage (raw features) + 0.20 * round-2 L1 LightGBM with identity features, clipped to [0, 300000]
(see build_round3_t3t5_update.py).  Run from anywhere:  python t3_twostage_fe.py   (needs CPE342_DATA for the data folder)."""
import sys, os, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
os.environ.setdefault("CPE342_DATA", "C:/Users/Wirachat/Desktop/CPE342-DATA-v2/kaggle/cpe342-kaggle-main")
R3 = Path(__file__).resolve().parent; R2 = R3.parent / "round-2"; R1 = R3.parent / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load, nmae_score, SEED
OUT = R3 / "data2"; OUT.mkdir(exist_ok=True)
# t3_fe2.py up to the fold definition: loads the data, recovers the segment where it is missing, defines fe()
pre = open(R2 / "t3_fe2.py", encoding="utf8").read().split("strat = (")[0].replace("n_jobs=12", "n_jobs=8")
_file = globals().get("__file__"); globals()["__file__"] = str(R2 / "t3_fe2.py")
exec(compile(pre, "t3_fe2_pre", "exec"))                                   # gives tr, te, y, fe, seg_tr, seg_te_hat, base_cols
globals()["__file__"] = _file; R3 = Path(__file__).resolve().parent; R2 = R3.parent / "round-2"; R1 = R3.parent / "round-1"; OUT = R3 / "data2"
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr, (tr.segment.fillna(-1).astype(int).astype(str) + (y > 0).astype(int).astype(str)).values))
Xa, Xt = fe(tr, seg_tr), fe(te, te.segment.fillna(seg_te_hat))
base = dict(learning_rate=0.05, num_leaves=63, min_child_samples=30, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=SEED, n_jobs=8)
pos = y > 0
pz = np.zeros(len(Xa)); mu = np.full(len(Xa), np.nan); pz_te = np.zeros(len(Xt)); mu_te = np.zeros(len(Xt))
for k, (a, b) in enumerate(folds):
    m1 = lgb.train({**base, "objective": "binary"}, lgb.Dataset(Xa.iloc[a], (y[a] > 0).astype(int)), 3000, valid_sets=[lgb.Dataset(Xa.iloc[b], (y[b] > 0).astype(int))], callbacks=[lgb.early_stopping(150, verbose=False)])
    pz[b] = 1 - m1.predict(Xa.iloc[b], num_iteration=m1.best_iteration); pz_te += (1 - m1.predict(Xt, num_iteration=m1.best_iteration)) / 5
    ap, bp = a[pos[a]], b[pos[b]]
    m2 = lgb.train({**base, "objective": "l2"}, lgb.Dataset(Xa.iloc[ap], np.log(y[ap])), 3000, valid_sets=[lgb.Dataset(Xa.iloc[bp], np.log(y[bp]))], callbacks=[lgb.early_stopping(150, verbose=False)])
    mu[b] = m2.predict(Xa.iloc[b], num_iteration=m2.best_iteration); mu_te += m2.predict(Xt, num_iteration=m2.best_iteration) / 5
    print(f"fold {k}: zero-AUC {roc_auc_score(y[b] > 0, 1 - pz[b]):.4f} | positive log-rmse {np.sqrt(np.mean((np.log(y[bp]) - mu[bp]) ** 2)):.4f}", flush=True)
seg = tr.segment.fillna(4).astype(int).values
sig = {s: float(np.std((np.log(y) - mu)[(seg == s) & pos])) for s in range(5)}; sigma = np.array([sig[s] for s in seg])
def median_pred(pz_, mu_, sigma_):
    q = np.clip((0.5 - pz_) / np.maximum(1 - pz_, 1e-9), 1e-4, 1 - 1e-4)
    return np.where(pz_ >= 0.5, 0.0, np.clip(np.exp(mu_ + sigma_ * norm.ppf(q)), 0, 300000))
oof = median_pred(pz, mu, sigma); test = median_pred(pz_te, mu_te, np.array([sig[s] for s in te.segment.fillna(4).astype(int).values]))
print(f"two-stage with FE features: OOF NMAE {nmae_score(y, oof):.5f} (raw-feature two-stage 0.78838) | sigma by segment {({s: round(v, 3) for s, v in sig.items()})}")
np.save(OUT / "t3_twostage_fe_oof.npy", oof); np.save(OUT / "t3_twostage_fe_test.npy", test)
# blend check against the saved round-1/2 arrays
two, fe2 = np.load(R2 / "t3_twostage_oof.npy"), np.load(R2 / "t3_fe2_oof.npy")
bl = np.clip(0.55 * oof + 0.25 * two + 0.20 * fe2, 0, 300000)
print(f"blend 0.55/0.25/0.20 OOF NMAE {nmae_score(y, bl):.5f} (current blend 0.5 two + 0.5 fe2: {nmae_score(y, np.clip(0.5 * two + 0.5 * fe2, 0, 300000)):.5f})")
print("test prediction summary: mean %.1f median %.1f zeros %d (OOF mean %.1f)" % (test.mean(), np.median(test), int((test == 0).sum()), oof.mean()))
