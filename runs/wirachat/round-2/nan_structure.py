"""Round 2 - is the missingness structured (co-occurring across columns = latent groups) or independent per cell? Tasks 1, 2, 3, 5, train and test."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binom
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "round-1"))
from r1_common import load
def report(name, df, drop):
    X = df.drop(columns=[c for c in drop if c in df.columns]); M = X.isna().values.astype(float)
    p = M.mean(0); n_cols = M.shape[1]
    k = M.sum(1)
    exp = binom.pmf(np.arange(0, 8), n_cols, p.mean()) * len(M)      # (approx.) independent-cell expectation with the mean rate
    obs = np.bincount(k.astype(int), minlength=8)[:8]
    C = np.corrcoef(M[:, p > 0].T); off = np.abs(C[np.triu_indices_from(C, 1)])
    print(f"{name:14s} NaN rate {p.mean():.3f} | max |corr| between NaN indicators {np.nanmax(off):.3f}, mean {np.nanmean(off):.4f} | #NaN per row observed {obs[:6].tolist()} vs independent-binomial {np.round(exp[:6]).astype(int).tolist()}")
for t, tgt in [(1, "is_cheater"), (2, "segment"), (3, "spending_30d")]:
    report(f"T{t} train", load(t, "train"), ["player_id", tgt]); report(f"T{t} test", load(t, "test"), ["player_id"])
report("T5 test", load(5, "test"), ["player_id"])
# Task 1: do the train-heavy-missing columns define a sub-population with a different label rate / feature distribution?
tr = load(1, "train"); lab = tr.dropna(subset=["is_cheater"])
heavy = ["avg_session_length_min", "aiming_smoothness", "friend_network_size", "account_age_days", "movement_pattern_score", "team_play_score", "survival_time_avg"]
for c in heavy[:4]:
    m = lab[c].isna()
    print(f"T1 rows with {c} missing: {m.mean():.3f} of labelled train, cheater rate {lab.loc[m,'is_cheater'].mean():.3f} vs {lab.loc[~m,'is_cheater'].mean():.3f}")
