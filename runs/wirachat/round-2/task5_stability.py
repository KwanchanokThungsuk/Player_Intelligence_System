"""Round 2 - Task 5: behavior_stability heavy tail - is it a separate population? (unsupervised, read-only)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load, OUT as R1_OUT

te = load(5, "test"); X = te.drop(columns=["player_id"])
flags = ["password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"]
S = pd.read_pickle(R1_OUT / "task5_S.pkl")["S"].fillna(0)
A = (S > 130).values
bs = X.behavior_stability
print("behavior_stability: NaN", int(bs.isna().sum()), "| quantiles .1/1/5/25/50/75/95/99/99.9:", bs.quantile([.001, .01, .05, .25, .5, .75, .95, .99, .999]).round(3).tolist(), "| min/max", round(bs.min(), 3), round(bs.max(), 3))
h, e = np.histogram(bs.dropna(), bins=30)
print("histogram (edge:count):", ", ".join(f"{a:.2f}:{c}" for a, c in zip(e[:-1], h)))
for thr in (0.7, 0.6, 0.5, 0.4):
    m = bs < thr
    print(f"bs<{thr}: n={int(m.sum())} ({m.mean():.3%}) | in obvious group A: {int((m & A).sum())}")
core = bs[(bs >= 0.7)]
print("core (bs>=0.7): mean", round(core.mean(), 3), "std", round(core.std(), 3), "| skew", round(core.skew(), 2))

low = (bs < 0.6).fillna(False).values & ~A
rest = ~low & ~A
print(f"\nlow-stability group (bs<0.6, outside A): n={int(low.sum())}")
print("flag rates low vs rest:", {f: (round(X.loc[low, f].mean(), 3), round(X.loc[rest, f].mean(), 3)) for f in flags})
diff = {}
for c in X.columns:
    if c in flags or c == "behavior_stability": continue
    a, b = X.loc[low, c].dropna(), X.loc[rest, c].dropna()
    sd = X.loc[rest, c].std()
    if sd and len(a) > 20: diff[c] = (a.mean() - b.mean()) / sd
d = pd.Series(diff); d = d.reindex(d.abs().sort_values(ascending=False).index)
print("largest standardised mean differences (low - rest):", ", ".join(f"{k}={v:+.2f}" for k, v in d.head(10).items()))
print("S score for low group quantiles (.5/.9/.99):", S[low].quantile([.5, .9, .99]).round(1).tolist(), "| rest:", S[rest].quantile([.5, .9, .99]).round(1).tolist())
