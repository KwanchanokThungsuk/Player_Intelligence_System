"""Round 3 - Task 5 on the NEW dataset (2026-11-12): re-run the unsupervised anomaly search from scratch (no old thresholds reused):
(1) period-4 change score S with a natural gap, (2) persistent gap groups across all columns, (3) flags. Unsupervised statistics on the Task 5 test file only."""
import sys, os, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
os.environ.setdefault("CPE342_DATA", "C:/Users/Wirachat/Desktop/CPE342-DATA-v2/kaggle/cpe342-kaggle-main")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "round-1"))
from r1_common import load
OUT = Path(__file__).resolve().parent / "data2"; OUT.mkdir(exist_ok=True)
te = load(5, "test"); X = te.drop(columns=["player_id"]); n = len(X)
bases = sorted({re.sub(r"_[1-4]$", "", c) for c in X.columns if re.search(r"_[1-4]$", c)})
def rz(s):
    med, iqr = s.median(), s.quantile(.75) - s.quantile(.25); return (s - med) / (iqr / 1.349 if iqr > 0 else (s.std() or 1))
# ---- (1) period-4 change score
d4 = pd.DataFrame({b: X[f"{b}_4"] - X[[f"{b}_1", f"{b}_2", f"{b}_3"]].mean(axis=1) for b in bases}).apply(rz)
nav = d4.notna().sum(axis=1); S = ((d4.clip(-8, 8) ** 2).sum(axis=1, min_count=1) / nav * d4.shape[1]).fillna(0)
h, e = np.histogram(np.log10(S[S > 0]), bins=30)
print("log10(S) histogram:", ", ".join(f"{10**a:.0f}:{c}" for a, c in zip(e[:-1], h)))
v = np.sort(S.values); hi = v[v > 60]; gaps = np.diff(np.log10(hi)); k = int(np.argmax(gaps)); cut = float(np.sqrt(hi[k] * hi[k + 1]))
print(f"largest gap in S above 60: between {hi[k]:.1f} and {hi[k+1]:.1f} (log-gap {gaps[k]:.2f}) -> threshold {cut:.0f}")
A = (S > cut).values
print(f"group A (period-4 change): {int(A.sum())} rows ({A.mean():.2%}) | old data: 835")
# ---- (2) gap scan on every continuous column outside A
def gap(v, bins=80):
    hh, ee = np.histogram(v, bins=bins); nz = np.where(hh > 0)[0]
    g = [(i1 - i0 - 1, (ee[i0 + 1] + ee[i1]) / 2) for i0, i1 in zip(nz[:-1], nz[1:])]
    if not g: return 0, None, 0, None
    ne, c = max(g); low = (v < c).sum() < (v > c).sum(); return ne, c, int(min((v < c).sum(), (v > c).sum())), "low" if low else "high"
rows = []
for c in X.columns:
    vv = X.loc[~A, c].dropna().values
    if len(np.unique(vv)) < 25: continue
    ne, cc, sm, side = gap(vv)
    if ne >= 4 and sm >= 30: rows.append((c, ne, round(cc, 3), sm, side))
R = pd.DataFrame(rows, columns=["column", "empty_bins", "cut", "n_small_side", "side"]).sort_values("empty_bins", ascending=False)
print(f"\ncolumns with an internal gap outside A (>=4 empty bins of 80, small side >= 30): {len(R)}")
print(R.head(30).to_string(index=False))
R.to_csv(OUT / "task5_gap_columns.csv", index=False)
np.save(OUT / "task5_A.npy", A); np.save(OUT / "task5_S.npy", S.values)
