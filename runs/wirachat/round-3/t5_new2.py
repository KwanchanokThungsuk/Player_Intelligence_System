"""Round 3 - Task 5 NEW data: build group memberships from the gap columns (family rule: odd in >=2 periods; static columns: 1), re-scan the rest, write flags."""
import sys, os, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
os.environ.setdefault("CPE342_DATA", "C:/Users/Wirachat/Desktop/CPE342-DATA-v2/kaggle/cpe342-kaggle-main")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "round-1"))
from r1_common import load
OUT = Path(__file__).resolve().parent / "data2"
te = load(5, "test"); X = te.drop(columns=["player_id"]); n = len(X)
A = np.load(OUT / "task5_A.npy"); R = pd.read_csv(OUT / "task5_gap_columns.csv")
R = R[(R.n_small_side <= 700) & (R.side == "low")]                       # drop discrete zero-mass columns (friend_removes) and 'high' sides
fam = {}
for _, r in R.iterrows():
    b = re.sub(r"_[1-4]$", "", r.column); fam.setdefault(b, []).append(r.column)
print("families with a gapped low group:", {b: len(c) for b, c in fam.items()})
def odd(col):                                                             # use the column's own gap cut
    v = X.loc[~A, col].dropna().values; hh, ee = np.histogram(v, bins=80); nz = np.where(hh > 0)[0]
    ne, cut = max(((i1 - i0 - 1, (ee[i0 + 1] + ee[i1]) / 2) for i0, i1 in zip(nz[:-1], nz[1:])), key=lambda t: t[0])
    return (X[col] < cut).fillna(False).values
G = {}
for b, cols in fam.items():
    allcols = [f"{b}_{p}" for p in (1, 2, 3, 4)] if f"{b}_1" in X.columns else [b]
    cnt = sum(odd(c).astype(int) for c in allcols)
    G[b] = cnt >= (2 if len(allcols) > 1 else 1)
for b, m in G.items(): print(f"  family {b:24s}: {int(m.sum()):4d} rows | overlap with A {int((m & A).sum())}")
names = list(G); M = np.column_stack([G[b] for b in names] + [A])
print("rows in more than one group:", int((M.sum(1) > 1).sum()))
flag = M.any(1)
print(f"TOTAL flagged: {int(flag.sum())} ({flag.mean():.2%}) | A {int(A.sum())} + families {int(flag.sum() - A.sum())}  [old data: 1,819 = 835 + 984]")
# second scan on the remainder
rest = ~flag; rows = []
for c in X.columns:
    v = X.loc[rest, c].dropna().values
    if len(np.unique(v)) < 25: continue
    hh, ee = np.histogram(v, bins=100); nz = np.where(hh > 0)[0]
    g = [(i1 - i0 - 1, (ee[i0 + 1] + ee[i1]) / 2) for i0, i1 in zip(nz[:-1], nz[1:])]
    if g:
        ne, cut = max(g); sm = int(min((v < cut).sum(), (v > cut).sum()))
        if ne >= 3 and sm >= 40 and sm < 700: rows.append((c, ne, round(cut, 3), sm))
print("remaining gapped columns after removing the flagged rows:", rows or "none")
np.save(OUT / "task5_flags.npy", flag.astype(int)); pd.DataFrame(M, columns=names + ["A"]).to_pickle(OUT / "task5_groups.pkl")
