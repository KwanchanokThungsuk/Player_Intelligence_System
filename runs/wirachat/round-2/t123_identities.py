"""Round 2 - search for exact / near-exact relations C = f(A, B) between numeric columns (products, ratios, sums, differences)."""
import sys, warnings, itertools
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load


def search(df, name, thr=0.985):
    num = df.select_dtypes(include=np.number)
    num = num[[c for c in num.columns if num[c].nunique() > 6 and not c.startswith("random_metric")]]
    cc = num.dropna()
    if len(cc) > 20000: cc = cc.sample(20000, random_state=1)
    print(f"\n=== {name}: {num.shape[1]} numeric columns, complete rows used: {len(cc)}")
    cols = list(cc.columns)
    V = cc.values.astype(float)
    pos = (V > 0).all(axis=0)
    Zs = (V - V.mean(0)) / (V.std(0) + 1e-12)
    L = np.where(pos, np.log(np.where(V > 0, V, 1)), np.nan)
    Zl = (L - np.nanmean(L, 0)) / (np.nanstd(L, 0) + 1e-12)
    found = []
    for i, j in itertools.combinations(range(len(cols)), 2):
        ops = {"A+B": V[:, i] + V[:, j], "A-B": V[:, i] - V[:, j]}
        for opn, f in ops.items():
            if f.std() == 0: continue
            fz = (f - f.mean()) / f.std()
            cors = np.abs(Zs.T @ fz) / len(fz)
            for k in np.where(cors > thr)[0]:
                if k not in (i, j): found.append((cols[k], opn.replace("A", cols[i]).replace("B", cols[j]), round(float(cors[k]), 4)))
        if pos[i] and pos[j]:
            for opn, f in {"A*B": L[:, i] + L[:, j], "A/B": L[:, i] - L[:, j]}.items():
                if np.std(f) == 0: continue
                fz = (f - f.mean()) / f.std()
                for k in range(len(cols)):
                    if k in (i, j) or not pos[k]: continue
                    c = abs(np.dot(Zl[:, k], fz)) / len(fz)
                    if c > thr: found.append((cols[k], opn.replace("A", cols[i]).replace("B", cols[j]) + " (log space)", round(float(c), 4)))
    found = sorted(found, key=lambda x: -x[2])
    print(f"relations with |corr| > {thr}: {len(found)}")
    for t, rel, c in found[:15]: print(f"  {t} ~ {rel}   corr={c}")
    return found


search(load(3, "train").drop(columns=["spending_30d"]), "Task 3")
search(load(2, "train").drop(columns=["segment"]), "Task 2")
search(load(1, "train").drop(columns=["is_cheater"]), "Task 1")
