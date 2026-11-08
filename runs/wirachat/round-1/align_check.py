import sys
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from r1_common import load
t2, t3, t1 = load(2, "train"), load(3, "train"), load(1, "train")
m = t3.segment.notna()
agree = (t2.segment[m].values == t3.segment[m].values).mean()
p = t2.segment.value_counts(normalize=True)
print(f"row-index agreement of Task2.segment vs Task3.segment (train, n={int(m.sum())}): {agree:.4f} | chance level ~{(p**2).sum():.3f}")
# also: does shuffling destroy it? (sanity) and are the two tables aligned on other shared-ish columns?
shared = [c for c in t2.columns if c in t3.columns and c not in ("player_id", "segment")]
print("columns shared by name between Task2 and Task3 train:", shared)
for c in shared[:6]:
    a, b = t2[c], t3[c]
    if a.dtype != object:
        ok = a.notna() & b.notna()
        print(f"  {c}: row-aligned corr = {np.corrcoef(a[ok], b[ok])[0,1]:.3f}")
shared13 = [c for c in t1.columns if c in t3.columns and c not in ("player_id",)]
print("shared Task1/Task3:", shared13)
