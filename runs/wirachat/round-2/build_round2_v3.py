"""Round 2 - candidate V3: V1 (Task 1 thr 0.18) with Task 5 = obvious period-4 group + four persistent odd groups.
Local file only; nothing is uploaded."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parent
v1 = pd.read_csv(OUT / "submission_v1_t1_thr018.csv")
G = pd.read_pickle(OUT / "task5_groups.pkl")
flag = (G["A"] | G["ip_ltv"] | G["comm"] | G["act"] | G["bs"]).values.astype(int)
assert len(flag) == len(v1) == 25889
v3 = v1.copy(); v3["task5"] = flag
changed = {c: int((v3[c].astype(str) != v1[c].astype(str)).sum()) for c in v1.columns}
assert [c for c, n in changed.items() if n > 0] == ["task5"], changed
v3.to_csv(OUT / "submission_v3_t5_groups.csv", index=False)
print("changed rows vs V1:", changed)
print(f"Task 5 flagged: {int(flag.sum())} ({flag.mean():.2%}) vs V1 {int(v1.task5.sum())} -> added {int(flag.sum() - v1.task5.sum())}")
print("flag breakdown:", {k: int(G[k].sum()) for k in G.columns}, "| overlap between groups:", int((G[['ip_ltv','comm','act','bs','A']].sum(axis=1) > 1).sum()))
print("rows:", v3.shape, "| ids unique:", v3.id.is_unique, "| NaN:", int(v3.isna().sum().sum()))
