"""Add Task 5 flags for rows whose behavior_stability is NaN (invisible to the gap rule) but whose side effects (purchase_frequency up, login_count down, account_age up)
mark them as members of the low-behavior_stability group. Rule: flag a row when P(group member) > 0.09 (~ F3/10: for F3 = 10TP/(9T+n) adding a flag helps iff precision > F3/10).
Needs data2/task5_nanstab_prob.npy from t5_nanstab.py.  usage: python build_round3_t5_nanstab.py <base_submission_name> <out_name> [threshold]"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
R3 = Path(__file__).resolve().parent
base, out = sys.argv[1], sys.argv[2]; thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.09
s = pd.read_csv(R3 / f"submission_{base}.csv"); p = np.load(R3 / "data2/task5_nanstab_prob.npy"); assert len(p) == len(s)
add = (p > thr) & (s.task5.values == 0)
print(f"rows added: {int(add.sum())} | expected true members among them {p[add].sum():.1f} | expected F3 change {(10 * p[add].sum() - 0.883 * add.sum()) / (9 * 2070 + 1806):+.4f} (Task 5 score)")
s.loc[add, "task5"] = 1; print("Task 5 flags:", int(s.task5.sum()))
s.to_csv(R3 / f"submission_{out}.csv", index=False)
