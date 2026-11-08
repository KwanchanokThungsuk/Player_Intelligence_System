import sys
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from r1_common import load, OUT
te = load(5, "test"); X = te.drop(columns=["player_id"])
Dz = pd.read_pickle(OUT / "task5_dz.pkl"); d4 = Dz.filter(like="d4_"); S = pd.read_pickle(OUT / "task5_S.pkl")["S"]
locc = ["d4_login_lat", "d4_login_lon", "d4_location_changes"]
nonloc = d4.drop(columns=locc)
cap = 64.0
Sn = (nonloc.clip(-8, 8) ** 2).sum(axis=1, min_count=1) / nonloc.notna().sum(axis=1) * nonloc.shape[1]
Sl = (d4[locc].clip(-8, 8) ** 2).sum(axis=1, min_count=1)
only_loc = (d4[locc].abs().max(axis=1) > 4) & (nonloc.abs().max(axis=1) < 3)
print("location-only rows:", int(only_loc.sum()), "| their S quantiles:", S[only_loc].quantile([.5, .9, .99, 1]).round(0).to_dict(), "| flagged by S>130:", int((S[only_loc] > 130).sum()))
print("location z of those rows (max abs) quantiles:", d4.loc[only_loc, locc].abs().max(axis=1).quantile([.5, .9, 1]).round(1).to_dict())
flag = S > 130
print("flagged set:", int(flag.sum()), "| of which non-location S_nonloc > 100:", int((flag & (Sn > 100)).sum()), "| flagged with S_nonloc<=60:", int((flag & (Sn <= 60)).sum()))
h, e = np.histogram(np.log10(Sn.dropna()), bins=22)
print("S_nonloc histogram:", ", ".join(f"{10**a:.0f}:{c}" for a, c in zip(e[:-1], h)))
# how extreme are location z in the cluster vs only_loc rows?
print("cluster location max|z| quantiles:", d4.loc[flag, locc].abs().max(axis=1).quantile([.1, .5, .9]).round(1).to_dict())
