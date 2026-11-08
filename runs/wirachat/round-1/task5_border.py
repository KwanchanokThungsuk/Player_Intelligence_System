import sys
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from r1_common import load, OUT
te = load(5, "test"); X = te.drop(columns=["player_id"])
S = pd.read_pickle(OUT / "task5_S.pkl")["S"]; Dz = pd.read_pickle(OUT / "task5_dz.pkl"); d4 = Dz.filter(like="d4_")
flags = ["password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"]
for lo, hi, name in [(60, 100, "S 60-100"), (100, 219, "S 100-219 (gap/border)"), (219, 1e9, "S>=219 (cluster)")]:
    m = (S > lo) & (S <= hi)
    prof = d4[m].abs().mean().sort_values(ascending=False).head(5)
    print(f"{name}: n={int(m.sum())} | top mean|z|: " + ", ".join(f"{k.replace('d4_','')}={v:.1f}" for k, v in prof.items()) +
          " | flag rates: " + ", ".join(f"{f[:8]}={X.loc[m, f].mean():.2f}" for f in flags))
print("overall flag rates:", ", ".join(f"{f[:8]}={X[f].mean():.2f}" for f in flags))
loc = d4[["d4_login_lat", "d4_login_lon", "d4_location_changes"]].abs()
print("location-only-extreme rows (loc z>4 but all other d4 |z|<3):",
      int(((loc.max(axis=1) > 4) & (d4.drop(columns=loc.columns).abs().max(axis=1) < 3)).sum()))
