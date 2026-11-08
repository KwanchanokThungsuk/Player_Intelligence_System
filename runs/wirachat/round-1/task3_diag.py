import sys
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from r1_common import load, OUT, nmae_score
tr = load(3, "train"); y = tr.spending_30d.values
oof = np.load(OUT / "task3_oof_l1_raw.npy")
err = np.abs(y - oof); tot = err.sum()
tr["err"] = err; tr["pred"] = oof
print("share of total abs error by segment:", (tr.groupby("segment", dropna=False).err.sum() / tot).round(3).to_dict())
print("rows by segment:", tr.segment.value_counts(dropna=False).sort_index().to_dict())
print("share of sum(y) by segment:", (tr.groupby("segment", dropna=False).spending_30d.sum() / y.sum()).round(3).to_dict())
q = pd.qcut(pd.Series(y).rank(method="first"), 20, labels=False)
print("abs-error share by y-vigintile (top 3):", (pd.Series(err).groupby(q).sum() / tot).round(3).tail(3).to_dict())
for c in ["prev_month_spending", "historical_spending", "segment", "vip_status", "event_participation_rate"]:
    print(f"  err share for rows with NaN {c}: {tr.loc[tr[c].isna(), 'err'].sum() / tot:.3f}  (rows frac {tr[c].isna().mean():.3f})")
s3 = tr[tr.segment == 3]
print("segment 3: y quantiles", s3.spending_30d.quantile([.05, .25, .5, .75, .95]).round(0).to_dict())
pos = s3[(s3.prev_month_spending > 0) & (s3.spending_30d > 0)]
r = pos.spending_30d / pos.prev_month_spending
print("segment 3, y/prev_month ratio quantiles:", r.quantile([.05, .25, .5, .75, .95]).round(2).to_dict(), "| corr(y, prev)=", round(pos.spending_30d.corr(pos.prev_month_spending), 3))
r2 = (pos.spending_30d / pos.historical_spending.replace(0, np.nan)).dropna()
print("segment 3, y/historical ratio quantiles:", r2.quantile([.05, .25, .5, .75, .95]).round(2).to_dict())
print("rel err (|e|/y) median among y>1000:", round((tr[tr.spending_30d > 1000].err / tr[tr.spending_30d > 1000].spending_30d).median(), 3))
print("segment value NaN rows:", tr.segment.isna().sum(), "| segment unique:", sorted(tr.segment.dropna().unique()))
