"""Round 1 - quick EDA for Tasks 1, 2, 3, 5. Read-only on the data."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "cpe342-kaggle-main"
pd.set_option("display.width", 200, "display.max_columns", 50, "display.max_rows", 100)


def load(task, name):
    return pd.read_csv(D / task / f"{name}.csv")


def miss(df, top=8):
    m = df.isna().mean().sort_values(ascending=False)
    m = m[m > 0]
    return f"{len(m)} cols with NaN; mean NaN frac {df.isna().mean().mean():.4f}; top: " + ", ".join(f"{k}={v:.3f}" for k, v in m.head(top).items())


# ---------------- Task 1
print("=" * 30, "TASK 1")
tr, te = load("01_binary_classification", "train"), load("01_binary_classification", "test")
print("train", tr.shape, "test", te.shape, "| dup player_id train:", tr.player_id.duplicated().sum())
print("target:", tr.is_cheater.value_counts(dropna=False).to_dict())
print("NaN:", miss(tr.drop(columns=["player_id", "is_cheater"])))
lab = tr.dropna(subset=["is_cheater"])
f = [c for c in lab.columns if c not in ("player_id", "is_cheater")]
corr = lab[f].corrwith(lab.is_cheater).abs().sort_values(ascending=False)
print("top |corr| with target:\n", corr.head(12).round(3).to_string())
print("is the target-missing group different? NaN frac of features by target-missing:",
      round(tr[tr.is_cheater.isna()][f].isna().mean().mean(), 4), "vs", round(lab[f].isna().mean().mean(), 4))
print("test NaN:", miss(te.drop(columns=["player_id"])))

# ---------------- Task 2
print("=" * 30, "TASK 2")
tr, te = load("02_multiclass_classification", "train"), load("02_multiclass_classification", "test")
print("train", tr.shape, "test", te.shape)
print("target:", tr.segment.value_counts(dropna=False).sort_index().to_dict())
cats = tr.select_dtypes(exclude=np.number).columns.tolist()
print("categorical:", {c: tr[c].nunique() for c in cats})
print("NaN:", miss(tr.drop(columns=["player_id", "segment"])))
num = [c for c in tr.columns if c not in cats + ["player_id", "segment"]]
print("numeric means by segment (selected):")
print(tr.groupby("segment")[["total_spending_thb", "avg_monthly_spending", "total_playtime_hours", "friend_count",
                              "team_play_percentage", "ranked_participation_rate", "vip_tier"]].mean().round(1).to_string())
for c in cats[:10]:
    ct = pd.crosstab(tr[c], tr.segment, normalize="index").round(2)
    spread = (ct.max(axis=1) - ct.min(axis=1)).max()
    print(f"  cat {c}: levels={tr[c].nunique()} max within-level class spread={spread:.2f}")

# ---------------- Task 3
print("=" * 30, "TASK 3")
tr, te = load("03_regression", "train"), load("03_regression", "test")
print("train", tr.shape, "test", te.shape)
y = tr.spending_30d
print("y: zero frac", round((y == 0).mean(), 4), "| quantiles", y.quantile([.5, .75, .9, .95, .99]).round(0).to_dict(), "| max", y.max(),
      "| share of rows at max:", round((y == y.max()).mean(), 4))
print("NaN:", miss(tr.drop(columns=["player_id", "spending_30d"])))
f = [c for c in tr.columns if c not in ("player_id", "spending_30d")]
print("top |spearman| with y:\n", tr[f].corrwith(y, method="spearman").abs().sort_values(ascending=False).head(10).round(3).to_string())
print("y by segment:", tr.groupby("segment").spending_30d.agg(["mean", "median", lambda s: (s == 0).mean()]).round(2).to_dict("index"))
print("zero-y vs prev_month_spending==0:", pd.crosstab(y == 0, tr.prev_month_spending == 0, normalize="index").round(3).to_dict())

# ---------------- Task 5
print("=" * 30, "TASK 5")
te5 = load("05_anomaly_detection", "test")
print("test", te5.shape, "| dup ids:", te5.player_id.duplicated().sum())
X = te5.drop(columns=["player_id"])
print("NaN:", miss(X))
flags = ["password_changed", "vpn_usage", "suspicious_login_time", "mass_item_sale"]
print("flag means:", X[flags].mean().round(3).to_dict())
print("flag co-occurrence (sum of flags):", X[flags].sum(axis=1).value_counts().sort_index().to_dict())
print("describe (selected):")
print(X[["login_count_1", "device_count_1", "skill_rating_1", "kda_mean_1", "accuracy_mean_1", "reaction_time_mean_1",
         "purchase_amount_mean_1", "currency_transfers_1", "account_age_days", "behavior_stability"]].describe().round(2).T.to_string())
