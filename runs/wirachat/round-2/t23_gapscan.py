"""Round 2 - Tasks 2 and 3: gap / sub-population scan on the features, and label association (read-only)."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load


def scan(df, label, name, drop):
    cols = [c for c in df.columns if c not in drop and df[c].dtype != object and df[c].nunique() > 25]
    print(f"\n=== {name}: {len(cols)} continuous columns scanned")
    rows = []
    for c in cols:
        v = df[c].dropna().values
        h, e = np.histogram(v, bins=100)
        nz = np.where(h > 0)[0]
        gaps = [(i1 - i0 - 1, (e[i0 + 1] + e[i1]) / 2) for i0, i1 in zip(nz[:-1], nz[1:])]
        if not gaps: continue
        n_empty, cut = max(gaps)
        small = min((v < cut).sum(), (v > cut).sum())
        if n_empty >= 3 and small >= 30:
            low_small = (v < cut).sum() < (v > cut).sum()
            grp = (df[c] < cut) if low_small else (df[c] > cut)
            lab = df.loc[grp.fillna(False), label].value_counts(normalize=True).round(2).to_dict()
            rows.append((c, n_empty, round(cut, 3), int(small), "low" if low_small else "high", lab))
    r = pd.DataFrame(rows, columns=["column", "empty_bins", "cut", "n_small", "side", f"{label} mix in small side"]).sort_values("empty_bins", ascending=False)
    print(r.head(12).to_string(index=False) if len(r) else "no gaps found")
    return r


t2 = load(2, "train"); t3 = load(3, "train")
scan(t2, "segment", "Task 2 train", ["player_id", "segment"])
scan(t3, "spending_30d", "Task 3 train", ["player_id", "spending_30d"]) if False else None
# Task 3: label is continuous -> report mean/median of y inside the small side instead
cols = [c for c in t3.columns if c not in ("player_id", "spending_30d") and t3[c].nunique() > 25]
print(f"\n=== Task 3 train: {len(cols)} continuous columns scanned")
rows = []
for c in cols:
    v = t3[c].dropna().values
    h, e = np.histogram(v, bins=100)
    nz = np.where(h > 0)[0]
    gaps = [(i1 - i0 - 1, (e[i0 + 1] + e[i1]) / 2) for i0, i1 in zip(nz[:-1], nz[1:])]
    if not gaps: continue
    n_empty, cut = max(gaps)
    small = min((v < cut).sum(), (v > cut).sum())
    if n_empty >= 3 and small >= 30:
        low_small = (v < cut).sum() < (v > cut).sum()
        grp = ((t3[c] < cut) if low_small else (t3[c] > cut)).fillna(False)
        rows.append((c, n_empty, round(cut, 3), int(small), "low" if low_small else "high", round(t3.loc[grp, "spending_30d"].median(), 1), round(t3.loc[~grp, "spending_30d"].median(), 1)))
r = pd.DataFrame(rows, columns=["column", "empty_bins", "cut", "n_small", "side", "median_y_small", "median_y_rest"]).sort_values("empty_bins", ascending=False)
print(r.head(12).to_string(index=False) if len(r) else "no gaps found")

# distribution shapes of the key columns (is anything a mixture / uniform / truncated?)
print("\nTask 3 key columns: quantiles 0/1/25/50/75/99/100 and zero share")
for c in ["historical_spending", "prev_month_spending", "event_participation_rate", "avg_transaction_value", "total_transactions", "vip_status", "segment"]:
    q = t3[c].quantile([0, .01, .25, .5, .75, .99, 1]).round(1).tolist()
    print(f"  {c}: {q} | zero share {(t3[c] == 0).mean():.3f} | NaN {t3[c].isna().mean():.3f}")
