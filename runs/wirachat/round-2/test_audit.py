"""Round 2 - test-side audit for Tasks 1-3: shift train vs test, missingness, range, gap groups in TEST, duplicates. Read-only (no model training)."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
warnings.filterwarnings("ignore")
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load


def gaps(v, bins=100):
    h, e = np.histogram(v, bins=bins); nz = np.where(h > 0)[0]
    g = [(i1 - i0 - 1, (e[i0 + 1] + e[i1]) / 2) for i0, i1 in zip(nz[:-1], nz[1:])]
    if not g: return 0, None, 0
    n, cut = max(g); return n, cut, min((v < cut).sum(), (v > cut).sum())


for t, tgt in [(1, "is_cheater"), (2, "segment"), (3, "spending_30d")]:
    tr, te = load(t, "train"), load(t, "test")
    cols = [c for c in te.columns if c != "player_id" and c in tr.columns]
    print(f"\n==================== TASK {t}: train {tr.shape} vs test {te.shape}")
    rows = []
    for c in cols:
        a, b = tr[c], te[c]
        rec = dict(col=c, nan_tr=a.isna().mean(), nan_te=b.isna().mean())
        if pd.api.types.is_numeric_dtype(a):
            x, y = a.dropna().values, b.dropna().values
            if len(x) > 20000: x = np.random.default_rng(1).choice(x, 20000, replace=False)
            if len(y) > 20000: y = np.random.default_rng(2).choice(y, 20000, replace=False)
            rec["ks"] = ks_2samp(x, y).statistic
            rec["out_of_range"] = float(((te[c] < a.min()) | (te[c] > a.max())).mean())
            rec["mean_tr"], rec["mean_te"] = a.mean(), b.mean()
            n_tr, cut_tr, s_tr = gaps(a.dropna().values) if a.nunique() > 25 else (0, None, 0)
            n_te, cut_te, s_te = gaps(b.dropna().values) if b.nunique() > 25 else (0, None, 0)
            rec["gap_tr"], rec["gap_te"], rec["small_te"] = n_tr, n_te, s_te
        rows.append(rec)
    R = pd.DataFrame(rows)
    print(f"missing-rate: train mean {R.nan_tr.mean():.4f} vs test mean {R.nan_te.mean():.4f}; per-column |diff| max {np.abs(R.nan_tr - R.nan_te).max():.4f}")
    print("largest KS shifts:", ", ".join(f"{r.col}={r.ks:.3f}" for r in R.dropna(subset=['ks']).sort_values('ks', ascending=False).head(6).itertuples()))
    print("columns with out-of-train-range test values >0.1%:", [(r.col, round(r.out_of_range, 4)) for r in R.dropna(subset=['out_of_range']).itertuples() if r.out_of_range > 0.001] or "none")
    g = R.dropna(subset=["gap_te"])
    print("gap columns in TEST (>=4 empty bins) vs TRAIN:", [(r.col, int(r.gap_tr), int(r.gap_te), int(r.small_te)) for r in g.itertuples() if r.gap_te >= 4 and r.small_te >= 30] or "none")
    # exact duplicate feature rows between train and test (round to 4 decimals, non-NaN cells only on a few stable columns)
    key = [c for c in cols if pd.api.types.is_numeric_dtype(tr[c]) and tr[c].nunique() > 1000][:6]
    if key:
        a = tr[key].round(4).dropna().apply(tuple, axis=1); b = te[key].round(4).dropna().apply(tuple, axis=1)
        print(f"exact duplicates on {len(key)} continuous columns between train and test: {len(set(a) & set(b))}")
    if t == 2:
        cat = [c for c in te.columns if not pd.api.types.is_numeric_dtype(te[c]) and c != "player_id"]
        for c in cat:
            d = pd.concat([tr[c].value_counts(normalize=True).rename("tr"), te[c].value_counts(normalize=True).rename("te")], axis=1).fillna(0)
            if (d.tr - d.te).abs().max() > 0.02: print(f"   categorical {c} shifts: {d.round(3).to_dict('index')}")

print("\n==================== missing-rate differences (train vs test), top 8 per task")
for t in (1, 2, 3):
    tr, te = load(t, "train"), load(t, "test")
    cols = [c for c in te.columns if c != "player_id" and c in tr.columns]
    d = pd.DataFrame({"train": tr[cols].isna().mean(), "test": te[cols].isna().mean()}); d["diff"] = d.train - d.test
    print(f"Task {t}:", ", ".join(f"{i}: {r.train:.3f} vs {r.test:.3f}" for i, r in d.reindex(d["diff"].abs().sort_values(ascending=False).index).head(8).iterrows()))
