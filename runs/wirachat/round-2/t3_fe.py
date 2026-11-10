"""Round 2 - Task 3: feature engineering for rows with missing / extreme spending history. Same folds as round 1."""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load, nmae_score, SEED

OUT = Path(__file__).resolve().parent
tr, te = load(3, "train"), load(3, "test")
y = tr.spending_30d.values
# --- relationships (rows where everything is present)
c = tr.dropna(subset=["historical_spending", "prev_month_spending", "total_transactions", "avg_transaction_value"])
c = c[(c.historical_spending > 0) & (c.prev_month_spending > 0) & (c.total_transactions > 0) & (c.avg_transaction_value > 0)]
lh, lp, lt = np.log(c.historical_spending), np.log(c.prev_month_spending), np.log(c.total_transactions * c.avg_transaction_value)
print(f"rows {len(c)} | corr(log hist, log prev)={np.corrcoef(lh, lp)[0,1]:.4f} | corr(log hist, log tx*avg)={np.corrcoef(lh, lt)[0,1]:.4f} | corr(log prev, log tx*avg)={np.corrcoef(lp, lt)[0,1]:.4f}")
print("log(hist/prev) quantiles 5/50/95:", np.round(np.quantile(lh - lp, [.05, .5, .95]), 3).tolist(), "| log(hist/(tx*avg)) quantiles 5/50/95:", np.round(np.quantile(lh - lt, [.05, .5, .95]), 3).tolist())
both = tr.historical_spending.isna() & tr.prev_month_spending.isna()
print(f"rows with BOTH hist and prev missing: {int(both.sum())} ({both.mean():.2%}); either missing: {int((tr.historical_spending.isna() | tr.prev_month_spending.isna()).sum())}")
k_hp = float(np.median(lh - lp))


def fe(df):
    X = df.drop(columns=["player_id", "spending_30d"], errors="ignore").copy()
    tx = df.total_transactions * df.avg_transaction_value
    h, p = df.historical_spending, df.prev_month_spending
    X["tx_total"] = tx
    X["hist_f"] = h.fillna(p * np.exp(k_hp)).fillna(tx)           # rebuild a missing history from prev, then from transactions
    X["prev_f"] = p.fillna(h * np.exp(-k_hp))
    X["log_hist_f"] = np.log1p(X.hist_f); X["log_prev_f"] = np.log1p(X.prev_f); X["log_tx"] = np.log1p(tx)
    X["hist_over_tx"] = X.hist_f / (tx + 1)
    X["spend_per_age"] = X.hist_f / (df.account_age_days + 1)
    X["n_missing_main"] = h.isna().astype(int) + p.isna().astype(int) + df.total_transactions.isna().astype(int) + df.avg_transaction_value.isna().astype(int)
    return X


strat = (tr.segment.fillna(-1).astype(int).astype(str) + (y > 0).astype(int).astype(str)).values
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr, strat))
p = dict(objective="l1", metric="l1", learning_rate=0.08, num_leaves=63, min_child_samples=30, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=SEED, n_jobs=12)
res = {}
for name, Xa, Xt in [("raw features", tr.drop(columns=["player_id", "spending_30d"]), te.drop(columns=["player_id"])), ("+ engineered", fe(tr), fe(te))]:
    oof = np.zeros(len(tr)); pte = np.zeros(len(te))
    for a, b in folds:
        m = lgb.train(p, lgb.Dataset(Xa.iloc[a], y[a]), 8000, valid_sets=[lgb.Dataset(Xa.iloc[b], y[b])], callbacks=[lgb.early_stopping(250, verbose=False)])
        oof[b] = np.clip(m.predict(Xa.iloc[b], num_iteration=m.best_iteration), 0, 300000); pte += np.clip(m.predict(Xt, num_iteration=m.best_iteration), 0, 300000) / 5
    res[name] = nmae_score(y, oof)
    miss = (tr.historical_spending.isna() | tr.prev_month_spending.isna()).values
    print(f"{name:14s}: NMAE {res[name]:.4f} | NMAE on rows with missing hist/prev: {nmae_score(y[miss], oof[miss]):.4f} | on complete rows: {nmae_score(y[~miss], oof[~miss]):.4f}", flush=True)
    np.save(OUT / f"t3_fe_oof_{'eng' if 'eng' in name else 'raw'}.npy", oof); np.save(OUT / f"t3_fe_test_{'eng' if 'eng' in name else 'raw'}.npy", pte)
json.dump(res, open(OUT / "t3_fe_summary.json", "w"), indent=1)
