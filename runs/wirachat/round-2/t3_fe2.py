"""Round 2 - Task 3 v2: exact identity hist = total_transactions * avg_transaction_value used first; segment recovered when missing."""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
warnings.filterwarnings("ignore")
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load, nmae_score, SEED

OUT = Path(__file__).resolve().parent
tr, te = load(3, "train"), load(3, "test")
y = tr.spending_30d.values
base_cols = [c for c in tr.columns if c not in ("player_id", "spending_30d")]
# --- is segment recoverable from the other columns?
known = tr.segment.notna().values
Xs = tr[[c for c in base_cols if c != "segment"]]
seg_oof = np.full((len(tr), 4), np.nan)
sp = dict(objective="multiclass", num_class=4, learning_rate=0.1, num_leaves=31, min_child_samples=30, verbose=-1, seed=SEED, n_jobs=12)
idx = np.where(known)[0]
for a, b in KFold(5, shuffle=True, random_state=SEED).split(idx):
    m = lgb.train(sp, lgb.Dataset(Xs.iloc[idx[a]], tr.segment.values[idx[a]].astype(int)), 600)
    seg_oof[idx[b]] = m.predict(Xs.iloc[idx[b]])
acc = (seg_oof[idx].argmax(1) == tr.segment.values[idx].astype(int)).mean()
print(f"segment recoverable from the other columns: CV accuracy {acc:.4f}")
mfull = lgb.train(sp, lgb.Dataset(Xs.iloc[idx], tr.segment.values[idx].astype(int)), 600)
seg_tr = tr.segment.copy(); seg_hat_tr = pd.Series(mfull.predict(Xs).argmax(1), index=tr.index)
seg_te_hat = pd.Series(mfull.predict(te[[c for c in base_cols if c != "segment"]]).argmax(1), index=te.index)
seg_tr[~known] = seg_hat_tr[~known]                       # used only where segment is missing


def fe(df, seg_filled):
    X = df[base_cols].copy()
    tx, av, h, p = df.total_transactions, df.avg_transaction_value, df.historical_spending, df.prev_month_spending
    k = 1.05
    h_f = h.copy()
    h_f = h_f.fillna(tx * av)                                      # exact identity first
    h_f = h_f.fillna(p * np.exp(k))                                # noisy fallback
    tx_f = tx.fillna(h_f / av.replace(0, np.nan)); av_f = av.fillna(h_f / tx.replace(0, np.nan))
    p_f = p.fillna(h_f * np.exp(-k))
    X["hist_f"], X["prev_f"], X["tx_f"], X["avg_f"] = h_f, p_f, tx_f, av_f
    X["log_hist_f"], X["log_prev_f"] = np.log1p(h_f), np.log1p(p_f)
    X["prev_over_hist"] = p / (h + 1)
    X["spend_per_age"] = h_f / (df.account_age_days + 1)
    X["seg_f"] = seg_filled.values
    return X


strat = (tr.segment.fillna(-1).astype(int).astype(str) + (y > 0).astype(int).astype(str)).values
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr, strat))
p = dict(objective="l1", metric="l1", learning_rate=0.08, num_leaves=63, min_child_samples=30, feature_fraction=0.8, bagging_fraction=0.8,
         bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=SEED, n_jobs=12)
Xa, Xt = fe(tr, seg_tr), fe(te, te.segment.fillna(seg_te_hat))
oof = np.zeros(len(tr)); pte = np.zeros(len(te))
for a, b in folds:
    m = lgb.train(p, lgb.Dataset(Xa.iloc[a], y[a]), 8000, valid_sets=[lgb.Dataset(Xa.iloc[b], y[b])], callbacks=[lgb.early_stopping(250, verbose=False)])
    oof[b] = np.clip(m.predict(Xa.iloc[b], num_iteration=m.best_iteration), 0, 300000); pte += np.clip(m.predict(Xt, num_iteration=m.best_iteration), 0, 300000) / 5
miss = (tr.historical_spending.isna() | tr.prev_month_spending.isna()).values
print(f"L1 LightGBM + exact-identity features: NMAE {nmae_score(y, oof):.4f} [raw 0.7833, v1 FE 0.7856] | missing hist/prev rows {nmae_score(y[miss], oof[miss]):.4f} | complete rows {nmae_score(y[~miss], oof[~miss]):.4f}")
sm = tr.segment.isna().values
print(f"   rows with missing segment: {nmae_score(y[sm], oof[sm]):.4f} (before: round-1 blend overall error share for them was 6.2%)")
np.save(OUT / "t3_fe2_oof.npy", oof); np.save(OUT / "t3_fe2_test.npy", pte)
json.dump(dict(nmae=nmae_score(y, oof), seg_acc=float(acc)), open(OUT / "t3_fe2_summary.json", "w"))
