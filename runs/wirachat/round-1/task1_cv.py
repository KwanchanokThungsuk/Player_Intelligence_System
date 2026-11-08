"""Round 1 - Task 1: LightGBM 5-fold CV, threshold tuned for MACRO F2 (the Kaggle metric)."""
import sys, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, fbeta_score
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from r1_common import load, macro_fbeta, best_threshold, OUT, SEED

FE = "--fe" in sys.argv
tr, te = load(1, "train"), load(1, "test")
lab = tr.dropna(subset=["is_cheater"]).reset_index(drop=True)
y = lab.is_cheater.astype(int).values
feat = [c for c in lab.columns if c not in ("player_id", "is_cheater")]


def fe(df):
    X = df[feat].copy()
    if FE:
        X["n_missing"] = df[feat].isna().sum(axis=1)
        X["reports_per_age"] = df.reports_received / (df.account_age_days + 1)
        X["kd_x_hs"] = df.kill_death_ratio * df.headshot_percentage
        X["aim_total"] = df.accuracy_score * df.aiming_smoothness
        X["react_x_cons"] = df.reaction_time_ms * df.input_consistency_score
    return X


X, Xte = fe(lab), fe(te)
params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=40, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=SEED, n_jobs=16)
oof = np.zeros(len(X)); pte = np.zeros(len(Xte)); iters = []
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
for k, (a, b) in enumerate(skf.split(X, y)):
    m = lgb.train(params, lgb.Dataset(X.iloc[a], y[a]), 3000, valid_sets=[lgb.Dataset(X.iloc[b], y[b])],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    oof[b] = m.predict(X.iloc[b], num_iteration=m.best_iteration)
    pte += m.predict(Xte, num_iteration=m.best_iteration) / 5
    iters.append(m.best_iteration)
    print(f"fold {k}: iters={m.best_iteration} auc={roc_auc_score(y[b], oof[b]):.4f}", flush=True)

auc = roc_auc_score(y, oof)
t_macro, f_macro, _, _ = best_threshold(y, oof, 2, macro=True)
t_bin, f_bin, _, _ = best_threshold(y, oof, 2, macro=False)
print(f"FE={FE} AUC={auc:.4f}")
print(f"threshold tuned on MACRO F2: t={t_macro:.3f} macro F2={f_macro:.4f}; binary F2 there={fbeta_score(y, oof >= t_macro, beta=2):.4f}")
print(f"threshold tuned on BINARY F2: t={t_bin:.3f} binary F2={f_bin:.4f}; macro F2 there={macro_fbeta(y, oof >= t_bin, 2):.4f}  <- baseline-style tuning")
print(f"test positive rate at macro-F2 threshold: {(pte >= t_macro).mean():.3f} | label prevalence {y.mean():.3f}")
tag = "fe" if FE else "raw"
np.save(OUT / f"task1_oof_{tag}.npy", oof); np.save(OUT / f"task1_test_prob_{tag}.npy", pte)
json.dump(dict(fe=FE, auc=auc, macro_f2=f_macro, threshold=t_macro, binary_f2_at_binary_thr=f_bin, iters=iters), open(OUT / f"task1_metrics_{tag}.json", "w"), indent=1)
