"""Round 2 - Task 1: XGBoost and CatBoost on the same 5 folds as round 1; blend check with the round-1 LightGBM OOF."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, fbeta_score
R1 = Path(__file__).resolve().parents[1] / "round-1"
sys.path.insert(0, str(R1))
from r1_common import load, macro_fbeta, best_threshold, SEED, OUT as R1_OUT

OUT = Path(__file__).resolve().parent
tr, te = load(1, "train"), load(1, "test")
lab = tr.dropna(subset=["is_cheater"]).reset_index(drop=True)
y = lab.is_cheater.astype(int).values
feat = [c for c in lab.columns if c not in ("player_id", "is_cheater")]
X, Xte = lab[feat], te[feat]
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))  # identical to round 1
res = {}
for name in ("xgb", "cat"):
    oof = np.zeros(len(X)); pte = np.zeros(len(Xte))
    for k, (a, b) in enumerate(folds):
        if name == "xgb":
            m = xgb.XGBClassifier(n_estimators=4000, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                                  reg_lambda=5.0, tree_method="hist", early_stopping_rounds=150, eval_metric="logloss", n_jobs=16, random_state=SEED)
            m.fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], verbose=False)
            oof[b] = m.predict_proba(X.iloc[b])[:, 1]; pte += m.predict_proba(Xte)[:, 1] / 5
        else:
            m = CatBoostClassifier(iterations=4000, learning_rate=0.05, depth=6, l2_leaf_reg=5, loss_function="Logloss", od_type="Iter", od_wait=150,
                                   thread_count=16, random_seed=SEED, verbose=False)
            m.fit(X.iloc[a], y[a], eval_set=(X.iloc[b], y[b]), use_best_model=True)
            oof[b] = m.predict_proba(X.iloc[b])[:, 1]; pte += m.predict_proba(Xte)[:, 1] / 5
        print(f"{name} fold {k}: auc={roc_auc_score(y[b], oof[b]):.4f}", flush=True)
    np.save(OUT / f"task1_oof_{name}.npy", oof); np.save(OUT / f"task1_test_prob_{name}.npy", pte)
    res[name] = (oof, pte)

lgb_oof = (np.load(R1_OUT / "task1_oof_raw.npy") + np.load(R1_OUT / "task1_oof_fe.npy")) / 2
lgb_te = (np.load(R1_OUT / "task1_test_prob_raw.npy") + np.load(R1_OUT / "task1_test_prob_fe.npy")) / 2
cands = {"lgb(round1)": lgb_oof, "xgb": res["xgb"][0], "cat": res["cat"][0],
         "lgb+xgb+cat": (lgb_oof + res["xgb"][0] + res["cat"][0]) / 3}
summary = {}
for n, o in cands.items():
    tm, fm, *_ = best_threshold(y, o, 2, macro=True); tb, fb, *_ = best_threshold(y, o, 2, macro=False)
    summary[n] = dict(auc=roc_auc_score(y, o), macro_f2=fm, thr_macro=tm, binary_f2=fb, thr_binary=tb)
    print(f"{n:12s} AUC={summary[n]['auc']:.4f} | best macroF2={fm:.4f} @thr {tm:.3f} | best binaryF2={fb:.4f} @thr {tb:.3f}")
np.save(OUT / "task1_test_prob_blend3.npy", (lgb_te + res["xgb"][1] + res["cat"][1]) / 3)
json.dump(summary, open(OUT / "task1_ens_summary.json", "w"), indent=1)
