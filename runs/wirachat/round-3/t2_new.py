"""Round 3 - Task 2 on the NEW dataset (2026-11-12): same method as round 1 (LightGBM multiclass, 5-fold, noise columns dropped, class weights tuned on OOF).
Trained from scratch on the new train.csv only. Outputs go to round-3/data2/."""
import sys, os, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
warnings.filterwarnings("ignore")
os.environ.setdefault("CPE342_DATA", "C:/Users/Wirachat/Desktop/CPE342-DATA-v2/kaggle/cpe342-kaggle-main")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "round-1"))
from r1_common import load, SEED
OUT = Path(__file__).resolve().parent / "data2"; OUT.mkdir(exist_ok=True)
tr, te = load(2, "train"), load(2, "test"); y = tr.segment.astype(int).values
cats = [c for c in tr.columns if c not in ("player_id", "segment") and not pd.api.types.is_numeric_dtype(tr[c])]
feat = [c for c in tr.columns if c not in ("player_id", "segment") and not c.startswith("random_metric")]
def prep(d):
    X = d[feat].copy()
    for c in cats:
        if c in X: X[c] = pd.Categorical(X[c], categories=sorted(tr[c].dropna().unique()))
    return X
X, Xte = prep(tr), prep(te)
p = dict(objective="multiclass", num_class=4, learning_rate=0.03, num_leaves=31, min_child_samples=40, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=SEED, n_jobs=12)
oof = np.zeros((len(X), 4)); pte = np.zeros((len(Xte), 4))
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)):
    m = lgb.train(p, lgb.Dataset(X.iloc[a], y[a]), 3000, valid_sets=[lgb.Dataset(X.iloc[b], y[b])], callbacks=[lgb.early_stopping(150, verbose=False)])
    oof[b] = m.predict(X.iloc[b], num_iteration=m.best_iteration); pte += m.predict(Xte, num_iteration=m.best_iteration) / 5
    print(f"fold {k}: iters {m.best_iteration} macroF1 {f1_score(y[b], oof[b].argmax(1), average='macro'):.4f}", flush=True)
w = np.ones(4); grid = np.linspace(0.6, 1.8, 25)
for _ in range(3):
    for c in range(4):
        best = (f1_score(y, (oof * w).argmax(1), average="macro"), w[c])
        for g in grid:
            w2 = w.copy(); w2[c] = g; s = f1_score(y, (oof * w2).argmax(1), average="macro")
            if s > best[0]: best = (s, g)
        w[c] = best[1]
print(f"NEW Task 2: OOF macro F1 argmax {f1_score(y, oof.argmax(1), average='macro'):.4f} | weighted {f1_score(y, (oof * w).argmax(1), average='macro'):.4f} w={np.round(w, 2).tolist()}")
pred = (pte * w).argmax(1); np.save(OUT / "task2_test_pred.npy", pred); np.save(OUT / "task2_test_prob.npy", pte)
print("test class shares:", np.round(np.bincount(pred, minlength=4) / len(pred), 3).tolist(), "| train shares:", np.round(np.bincount(y) / len(y), 3).tolist())
