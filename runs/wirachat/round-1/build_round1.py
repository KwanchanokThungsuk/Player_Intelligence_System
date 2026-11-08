"""Round 1 - assemble the local submission from the per-task results (does NOT upload anything)."""
import sys, json
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from r1_common import load, macro_fbeta, best_threshold, nmae_score, OUT, ROOT, DATA, SEED
from sklearn.metrics import f1_score

summary = {}
# ---------------- Task 1: average raw + fe LightGBM probabilities, threshold tuned on macro F2 (OOF)
tr1 = load(1, "train").dropna(subset=["is_cheater"]); y1 = tr1.is_cheater.astype(int).values
oof1 = (np.load(OUT / "task1_oof_raw.npy") + np.load(OUT / "task1_oof_fe.npy")) / 2
te1p = (np.load(OUT / "task1_test_prob_raw.npy") + np.load(OUT / "task1_test_prob_fe.npy")) / 2
t1, f1, _, _ = best_threshold(y1, oof1, 2, macro=True)
task1 = (te1p >= t1).astype(int)
summary["task1"] = dict(oof_macro_f2=f1, threshold=t1, test_positive_rate=float(task1.mean()))

# ---------------- Task 2: class-weighted argmax (weights tuned on OOF macro F1)
tr2 = load(2, "train"); y2 = tr2.segment.astype(int).values
m2 = json.load(open(OUT / "task2_metrics_clean.json")); w2 = np.array(m2["weights"])
task2 = (np.load(OUT / "task2_test_prob_clean.npy") * w2).argmax(1)
summary["task2"] = dict(oof_macro_f1=float(f1_score(y2, (np.load(OUT / "task2_oof_clean.npy") * w2).argmax(1), average="macro")), weights=w2.tolist(),
                        test_class_share=np.bincount(task2, minlength=4).round(0).tolist())

# ---------------- Task 3: blend of L1 LightGBM configs
y3 = load(3, "train").spending_30d.values
names = ["l1_raw", "A_lr08_l63", "B_lr08_l127", "C_lr08_l31", "D_lr05_l255", "E_lr08_l15"]
oofs = {n: np.load(OUT / f"task3_oof_{n}.npy") for n in names}; tes = {n: np.load(OUT / f"task3_test_{n}.npy") for n in names}
for n in names: print(f"task3 {n}: NMAE {nmae_score(y3, oofs[n]):.4f}")
mean_oof = np.mean([oofs[n] for n in names], axis=0); med_oof = np.median([oofs[n] for n in names], axis=0)
print(f"task3 blend mean NMAE {nmae_score(y3, mean_oof):.4f} | median-blend {nmae_score(y3, med_oof):.4f}")
use_median = nmae_score(y3, med_oof) > nmae_score(y3, mean_oof)
task3 = (np.median if use_median else np.mean)([tes[n] for n in names], axis=0)
summary["task3"] = dict(oof_nmae=max(nmae_score(y3, mean_oof), nmae_score(y3, med_oof)), blend="median" if use_median else "mean")

# ---------------- Task 4: keep the baseline's ResNet18 predictions (already in the repo)
base4 = pd.read_csv(ROOT / "predictions" / "task4_test_predictions.csv")
te4 = load(4, "test"); assert base4.id.tolist() == te4.id.astype(str).tolist()
task4 = base4.task4.values

# ---------------- Task 5: flag rows whose period-4 change score is far beyond the normal range (natural gap)
S = pd.read_pickle(OUT / "task5_S.pkl")["S"]
te5 = load(5, "test"); assert len(S) == len(te5)
task5 = (S > 130).fillna(False).astype(int).values
summary["task5"] = dict(threshold_S=130, flagged=int(task5.sum()), flagged_rate=float(task5.mean()))

# ---------------- assemble using the sample submission ids (test rows are positionally aligned across tasks)
sample = pd.read_csv(DATA / "sample_submission.csv")
n = len(sample); assert all(len(v) == n for v in (task1, task2, task3, task4, task5))
sub = pd.DataFrame({"id": sample.id, "task1": task1, "task2": task2, "task3": task3, "task4": task4, "task5": task5})
valid4 = set(load(4, "train")["class"].astype(str))
assert sub.shape == (25889, 6) and sub.id.is_unique and sub.isna().sum().sum() == 0
assert set(sub.task1) <= {0, 1} and set(sub.task5) <= {0, 1} and set(sub.task2) <= {0, 1, 2, 3} and set(sub.task4) <= valid4
assert np.isfinite(sub.task3).all() and (sub.task3 >= 0).all()
sub.to_csv(OUT / "submission.csv", index=False)
json.dump(summary, open(OUT / "round1_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
print("wrote", OUT / "submission.csv", sub.shape)
