"""Analyse the style features from t4_diag_style.py: class predictability from style alone, train-vs-test shift."""
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
R3 = Path(__file__).resolve().parent; D = R3.parents[2] / "cpe342-kaggle-main" / "04_image_classification"
F = np.load(R3 / "data3" / "t4_style_feats.npy"); tr = pd.read_csv(D / "train.csv"); n = len(tr); Ftr, Fte = F[:n], F[n:]
classes = sorted(tr["class"].unique()); y = tr["class"].map({c: i for i, c in enumerate(classes)}).values
names = ["black_frac", "black_border", "cornTL", "cornTR", "cornBL", "cornBR", "gray_mean", "gray_std", "sat_mean", "sat_std", "val_mean", "noise", "colorful", "clip_hi", "clip_lo", "R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std"] + [f"hue{i}" for i in range(12)]
# (a) class from style alone (random 5-fold; train images are from clusters but style features carry no content)
m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=12)
p = cross_val_predict(m, Ftr, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
print(f"(a) class from STYLE ONLY: macro F1 {f1_score(y, p.argmax(1), average='macro'):.3f} acc {accuracy_score(y, p.argmax(1)):.3f} (majority-class acc {np.bincount(y).max()/len(y):.3f}, chance macroF1 0.2)")
# (b) train vs test shift
Xa = np.vstack([Ftr, Fte]); ya = np.r_[np.zeros(n), np.ones(len(Fte))]
pa = cross_val_predict(lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1, n_jobs=12), Xa, ya, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
print(f"(b) train-vs-test from STYLE: AUC {roc_auc_score(ya, pa):.3f}")
# per-feature distribution: train per class vs test (medians)
df = pd.DataFrame(Ftr, columns=names); df["cls"] = [classes[i] for i in y]
tab = df.groupby("cls")[names[:15]].median().round(3); tab.loc["TEST"] = pd.DataFrame(Fte, columns=names)[names[:15]].median().round(3)
pd.set_option("display.width", 250); print(tab.T.to_string())
# train/test per-feature KS
from scipy.stats import ks_2samp
ks = sorted([(ks_2samp(Ftr[:, i], Fte[:, i]).statistic, names[i]) for i in range(len(names))], reverse=True)[:8]; print("largest train-vs-test KS:", [(b, round(a, 3)) for a, b in ks])
# which features matter for class
m.fit(Ftr, y); imp = sorted(zip(m.feature_importances_, names), reverse=True)[:8]; print("style features most used for class:", imp)
