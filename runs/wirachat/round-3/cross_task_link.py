"""Round 3 - are our Task 4 predictions (ConvNeXt, test.csv order) statistically linked to any discrete column of the other tasks' test files at the same row index?
Mutual information against a shuffled null. Uses only provided test features and our own predictions (no labels)."""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import mutual_info_score
warnings.filterwarnings("ignore")
D = Path(__file__).resolve().parents[3] / "cpe342-kaggle-main"; R2 = Path(__file__).resolve().parents[1] / "round-2"
pred = pd.read_csv(R2 / "submission_v10_t4_convnext.csv").task4.values
rng = np.random.default_rng(0)
rows = []
for t, f in [(1, "01_binary_classification"), (2, "02_multiclass_classification"), (3, "03_regression"), (5, "05_anomaly_detection")]:
    df = pd.read_csv(D / f / "test.csv")
    for c in df.columns:
        if c == "player_id": continue
        v = df[c]
        if pd.api.types.is_numeric_dtype(v):
            if v.nunique() > 12: continue
        x = v.fillna(-999).astype(str).fillna('NA').values
        mi = mutual_info_score(pred, x); null = np.array([mutual_info_score(pred, rng.permutation(x)) for _ in range(60)])
        rows.append((f"T{t}.{c}", v.nunique(), mi, null.mean(), (mi - null.mean()) / (null.std() + 1e-12)))
r = pd.DataFrame(rows, columns=["column", "levels", "MI", "null_mean", "z"]).sort_values("z", ascending=False)
print(f"{len(r)} discrete columns tested against the Task 4 predictions at the same row index")
print(r.head(8).to_string(index=False))
print("max z:", round(float(r.z.max()), 2), "| number of columns with z > 4:", int((r.z > 4).sum()), "(expected by chance about", round(len(r) * 3e-5, 3), ")")
