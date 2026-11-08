import sys
import numpy as np, pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from r1_common import load
for t, drop in [(2, ["player_id", "segment"]), (3, ["player_id", "spending_30d"]), (1, ["player_id", "is_cheater"])]:
    df = load(t, "train").drop(columns=drop)
    num = df.select_dtypes(include=np.number)
    num = num[[c for c in num.columns if not c.startswith("random_metric")]]
    c = num.corr().abs()
    pairs = [(a, b, c.loc[a, b]) for i, a in enumerate(c.columns) for b in c.columns[i + 1:] if c.loc[a, b] > 0.8]
    print(f"Task {t}: {num.shape[1]} numeric features; pairs with |corr|>0.8:", sorted(pairs, key=lambda x: -x[2])[:12] or "none")
    # best single-feature linear predictability (R^2) of each feature from the others combined (linear, complete cases subset)
    cc = num.dropna(thresh=int(num.shape[1] * .9)).fillna(num.median())
    out = {}
    for col in cc.columns:
        A = cc.drop(columns=col).values; yv = cc[col].values
        A1 = np.c_[A, np.ones(len(A))]
        beta, *_ = np.linalg.lstsq(A1[:20000], yv[:20000], rcond=None)
        r2 = 1 - ((yv[20000:40000] - A1[20000:40000] @ beta) ** 2).sum() / ((yv[20000:40000] - yv[20000:40000].mean()) ** 2).sum()
        out[col] = r2
    top = sorted(out.items(), key=lambda x: -x[1])[:6]
    print("   most linearly-recoverable features (R2 from all other features):", [(k, round(v, 3)) for k, v in top])
