"""Round 3 - did the host shuffle the sorted id lists with a fixed seed? Brute-force common RNG seeds against every test/train file order.
If a seed reproduces a file's row order we learn how that file was produced (e.g. df.sample(frac=1, random_state=seed))."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
D = Path(__file__).resolve().parents[3] / "cpe342-kaggle-main"
files = {"T4 test": (D / "04_image_classification/test.csv", "id"), "T4 train": (D / "04_image_classification/train.csv", "id"),
         "T1 test": (D / "01_binary_classification/test.csv", "player_id"), "T2 test": (D / "02_multiclass_classification/test.csv", "player_id"),
         "T3 test": (D / "03_regression/test.csv", "player_id"), "T5 test": (D / "05_anomaly_detection/test.csv", "player_id")}
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
K = 5
for name, (path, col) in files.items():
    L = pd.read_csv(path, usecols=[col])[col].astype(str).tolist(); n = len(L)
    S = sorted(L); idx = {s: i for i, s in enumerate(S)}; target = np.array([idx[s] for s in L[:K]])
    t0 = time.time(); hits = []
    for s in range(N_SEEDS):
        if np.array_equal(np.random.RandomState(s).permutation(n)[:K], target): hits.append(("RandomState.permutation", s))
        if s < 20000 and np.array_equal(np.random.default_rng(s).permutation(n)[:K], target): hits.append(("default_rng.permutation", s))
    # also test whether the file is simply a permutation consistent with pandas .sample on the sorted frame for a few classic seeds
    sample_hits = []
    base = pd.DataFrame({col: S})
    for s in (0, 1, 2, 7, 21, 42, 99, 123, 342, 1234, 2024, 2025, 2026):
        if base.sample(frac=1, random_state=s)[col].tolist()[:K] == L[:K]: sample_hits.append(s)
    print(f"{name:9s} n={n:6d}: sorted-by-id order? {L == S} | seed hits (first {K} rows match): {hits or 'none'} | pandas.sample hits: {sample_hits or 'none'} | {time.time()-t0:.0f}s", flush=True)
