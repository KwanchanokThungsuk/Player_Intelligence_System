import sys
import pandas as pd
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from r1_common import load
ids = {}
for t in (1, 2, 3):
    ids[f"t{t}_train"] = set(load(t, "train").player_id); ids[f"t{t}_test"] = set(load(t, "test").player_id)
ids["t5_test"] = set(load(5, "test").player_id)
t4 = load(4, "train"); print("t4 train ids sample:", t4.id.head(3).tolist(), "| t4 test sample:", load(4, "test").id.head(3).tolist())
names = list(ids)
print("sizes:", {k: len(v) for k, v in ids.items()})
for i, a in enumerate(names):
    for b in names[i + 1:]:
        n = len(ids[a] & ids[b])
        if n: print(f"overlap {a} & {b}: {n}")
sub = pd.read_csv("submissions/submission_baseline_v1.csv") if False else None
te3 = load(3, "test"); print("T3 test: segment NaN", int(te3.segment.isna().sum()), "of", len(te3))
te1 = load(1, "test"); te2 = load(2, "test")
print("test order identical T1/T2/T3/T5:", list(te1.player_id) == list(te2.player_id) == list(te3.player_id) == list(load(5, "test").player_id))
print("first ids T1 test:", te1.player_id.head(3).tolist(), "T3 test:", te3.player_id.head(3).tolist())
