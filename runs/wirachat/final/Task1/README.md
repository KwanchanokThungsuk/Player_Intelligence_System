# Task 1: anti-cheat pre-filter (binary, F2)
**Method.** Two LightGBM binary models (raw columns; raw + 5 engineered features) averaged over 5 folds, then averaged; threshold 0.18 tuned on OOF predictions for the plain positive-class F2. Seed 342. **Result:** OOF AUC 0.891, OOF F2 0.837.

| Step | Script |
|---|---|
| shared loading, folds, metrics | [`round-1/r1_common.py`](../../round-1/r1_common.py) |
| CV training (raw; `--fe` for engineered) | [`round-1/task1_cv.py`](../../round-1/task1_cv.py) |
| missing-value shift check, structure checks | [`round-2/t1_missing_shift.py`](../../round-2/t1_missing_shift.py), [`round-2/t1_structure.py`](../../round-2/t1_structure.py) |
| model ensemble check (no gain) | [`round-2/task1_ens.py`](../../round-2/task1_ens.py) |
| neural-net comparison (worse) | [`round-3/t12_nn.py`](../../round-3/t12_nn.py) |
| threshold step in the submission | [`round-2/build_round2_variants.py`](../../round-2/build_round2_variants.py) |
| used in the final file | [`round-3/build_round3_v14.py`](../../round-3/build_round3_v14.py) (Task 1 is identical to V14) |

Details and caveats: report section 2.
