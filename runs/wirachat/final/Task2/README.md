# Task 2: player segment (4 classes, macro F1)
**Method.** LightGBM multiclass, 5-fold, `random_metric*` noise columns dropped, categoricals as pandas categories, class weights tuned by coordinate search on the OOF probabilities. Retrained on the host's new Task 2 data (2026-11-12). **Result:** OOF macro F1 0.7512 (argmax), 0.7524 (weighted).

| Step | Script |
|---|---|
| final training on the new data | [`round-3/t2_new.py`](../../round-3/t2_new.py) |
| method origin (old data) | [`round-1/task2_cv.py`](../../round-1/task2_cv.py), [`round-2/t2_models.py`](../../round-2/t2_models.py) |
| diagnostics (noise columns, structure) | [`round-1/task2_diag.py`](../../round-1/task2_diag.py), [`round-2/t23_gapscan.py`](../../round-2/t23_gapscan.py) |
| neural-net comparison (worse) | [`round-3/t12_nn.py`](../../round-3/t12_nn.py) |
| used in the final file | [`round-3/build_round3_v14.py`](../../round-3/build_round3_v14.py) |

Details: report section 3.
