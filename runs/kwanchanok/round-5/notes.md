# Improvement Round 5

## Environment and data

- Branch: `runs/kwanchanok`
- Python environment: repository `.venv`
- CatBoost: 1.2.10; scikit-learn: 1.6.1; pandas: 2.3.3; NumPy: 2.0.2
- LightGBM and XGBoost were not installed and were not added.
- Data root: `cpe342-kaggle-main/`
- Task 1: 104,000 train rows (90,816 labeled), 31 raw predictors, 25,889 test rows.
- Task 2: 104,000 train rows, 44 predictors including 10 categorical columns, 25,889 test rows.
- Task 3: 104,000 train rows, 32 predictors, strongly skewed non-negative target, 25,889 test rows.
- Task 5: test-only unsupervised data, 25,889 rows and 122 predictors.

Official metrics were verified from the project PDF: Task 1 Binary F2; Task 2 multiclass F1 (averaging unspecified, so Macro F1 is the declared primary metric and Weighted F1 is also reported); Task 3 NMAE `1/(1 + sum(abs(y-pred))/sum(y))`; Task 4 Macro F1; Task 5 hidden-label F3 with no local labels.

## Validation and leakage controls

Tasks 1 and 2 used `StratifiedKFold(n_splits=5, shuffle=True, random_state=342)`. Task 3 used the corresponding shuffled five-fold `KFold`. Candidates within each task used exactly the same folds. Preprocessing and models were fitted only on each fold's training portion, every labeled row received one finite OOF prediction, and test data and Kaggle scores were not used for feature, threshold, blend, or model selection. Final predictions used five fold models and averaged probabilities/predictions.

Feature engineering was deterministic and row-local. It added missing count/rate, log1p only for observed non-negative skewed columns, and a small number of schema-supported ratios using missing values for zero denominators. Task 3 additionally used purchase frequency times average transaction value. Player IDs and targets were excluded.

## Task 1 - Binary F2

| Candidate | OOF F2 | Fold mean +/- SD | Threshold | Runtime (s) |
|---|---:|---:|---:|---:|
| HGB raw baseline | 0.835337 | 0.835335 +/- 0.002295 | 0.1950 | 5.93 |
| HGB engineered | 0.835531 | 0.835532 +/- 0.002299 | 0.1875 | 7.03 |
| CatBoost raw | 0.836055 | 0.836059 +/- 0.003078 | 0.1975 | 8.99 |
| CatBoost engineered | 0.835296 | 0.835297 +/- 0.002025 | 0.1975 | 9.22 |
| Blend CatBoost 0.25 | 0.836552 | 0.836554 +/- 0.002662 | 0.2000 | 1.24 |
| Blend CatBoost 0.50 | **0.836879** | **0.836882 +/- 0.002518** | 0.2050 | 1.24 |
| Blend CatBoost 0.75 | 0.836632 | 0.836634 +/- 0.002443 | 0.2050 | 1.25 |

The best blend won all five folds, with scores `[0.836583, 0.840234, 0.833831, 0.835341, 0.838419]`, but improved over the comparable HGB baseline by only 0.001542. This is below the predeclared 0.003 replacement threshold. The selected output therefore remains the root baseline, using its comparable OOF threshold 0.195; changed test rows: 0. The historical single-split root F2 0.835567 is reported only as context and is not directly subtracted from OOF results.

## Task 2 - Multiclass F1

| Candidate | OOF Macro F1 | OOF Weighted F1 | Fold Macro mean +/- SD | Runtime (s) |
|---|---:|---:|---:|---:|
| HGB raw baseline | 0.744579 | 0.758937 | 0.744574 +/- 0.002011 | 33.74 |
| HGB engineered | 0.744924 | 0.759270 | 0.744913 +/- 0.002181 | 36.26 |
| CatBoost raw | 0.748821 | 0.762962 | 0.748820 +/- 0.001983 | 330.21 |
| CatBoost engineered | 0.748519 | 0.762663 | 0.748511 +/- 0.001612 | 324.98 |
| Blend CatBoost 0.25 | 0.747953 | 0.761900 | 0.747947 +/- 0.003036 | <0.01 |
| Blend CatBoost 0.50 | 0.749705 | 0.763583 | 0.749702 +/- 0.002557 | <0.01 |
| Blend CatBoost 0.75 | **0.750436** | **0.764401** | **0.750433 +/- 0.001996** | <0.01 |

The CatBoost-weight-0.75 blend passed: gain 0.005857 over HGB raw, wins in all five folds, no minority-class F1 collapse, and fold Macro F1 `[0.752415, 0.752739, 0.748733, 0.749611, 0.748667]`. It changed 1,593 test rows. Probability columns were aligned to labels 0-3 before blending.

## Task 3 - NMAE

| Candidate | OOF NMAE | MAE (THB) | Fold NMAE mean +/- SD | Runtime (s) |
|---|---:|---:|---:|---:|
| HGB raw baseline | 0.776204 | 3,026.52 | 0.776224 +/- 0.003838 | 6.96 |
| HGB engineered | **0.776250** | **3,025.71** | **0.776276 +/- 0.004642** | 8.58 |
| ExtraTrees raw | 0.763627 | 3,249.25 | 0.763620 +/- 0.003961 | 46.70 |
| CatBoost raw target | 0.775491 | 3,038.95 | 0.775600 +/- 0.006620 | 15.80 |
| CatBoost log1p target | 0.631114 | 6,135.52 | 0.631156 +/- 0.003696 | 8.23 |
| Blend log weight 0.25 | 0.758824 | 3,336.26 | 0.758877 +/- 0.005504 | <0.01 |
| Blend log weight 0.50 | 0.717551 | 4,131.94 | 0.717607 +/- 0.005176 | <0.01 |
| Blend log weight 0.75 | 0.672349 | 5,115.45 | 0.672399 +/- 0.004404 | <0.01 |

Engineered HGB won/tied 3/5 folds, with NMAE `[0.777501, 0.780900, 0.774011, 0.769398, 0.779572]`, but the gain was only 0.000047 versus the required 0.001. High-value predictions passed the stability guard, but the root baseline was retained; changed rows: 0. Log-target models were transformed back with `expm1`, and all candidate metrics were calculated on the original target scale.

## Task 5 and Task 4

Task 5 reused Round 4's pre-existing stability evidence. Baseline and robust Isolation Forest both had mean seed Jaccard 0.234773, while LOF agreement with the baseline was 0.148050. No additional ensemble was justified without labels, so the root Isolation Forest prediction was copied exactly; anomaly rate 0.050021 and changed rows 0. Task 4 was not trained or altered; its ordered prediction hash is `3ba2dfa98f7e48151375e4f6283c6fe6f527647ed63196de80d6ebbd51ccbdff`.

## Submission decision

Only Task 2 passed. Therefore `combined.csv` and `task2_only.csv` are byte-equivalent and differ from the root baseline only in 1,593 Task 2 rows. Submit `task2_only.csv` first because it isolates the sole evidence-supported change. Do not upload `task1_only.csv`, `task3_only.csv`, or `task5_only.csv`; all are value-equivalent to the root baseline.

Suggested Kaggle description: `Round 5 Task 2 only - 5-fold OOF HGB/CatBoost 0.75 blend, Macro F1 0.750436, +0.005857 vs HGB, Task 4/root Tasks 1,3,5 unchanged`

## Reproduction

```bash
.venv/bin/python runs/kwanchanok/round-5/task1/train.py
.venv/bin/python runs/kwanchanok/round-5/task2/train.py
.venv/bin/python runs/kwanchanok/round-5/task3/train.py
.venv/bin/python runs/kwanchanok/round-5/task5/analyze.py
.venv/bin/python runs/kwanchanok/round-5/build_submission.py
```

## Kaggle result

- Submission: `task2_only.csv`
- Public score: `0.51051`
- Previous best baseline: `0.50879`
- Improvement: `+0.00172`

Only Task 2 was changed in this submission. The selected 5-fold OOF
HGB/CatBoost blend improved Task 2 OOF Macro F1 from 0.744579 to
0.750436 and changed 1,593 test predictions. Tasks 1, 3, 4, and 5
remained identical to the root baseline.

The public leaderboard improvement supports promoting the Round 5
Task 2 blend as the current best Task 2 prediction.