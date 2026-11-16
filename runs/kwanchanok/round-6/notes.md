# Task 3 Improvement Round 6

## Environment and data analysis

- Branch: `runs/kwanchanok`; no shared project files were changed.
- CatBoost `1.2.10` was reused. LightGBM and XGBoost were not installed and were not added.
- Data: `cpe342-kaggle-main/03_regression/train.csv` and `test.csv`.
- Train rows: 104,000; test rows: 25,889; predictors: 32 numeric columns; target: `spending_30d`.
- Target min/max/mean/median/std: `0`, `300000`, `10497.040585`, `79.435`, `30568.814479`.
- Quantiles (0, .25, .50, .75, .90, .95, .99, .995, 1): `0`, `0`, `79.435`, `2461.14`, `37718.851`, `75969.751`, `150032.2954`, `187037.26135`, `300000`.
- Zero targets: 50,049 (`48.1240%`); targets <= 1: 50,075 (`48.1490%`); skewness: `4.170021`.
- This meaningful zero mass justifies a two-stage positive-spending model. The target is strongly right-skewed, but test targets were never used.
- Largest train/test standardized mean differences among numeric features were small: primary_game `0.02085`, owns_limited_edition `-0.01789`, competitive_rank `0.01292`, total_playtime_hours `0.01002`, guild_membership `0.00986`.
- Exact duplicate feature rows with conflicting targets: `0` groups. Duplicate feature rows were counted in the metrics JSON.

## Validation and leakage controls

All candidates use the same `KFold(n_splits=5, shuffle=True, random_state=342)`. The fold assignment hash and sizes are stored in `outputs/metrics/task3.json`. Preprocessing, classifiers and regressors are fitted only on each fold's training rows. Every labeled row receives exactly one finite OOF prediction per candidate. Two-stage Stage 2 models use only positive rows from the fold's training partition. Gate threshold selection and blend selection use OOF predictions only. Metrics are computed on original THB scale using the official `1/(1 + sum(abs(y-pred))/sum(y))` NMAE score; higher is better.

Features are conservative and row-local: missing count/rate, log1p transforms of observed non-negative skewed spending/count features, safe behavioral ratios with zero denominators converted to missing, and purchase-frequency times average transaction value. `player_id` and target are excluded.

## Candidate comparison

| Candidate | OOF NMAE | OOF MAE | Fold mean +/- SD | Wins/ties vs HGB |
|---|---:|---:|---:|---:|
| HGB raw comparable baseline | 0.776203690 | 3026.523809 | 0.776223815 +/- 0.003838469 | baseline |
| HGB engineered | 0.775891339 | 3031.968002 | 0.775917102 +/- 0.004563806 | 2/5 |
| CatBoost raw target | 0.774851391 | 3050.125630 | 0.774949895 +/- 0.007134529 | 2/5 |
| CatBoost log1p target | 0.629770994 | 6170.987465 | 0.629817127 +/- 0.003389263 | 0/5 |
| ExtraTrees raw | 0.763839554 | 3245.427352 | 0.763832419 +/- 0.003998325 | 0/5 |
| Two-stage expected value | 0.772827214 | 3085.608148 | 0.772856853 +/- 0.004607700 | 2/5 |
| **Two-stage hard gate** | **0.778422264** | **2987.980428** | **0.778435332 +/- 0.004668234** | **5/5** |
| Blend log weight 0.25 | 0.758922584 | 3334.463182 | 0.758971445 +/- 0.004727429 | 0/5 |
| Blend log weight 0.50 | 0.717196782 | 4139.166452 | 0.717249976 +/- 0.004635571 | 0/5 |
| Blend log weight 0.75 | 0.671387598 | 5137.803754 | 0.671437456 +/- 0.003982355 | 0/5 |

The selected hard-gate fold NMAE values were `[0.781487238, 0.781667720, 0.774660077, 0.772180462, 0.782181162]`. Its OOF gain over the comparable Round 5 HGB baseline is `0.002218574`, it wins all five folds, has no severe fold degradation, and passes the high-value distribution guard. The OOF gate threshold is `0.50`. The selected five-fold test ensemble changed 25,425 Task 3 rows versus both root and Round 5 Task 3 predictions.

## Submission composition and verification

`task3_on_round5_best.csv` uses root Task 1, **Round 5 selected Task 2**, Round 6 Task 3, root Task 4, and root Task 5. The builder asserts that Round 5 Task 2 differs from root Task 2 by exactly 1,593 rows. `task3_only_root_base.csv` is diagnostic only and restores root Task 2.

Both submissions have 25,889 rows and 6 columns, use the detected comma delimiter, preserve sample ID order, pass test-order/domain/finite-value checks, and pass readback validation. Task 1, Task 4 and Task 5 are identical to root in the primary submission; Task 2 retains the Round 5 improvement. No Kaggle submission was made.

Recommended upload: `task3_on_round5_best.csv`.

Suggested Kaggle description: `Round 6 Task 3 on Round 5 best - two-stage HGB hard gate, OOF NMAE 0.778422, +0.002219 vs comparable HGB; Round 5 Task 2 retained; Tasks 1,4,5 unchanged`

## Reproduction

```bash
.venv/bin/python runs/kwanchanok/round-6/task3/train.py
.venv/bin/python runs/kwanchanok/round-6/build_submission.py
```
