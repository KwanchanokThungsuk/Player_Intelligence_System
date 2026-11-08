# Round 1 (wirachat)

**Kaggle score:** 0.54617 (baseline was 0.50879)
**Date built:** 2026-11-09
**Branch:** runs/wirachat (based on main @ 3ce7e64, the merged baseline)

## What changed vs the baseline (PR #1, Kaggle 0.50879)
| Task | Baseline | Round 1 | CV (5-fold, 104k rows unless noted) |
|---|---|---|---|
| 1 | HistGB, threshold tuned on binary F2 | LightGBM (raw + engineered features averaged), threshold tuned on **macro F2** | macro F2 0.780 (same predictions with the baseline-style threshold: 0.720) |
| 2 | HistGB | LightGBM, noise columns dropped, class weights tuned on OOF | macro F1 0.7515 (baseline 0.749 on one split) |
| 3 | HistGB absolute error | Median blend of 6 LightGBM L1 models (raw scale) | NMAE 0.7854 (baseline 0.778 on one split) |
| 4 | ResNet18 | unchanged (baseline predictions reused) | - |
| 5 | Isolation Forest on raw columns, 5% contamination | Period-change features + robust z-score sum, threshold in a natural gap | no labels; flags 835 rows (3.2%) |

## Task 5 reasoning
- Anomalies show up as large changes in period 4 vs periods 1-3 (heavy tails only in `_4` columns).
- The per-row score (sum of capped squared robust z-scores over 29 period-change features) has a clear gap
  between about 124 and 219; about 825 rows lie above it. Threshold used: 130.
- Two types in that group: account takeover (location jumps, transfers, purchases; password_changed 61%,
  mass_item_sale 42%) and skill boosting / selling (skill_trajectory, friend_removes, skill_rating).
- 213 rows with only a moderate location shift (z about 4.5-6.6, probably legitimate travelers) are not flagged.
- Only unsupervised statistics fitted on the Task 5 test features (the only data provided). No model was trained
  on pseudo-labels.
- Not validated: no labels. If the baseline's predictions are compared against this flagged set, they overlap on only 161 rows.

## Reproduce
Run from the repo root with the project venv: `eda1.py`, `task1_cv.py` and `task1_cv.py --fe`, `task2_cv.py`,
`task3_cv.py`, `task3_tune.py`, `task5_delta.py`, `task5_score.py`, then `build_round1.py`
(writes `submission.csv`, which is not committed). Seed 342, 5-fold stratified CV.

## Notes
- Player IDs are NOT shared entities across tasks (overlaps match random chance); do not join tasks on player_id.
- Rows are not aligned across task files either (Task2.segment vs Task3.segment agree 0.280 at the same row index,
  chance 0.282), so there is no cross-task leakage to use.
- The Task 1 metric is ambiguous: the Kaggle page says "F2 (Macro)" but the slides only give F2. The threshold
  0.40 is best for macro F2 (0.780) but costs binary F2 (0.739 vs 0.837 at 0.18).
- Task 1: 13% of training labels are missing (rows dropped); test has fewer missing values than train (8% vs 12%).
- Task 3: whales (segment 3) cause 79% of the absolute error; NMAE looks capped near 0.785 with these features.
