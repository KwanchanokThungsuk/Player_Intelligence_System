# Round 2 (wirachat)

**Best Kaggle score:** 0.65561 (`submission_v3_t5_groups`; round 1 was 0.54617, baseline 0.50879)
**Branch:** runs/wirachat (based on main @ 3ce7e64)

## Submissions this round (each differs from its base in one task only)
| File | Change | Kaggle |
|---|---|---|
| submission_v1_t1_thr018 | Task 1 threshold 0.40 -> 0.18 (round-1 base 0.54617) | 0.56346 |
| submission_v2_t5_loose | Task 5 flags 1,350 rows (S>80) instead of 835 (round-1 base 0.54617) | 0.54818 |
| submission_v3_t5_groups | V1 + Task 5 = obvious group + four persistent odd groups (1,819 rows) | **0.65561** |

## What we learned
- **Task 1 is scored with plain (positive-class) F2, not macro.** V1 went up although macro F2 would have fallen.
  The Kaggle page's "(Macro)" labels are not reliable. Best OOF threshold for plain F2 is about 0.18 (OOF F2 0.837).
- **Task weights look equal (about 1/5 each).** The V1 gain (+0.0173) matched the predicted +0.098/5.
- **Task 5 is also scored with a plain (positive-class) F3**, which is 10*TP/(9*T+n) (T = true anomalies, n = flagged).
  From the scores, the true anomaly count is about 2,000 (about 7-8% of rows); 1,819 are flagged now.
- CatBoost/XGBoost blends do not help Task 1 (plain F2 0.8370 vs 0.8373 for LightGBM alone).
- Task 4's grader seems to return a constant, so Task 4 adds nothing to the score for now.

## Task 5: the anomaly groups (unsupervised, fitted on the Task 5 test features only; no pseudo-labelling)
1. Obvious group (835 rows): large period-4 change vs periods 1-3 (score S>130, see round-1 notes).
2. Four persistent groups, the same rows in all four periods, each separated from the rest by an empty gap,
   with no overlap between groups or with group 1:
   - low ip_hash_entropy and low login_time_variance together: 207 rows
   - low communication_frequency: 237 rows
   - low activity_level: 266 rows
   - low behavior_stability (0.20-0.39, none between 0.40 and 0.68): 274 rows
3. A second gap scan with these removed finds no more gap-defined groups.
- False lead: the apparent ip_hash_entropy "tail" under a robust z-score was an artefact of a bounded feature.
- Missing values: about 13 members of the behavior_stability group are probably missed (that column is NaN in 4.9% of rows).

## Ideas not done yet
- About 200 anomalies are probably still unflagged (estimated T about 2,000); they may have no gap.
- Tasks 2 and 3 may not be at their ceiling (Task 2 CV macro F1 0.7515, Task 3 CV NMAE 0.7854). A structure search
  like the one that worked for Task 5 has not been done there.
- Task 1 is at AUC 0.89; no feature engineering has helped so far.

## Reproduce
From the repo root with the project venv: round-1 scripts first (they produce the OOF/test arrays and the Task 5 score),
then `build_round2_variants.py` (V1, V2), `task5_subtle.py`, `task5_ip.py`, `task5_stability.py`, `task5_gapscan.py`,
`task5_groups2.py`, `task5_gapscan2.py` (writes the group indicators), and `build_round2_v3.py` (V3).
`task1_ens.py` is the XGBoost/CatBoost check. Submission CSVs and arrays are local and not committed.
