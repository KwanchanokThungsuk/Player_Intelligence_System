# Round 2 (wirachat)

**Best Kaggle score:** 0.69464 (`submission_v4_t4_int`; V3 0.65561, round 1 0.54617, baseline 0.50879)
**Branch:** runs/wirachat (based on main @ 3ce7e64)

## Submissions this round (each differs from its base in one task only)
| File | Change | Kaggle |
|---|---|---|
| submission_v1_t1_thr018 | Task 1 threshold 0.40 -> 0.18 (round-1 base 0.54617) | 0.56346 |
| submission_v2_t5_loose | Task 5 flags 1,350 rows (S>80) instead of 835 (round-1 base 0.54617) | 0.54818 |
| submission_v3_t5_groups | V1 + Task 5 = obvious group + four persistent odd groups (1,819 rows) | 0.65561 |
| submission_v4_t4_int | V3 with Task 4 written as integer class ids (0=7-11k, 1=fiveN, 2=freefried, 3=pube, 4=roblock) | **0.69464** |

## What we learned
- **Task 1 is scored with plain (positive-class) F2, not macro.** V1 went up although macro F2 would have fallen.
  The Kaggle page's "(Macro)" labels are not reliable. Best OOF threshold for plain F2 is about 0.18 (OOF F2 0.837).
- **Task weights look equal (about 1/5 each).** The V1 gain (+0.0173) matched the predicted +0.098/5.
- **Task 5 is also scored with a plain (positive-class) F3**, which is 10*TP/(9*T+n) (T = true anomalies, n = flagged).
  From the scores, the true anomaly count is about 2,000 (about 7-8% of rows); 1,819 are flagged now.
- CatBoost/XGBoost blends do not help Task 1 (plain F2 0.8370 vs 0.8373 for LightGBM alone).
- **Task 4 was a format problem, not a broken grader.** The sample submission's Task 4 column is numeric, and class names
  never matched, so Task 4 scored about 0. With integer ids the score rose only +0.039, i.e. Task 4 itself rose by about
  0.195 (each task counts 1/5): about one class matched, or chance-level alignment. A fully correct encoding would add
  about 0.9. Candidate encodings that do NOT explain it: Kaggle overview order (0 classes match the datacard order),
  first-appearance order (2 match), frequency order (3 match). The Kaggle Discussion board is empty.
- Score decomposition (equal weights) fits: T1 about 0.84, T2 0.75, T3 0.79, T4 about 0.2, T5 about 0.90.
- Task 2: logistic regression (0.7506) equals LightGBM (0.7503), so the signal is linear and 0.75 is the ceiling. Only
  loyalty_tier and engagement_level carry categorical information; missingness carries none; no gapped sub-populations.
- Task 3: the positive part of spending is almost deterministic (log-space R2 0.99); the zero/non-zero outcome is mostly
  random (AUC 0.78). A two-stage lognormal-median model gets NMAE 0.7884 vs 0.7854 for the L1 blend (0.75/0.25 blend 0.7889).
- Task 1: strongly nonlinear (logistic regression AUC 0.765 vs 0.891 for LightGBM). Accounts with reports_received > 27.6
  are never cheaters (1,011 rows).
- Task 5: no further gap-defined group exists. About 240 anomalies are probably still unflagged; checks on count-like columns,
  persistent low tails (an artefact of uniform baselines) and derived statistics found nothing.

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

## Next
- Find the correct Task 4 integer encoding (biggest lever, about +0.15). Plan: use the score as feedback on how many classes
  match, starting from the 45 encodings consistent with one match; or ask the TAs for the encoding and key row order.
- Small extras: Task 3 two-stage blend (about +0.0007 overall); about 240 unflagged Task 5 anomalies (about +0.02 at most).

## Reproduce
From the repo root with the project venv: round-1 scripts first (they produce the OOF/test arrays and the Task 5 score),
then `build_round2_variants.py` (V1, V2), `task5_subtle.py`, `task5_ip.py`, `task5_stability.py`, `task5_gapscan.py`,
`task5_groups2.py`, `task5_gapscan2.py` (writes the group indicators), and `build_round2_v3.py` (V3).
`task1_ens.py` is the XGBoost/CatBoost check. `task5_remaining.py`, `task5_econ.py`, `task5_zeros.py` and `task5_gapscan3.py`
are the further Task 5 searches; `t23_gapscan.py`, `t1_structure.py`, `t2_models.py` and `t3_twostage.py` are the Task 1-3
structure checks; `build_round2_v4.py` builds V4. Submission CSVs and arrays are local and not committed.
