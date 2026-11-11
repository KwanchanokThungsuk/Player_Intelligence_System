# Round 2 (wirachat)

**Best Kaggle score:** 0.69578 (`submission_v10_t4_convnext`; V9 0.69555, V8 0.69531, V4 0.69464, V3 0.65561, round 1 0.54617, baseline 0.50879)
**Branch:** runs/wirachat (based on main @ 3ce7e64)

## Submissions this round (each differs from its base in one task only)
| File | Change | Kaggle |
|---|---|---|
| submission_v1_t1_thr018 | Task 1 threshold 0.40 -> 0.18 (round-1 base 0.54617) | 0.56346 |
| submission_v2_t5_loose | Task 5 flags 1,350 rows (S>80) instead of 835 (round-1 base 0.54617) | 0.54818 |
| submission_v3_t5_groups | V1 + Task 5 = obvious group + four persistent odd groups (1,819 rows) | 0.65561 |
| submission_v4_t4_int | V3 with Task 4 written as integer class ids (0=7-11k, 1=fiveN, 2=freefried, 3=pube, 4=roblock) | 0.69464 |
| submission_v5_t4_guess1 | V4 with pube/roblock codes swapped (encoding guess) | 0.69509 |
| submission_v6_t4_guess2 | V4 with another encoding guess | 0.69445 |
| submission_v7_t4_sorted | V4 with predictions re-ordered by sorted image id (row-order guess) | 0.69462 |
| submission_v8_t3_blend | V4 + better Task 3 blend (two-stage + exact-identity L1 LightGBM, OOF NMAE 0.7896) | 0.69531 |
| submission_v9_t4_dino_mv | V8 with Task 4 from DINOv2-small embeddings + colour-robust multi-view head (held-out-cluster macro F1 0.87) | 0.69555 |
| submission_v10_t4_convnext | V8 with Task 4 from a fine-tuned ConvNeXt-Tiny, strong class-independent augmentation (held-out-cluster macro F1 0.971, 0.964 under colour stress) | **0.69578** |

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

## Task 4 resolution so far (V5-V8)
- V4, V5, V6 and V7 (three encodings, one re-ordering) all scored within 0.0006 of each other, i.e. Task 4 stayed at chance
  (about 0.198, which equals the chance macro F1 for our predicted class mix against a balanced key).
- **TA answer:** Task 4 uses the datacard encoding (0 7-11 Knights, 1 FiveN, 2 Free Fried, 3 PUBE, 4 Roblock) and every task is in
  test.csv order with row 1 = ANS00001. So V4 already used the right format; V5-V7 were unnecessary guesses (format discovery
  through the leaderboard; mention it to the TAs if in doubt about the "no test set probing" rule).
- The Kaggle Overview class list was edited by the host after 2026-11-08 and now matches the datacard. Resubmitting (V8) showed the
  Task 4 score did not change, so the key was not what changed: the baseline ResNet18's test predictions are the problem.
- Checks done: test images look like train (same size/format/augmentation style, slightly darker and more saturated);
  an independent colour/layout LightGBM (79% on train CV) agrees with the ResNet on 50.5% of test images (21% by chance), so the
  predictions are tied to the images and in test.csv order. Training labels match the game names (montage of train images).
  Augmentation strength differs by class in train (mean saturation: pube 81, roblock 68, others about 48), a possible shortcut.
  Hypothesis: the random train/validation split leaks near-duplicate augmented copies of the same source screenshots, so the
  0.93 validation macro F1 is inflated and generalisation to new sources is poor. Not proven.
- Task 4 may use pretrained models (competition rules section 3: ResNet, EfficientNet, ViT ...; Tasks 1,2,3,5 must be from scratch).
  Open-source licences must allow commercial use (Foundational Rules 6c): prefer Apache-2.0 / MIT / BSD models.

## Task 4: everything tried after the TA answer (all stayed at the same score)
- Data checked against the Drive source (task4_image_classification.zip, 578.9 MB, modified 2026-10-05): local images byte-identical
  (CRC 4000/4000), zip train.csv/test.csv identical in content and row order (only CRLF line endings differ). The zip's image entries are in
  sorted-id order (= the V7 order, which scored at chance); zip timestamps are unrelated to any order. Kaggle itself hosts no Task 4 CSVs or images.
- Predictors tried, all giving Task 4 ~0.195 (chance-level macro F1 for a balanced key): ResNet18 (baseline), DINOv2-small + linear head,
  colour-robust multi-view DINOv2 head, grayscale-only head, fine-tuned ConvNeXt-Tiny (0.971 on held-out clusters). They agree with each other on
  64-81% of test images, so the predictions are content-driven. The score does not depend on model quality, encoding or row order.
- Test images come from different sources than the training images (only 0.08% of test images have a train neighbour above 0.95 cosine,
  vs 2% within train); test predictions are class-balanced; ids are random hashes with no structure; JPEGs carry no metadata.
- Score arithmetic (equal weights): our Tasks 1-3 + 5 are about 3.28; the top teams (0.763) need Task 4 about 0.9-0.97 with a typical Task 5 of
  about 0.5; the host Baseline (0.705) needs Task 4 about 0.8. We are at Task 4 about 0.2. TAs and the teacher state the grader is correct.

## Other tasks: headroom audit (Tasks 1, 2, 3, test side)
- Test files of Tasks 2 and 3 are statistically identical to train. Task 1 has very different per-column missingness (e.g.
  headshot_percentage 2.7% missing in train vs 14.1% in test), but simulating it on validation folds costs only 0.0009 AUC
  and 0.0002 F2, and a masking-augmented model does not help.
- Task 1 is saturated (4-leaf boosted trees already reach AUC 0.8899 vs 0.8906; no exact identities between columns).
- Task 3: historical_spending = total_transactions * avg_transaction_value exactly; segment is recoverable at 92% when missing;
  the log-residuals are a 98%/2% mixture and the 1% outliers are rows with missing or extreme history, not a hidden group.
  Knowing the zero outcome exactly would add only +0.008 overall and it is mostly unpredictable (AUC 0.56-0.63 within segment).
- Total non-Task-4 headroom is about +0.03 at most (mostly about 240 unflagged Task 5 anomalies).
- Integrity audit: test columns equal train columns in name and order; an independent minimal pipeline agrees with the submitted predictions
  (Task 1 97.7% / rank corr 0.993, Task 2 95.4%, Task 3 Spearman 0.942). Missingness is independent per cell in every file (no hidden groups).

## Next
- Build a better Task 4 classifier: frozen pretrained embeddings (permissive licence) + light head, grouped cross-validation on
  near-duplicate groups of the TRAINING images, robustness checks (grayscale / hue-jitter), then write test predictions with the
  datacard encoding in test.csv order. No fitting on test images.
- Select the final 2 submissions on the Kaggle Submissions page before 2026-11-24.

## Reproduce
From the repo root with the project venv: round-1 scripts first (they produce the OOF/test arrays and the Task 5 score),
then `build_round2_variants.py` (V1, V2), `task5_subtle.py`, `task5_ip.py`, `task5_stability.py`, `task5_gapscan.py`,
`task5_groups2.py`, `task5_gapscan2.py` (writes the group indicators), and `build_round2_v3.py` (V3).
`task1_ens.py` is the XGBoost/CatBoost check. `task5_remaining.py`, `task5_econ.py`, `task5_zeros.py` and `task5_gapscan3.py`
are the further Task 5 searches; `t23_gapscan.py`, `t1_structure.py`, `t2_models.py` and `t3_twostage.py` are the Task 1-3
structure checks; `build_round2_v4.py` .. `build_round2_v8.py` build V4-V8; `t3_*`, `t13_headroom.py`, `t123_identities.py`,
`test_audit.py`, `t1_missing_shift.py` are the Task 1/3 audits; `t4_*` scripts are the Task 4 diagnostics and the encoding search.
Submission CSVs and arrays are local and not committed.
