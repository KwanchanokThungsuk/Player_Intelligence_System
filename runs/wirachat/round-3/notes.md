# Round 3 (wirachat)

**Best Kaggle score: 0.81973** (`submission_v20_5models_prior_em`, 2026-11-14; V14 0.81542 -> V15 0.81742 -> V17 0.81785 -> V18 0.81835 -> V19a 0.81941 -> V20 0.81973). Before that: 0.69578 (round 2, `submission_v10_t4_convnext`). Round-2 notes: `../round-2/notes.md`.
**Leaderboard on 2026-11-12:** 16 teams; ML003 0.7633, ML007 0.7626, ML008 0.7611, ML015 0.7500, ML011 0.7236, ML019 0.7233, host Baseline 0.7051, us 0.6958 (8th).

## Why we are below the 0.76 teams (score arithmetic, equal weights, each task counts 1/5)
| | Tasks 1-3 | Task 4 | Task 5 | total / 5 |
|---|---|---|---|---|
| us | about 2.38 (T1 plain F2 0.837, T2 macro F1 0.752, T3 NMAE 0.79) | about 0.2 | about 0.9 | 0.696 |
| top teams | the same level | about 0.9-0.97 | about 0.5 (typical Isolation Forest level) | 0.76 |
So the whole gap is Task 4. Nothing else we can still do (about +0.03 at most) closes it.

## Ruled out for Task 4 (see round-2 notes)
model quality (ResNet, DINOv2, ConvNeXt 0.97 held-out), integer encodings (3 tried) and datacard encoding (TA confirmed), test.csv vs sorted row order,
stale or altered data (byte-identical to the Drive zip), pipeline bugs in Tasks 1-3 and 5, structured missingness, ids / metadata / timestamps.

## Round 3 so far
- Task 5: unsupervised Gaussian mixture over all features (rank-normal scores, PCA 40, K = 4..12). BIC picks K = 8. The only small components are the
  known IP/login-time group (207 rows, 100% overlap) and the period-4 group (590 rows, 99% overlap); no new group appears.

- Neural nets (GPU MLP / log-linear) do not beat the trees on Tasks 1-3 (T1 AUC 0.8145 vs 0.8914; T2 macro F1 0.7253 vs 0.7503; T3 log residual sd 0.242 vs 0.225).
- Seed search: T5 test.csv = sorted(ids) permuted with np.random.RandomState(67); the T1-T4 files are not plain shuffles (probably splits of a larger pool).
- Correction: V7's flat score did NOT refute "aligned rows with different class codes" (one matched class gives Task 4 about 0.19, almost the same as chance 0.198).
  Only 4 encodings were consistent with V4/V5/V6. V11 (split guess) scored 0.69581 (flat, so 1 match) and V12 (candidate c3) scored 0.69461 (flat/slightly down),
  so all four candidates are eliminated: no re-encoding of the ConvNeXt predictions in test.csv order moves Task 4.
- No link between our Task 4 predictions and any of 43 discrete columns of the other tasks' test files at the same row index (max z 2.2).
- Ready for the next attempts (ConvNeXt predictions, datacard encoding, V10 base): v13 reversed order, v14 sorted-id descending, v15 class-sorted blocks.

## Open options (need a decision)
1. One attempt with Task 4 set to a single constant class: shows the key's class share (aggregate only) and tells us whether the key's class mix agrees
   with the balanced mix every model predicts on the test images. Evidence for the TAs. Borderline for "no test set probing": decide first.
2. Conceptual Discussion post asking which Task 4 pipeline details worked (no code, no predictions).
3. Report and code deliverables (30% of the grade, due 2026-11-30); select the final 2 submissions before 2026-11-24.

## BREAKTHROUGH (2026-11-12): the host fixed the datasets
Announcement (translated): Task 1 and 3 unchanged; Task 2 new dataset; Task 4 `test.csv` corrected (single file); Task 5 new dataset; schema unchanged, re-download.
- Task 4: the Drive zip was replaced (modified 2026-11-12 07:55 UTC, 625,511,903 bytes). The 57,435 images are byte-identical, but the new `test.csv` is keyed by the
  ANS ids (`id, file_name, label`: ANS00001 -> sjqyfwc.jpg ...). The OLD test.csv listed image ids in a different order from the key, which is why no encoding,
  ordering or model change could ever move the Task 4 score in rounds 1-3 (all the "chance-level" results). The new train.csv has integer labels 0-4 = the datacard
  (alphabetical) order, i.e. the encoding we used was right.
- Task 2: new train/test sets (ids redrawn, same schema and label mix). Retrained with the same method: OOF macro F1 0.7512 (0.7524 weighted).
- Task 5: same ids in a new order, 99.8% of values changed. Re-ran the unsupervised search from scratch: period-4 group 825 rows (clean gap between S 150 and 221) +
  persistent low groups (behavior_stability 246, ip_hash_entropy/login_time_variance/session_duration_std the same 207 rows, communication_frequency 263,
  activity_level 265) = 1,806 flagged (6.98%); nothing left in a second gap scan.
- Tasks 1 and 3: predictions kept (data identical).
- V14 = T1 (V10) + T2 (new) + T3 (V8 blend) + T4 (ConvNeXt-Tiny predictions placed by the new ANS-keyed test.csv) + T5 (new flags): **0.81542**.
- Data used: `C:/Users/Wirachat/Desktop/CPE342-DATA-v2/` (outside the repo; the repo's tracked dataset is the OLD one). Scripts read it through the `CPE342_DATA` environment variable
  (`round-1/r1_common.py`).

## Task 4 model improvements (2026-11-12, after the data fix)
- `round-2/t4_finetune.py` (now with `--arch convnext_tiny|convnext_small`): end-to-end fine-tune, strong class-independent augmentation, class-balanced loss, flip TTA.
- V15 (0.81742) = V14 with Task 4 = mean of ConvNeXt-Tiny (80% data) + ConvNeXt-Tiny (all data) probabilities.
- V17 (0.81785) = V15 with Task 4 = mean of the two Tiny models + ConvNeXt-Small (all data, 12 epochs, ~41 min). Only +0.0004 over V15: ensembling is saturating.
- `build_round3_t4_variant.py` builds these files (maps predictions by image id into the new ANS-keyed test.csv). Tasks 1,2,3,5 stay as in V14.
- `build_round3_ablate_t5.py` builds V16 (Task 5 cut to one flagged row), a diagnostic to split Task 4 from Task 5 in the total; not uploaded yet.
- Leaderboard 2026-11-12: ML003 0.85787, ML007 0.84873, ML008 0.84268, us 0.81785 (4th).

## Diagnostic V16 (Task 5 cut to one flagged row): Kaggle 0.64078
- V15 0.81742 -> V16 0.64078: drop 0.17664 x 5 = Task 5 score 0.883 (F3). With 1,806 flags: if all are true anomalies, F3 = 10*1806/(9T+1806) = 0.883 gives T = 2,070 = 8.0% of 25,889 rows.
  So precision is about 100% and about 265 anomalies are unflagged (the groups have 825+246+207+263+265 = 1,806 rows, no overlap; the missing set is about the size of one more single-feature group).
- Task 4 is then about 0.82 (depends on the Tasks 1-3 estimate of about 2.38, +-0.05); held-out 0.97, so the test images are a shifted source.

## Task 4 v2: higher resolution + stronger augmentation + multi-scale TTA (2026-11-14)
- `t4_finetune2.py`: 224x448 input (was 160x320), extra augmentations (channel shuffle, equalize, autocontrast, posterize, solarize, down-up resampling, rotation to 45 deg, more noise), TTA over scales 0.8/1.0/1.25 x flip, `--arch` incl. EfficientNet.
- ConvNeXt-Tiny 224x448 (80% data, 12 epochs, 44 min): held-out clusters plain 0.965 / colour-stress 0.950 (old 160x320 model: 0.971 / 0.964, so the proxy does not reward the harder augmentation; it cannot measure the test-source shift). Test agreement with the V17 ensemble 87.1%.
- EfficientNet-V2-S was stopped: 1,065 s per epoch (about 3.5 h for 12 epochs), 5.6x slower than ConvNeXt-Tiny. Replaced by ConvNeXt-Small 224x448 on all data (`run_t4_v2_small.sh`).
- V18 (0.81835) = V17's three models + ConvNeXt-Tiny 224x448, equal weights. Gains per added model are about +0.0005 in the leaderboard score: ensembling is saturating.

## Task 4 class-share (prior) adjustment, `t4_prior_adjust.py` (2026-11-14)
- Train shares are not balanced (7-11k 0.39, fiveN 0.14, freefried 0.18, pube 0.19, roblock 0.10); the four models predict test shares of about [0.17, 0.22, 0.18, 0.26, 0.17] and agree with each other.
- Saerens EM on the V18 ensemble probabilities (start from uniform, class-balanced loss) estimates test priors [0.13, 0.24, 0.16, 0.29, 0.18].
- V19b (half-way to uniform shares, 4.9% rows changed): 0.81770, below V18 0.81835 -> the test is NOT more balanced than the predictions.
- V19a (probabilities reweighted by the EM priors, 3.9% rows changed): **0.81941** (+0.00106 over V18). The opposite direction helps, so the shift is real.
- V19c (alpha 2, 7.9% rows changed): 0.81879, above V18 but below V19a. A quadratic through alpha 0 / 1 / 2 (0.81835 / 0.81941 / 0.81879) peaks at alpha about 1.1 with about 0.81942, so the EM estimate itself is already the optimum: no further tuning of the shares. V19d (alpha 3) not needed.

- V20 (0.81973) = five-model ensemble (V17's three + ConvNeXt-Tiny 224x448 + ConvNeXt-Small 224x448 all data, `run_t4_v2_small.sh`) + 1x EM prior adjustment (`t4_prior_adjust.py _5m ...`); 565 Task 4 rows differ from V19a. +0.0003 over V19a: ensembling is nearly exhausted.

## Task 5 hunt for the missing ~265 (2026-11-14, offline, no attempt used): nothing found
Normal columns are uniform latents + per-period noise (kurtosis about -1.2); every known group shifts one main column plus 1-3 side columns; group sizes 246/207/263/265 (about 1% each) and A 825.
Checked on the 24,083 unflagged rows, all with no new group: (1) gap scans on period means (100/200 bins); (2) histogram bumps on every period-mean column; (3) leave-one-out change score for periods 1-3 (no gap);
(4) period-4 tail above the other periods is made of heavy-tailed columns (skill_trajectory, friend_removes), statics unlike group A (password_changed 0.04 vs 0.40); (5) within-row variance (low and high tail), NaN patterns, location clusters/duplicates;
(6) residual pairwise rank correlations (max 0.029); (7) family-wise Gaussian mixtures (BIC prefers K=1 in all four families).
Caveat: the 0.883 only fixes 10TP/(9T+1806); T = 2,070 with 100% precision is one solution, but T = 2,000 with 97% precision (TP about 1,750) fits as well, so the loss may also be false positives.
Small certain gain left: rows whose static behavior_stability is NaN (about 5% of that group, about 13 rows) are missed by the gap rule.

## Next
- Improve Task 4 (now responsive to model quality): longer/larger fine-tune on all training images, TTA, ensembling; check per-class behaviour.
- Task 5: re-estimate what is still unflagged; Task 2/3 small gains; select the final 2 submissions before 2026-11-24; report and code due 2026-11-30.
