# Round 3 (wirachat)

**Best Kaggle score: 0.81542** (`submission_v14_full_new_data`, 2026-11-12). Before that: 0.69578 (round 2, `submission_v10_t4_convnext`). Round-2 notes: `../round-2/notes.md`.
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

## Next
- Improve Task 4 (now responsive to model quality): longer/larger fine-tune on all training images, TTA, ensembling; check per-class behaviour.
- Task 5: re-estimate what is still unflagged; Task 2/3 small gains; select the final 2 submissions before 2026-11-24; report and code due 2026-11-30.
