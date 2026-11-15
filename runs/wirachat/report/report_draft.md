# CPE342 Karena Player Intelligence System: Technical Report (DRAFT, 2026-11-15)

Status: working draft written from the repo notes (`runs/wirachat/round-1..3/notes.md`). Items marked **[TEAM]** need the team's input, **[VERIFY]** need a check against the task slides or the final code, **[ESTIMATE]** are derived from leaderboard arithmetic, not measured.
Required by the project instruction (slide 6): methodology per task (EDA, preprocessing, model design, results); insights and common mistakes; domain interpretation (business impact); team collaboration (roles and takeaways); conclusion (lessons learned, ML insights, future improvements). English or Thai. Due 2026-11-30 on LEB2; Kaggle closes 2026-11-24.

## 0. Summary

| | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|---|---|---|---|---|---|
| Problem | binary, cheater pre-filter (plain F2) | 4-class player segment (macro F1) | spending regression (NMAE-based score) | 5-class game-screenshot classification (macro F1) | unsupervised account anomaly detection (plain F3) |
| Final approach | LightGBM (raw + engineered features), threshold tuned for F2 | LightGBM, noise columns dropped, tuned class weights | two-stage model + L1 LightGBM blend | ensemble of fine-tuned pretrained backbones (ConvNeXt-B/S/T, DINOv2-B), strong augmentation, multi-scale TTA | gap-based group discovery in feature space + classifier for rows with missing values |
| Quality (5-fold OOF, 2026-11) | F2 0.837 | macro F1 0.752 | 0.790 | held-out clusters 0.97; **[ESTIMATE]** about 0.90 on the test set | **[ESTIMATE]** about 0.89 |

Kaggle public leaderboard (equal task weights, so total = 5 x score): our best **0.83310** (2026-11-15). Leader on 2026-11-14: 0.8675.

Score history (public leaderboard): baseline 0.50879 -> round 1 0.54617 -> Task 1 threshold fix 0.56346 -> Task 5 groups 0.65561 -> Task 4 integer ids 0.69464 -> ConvNeXt 0.69578 -> new data release and corrected Task 4 key (V14) 0.81542 -> larger input and augmentation 0.81835 (V18) -> class-share adjustment 0.81973 (V20) -> big backbones 0.82473 (V22c) -> augmentation bug fixed 0.82772 (V24a) -> Task 5 missing-value flags 0.82867 (V25) -> fresh DINOv2-B 0.83096 (V26a) -> drop old models 0.83227 (V27a) -> all four fresh 0.83310 (V27c).

## 1. Common setup and rules we followed
- Data: only the provided Kaggle datasets. Seed 342, 5-fold stratified CV for Tasks 1-3 (OOF predictions for thresholds and blends; Task 1 has about 104k rows) [VERIFY row counts]. Tasks 1, 2, 3, 5 trained from scratch (no pretrained weights); Task 4 uses pretrained backbones as the rules allow. No AutoML.
- No test labels were used and no model was trained on test data (no pseudo-labelling).
- Licences of the pretrained backbones: ConvNeXt (MIT), DINOv2 (Apache-2.0), OpenCLIP ConvNeXt (MIT) **[VERIFY each model card]**.
- Disclosures (honest list): (a) Task 5 has only the unlabelled test file, so all analysis is unsupervised on it; (b) in round 2 we compared brightness/colour statistics of test and train images, which motivated the colour/brightness augmentation in Task 4; (c) the class-share ("prior") adjustment of Task 4 uses test predictions (Saerens EM), gains about +0.001, and is excluded from the clean variant of the final submission **[TEAM: confirm which two files are the final picks]**; (d) the Kaggle leaderboard was used to discover the metric definitions (plain vs macro F2/F3) and Task 4's row/ID format.

## 2. Task 1: Anti-cheat pre-filter (binary, F2)
**EDA.** 104k rows, 32 predictive columns (kill/death ratio, headshot percentage, win rate, accuracy, reaction time, ...). About 13% of training labels missing (rows dropped). Test has different per-column missingness than train (for example headshot_percentage 2.7% missing in train vs 14.1% in test) but identical distributions otherwise (KS about 0.01, same correlation structure).
**Preprocessing and features.** Native NaN handling in LightGBM; engineered interaction features; the averaged model uses raw and engineered feature sets.
**Model.** LightGBM; XGBoost/CatBoost blends gave no gain (0.8370 vs 0.8373). Logistic regression reaches AUC 0.765 vs 0.891, so the signal is non-linear. Four-leaf boosted trees already reach AUC 0.8899, so the task is saturated. Accounts with `reports_received` > 27.6 are never cheaters (1,011 rows).
**Metric insight.** The Kaggle page says "F2 (Macro)" but the metric is plain positive-class F2. We found this by submission: switching the threshold from 0.40 (best macro F2) to 0.18 (best plain F2, OOF F2 0.837) raised the score by +0.0173, exactly the predicted +0.098/5.
**Results.** OOF F2 0.837 at threshold 0.18; simulating the test missingness pattern on validation folds costs only 0.0009 AUC.
**Common mistake.** Trusting the page's metric label, and tuning the threshold for the wrong averaging.
**Business meaning.** The system ranks incoming cheat reports for manual review: high recall (F2 weights recall 4x precision) while the 500-case daily review capacity limits how many can be passed on; the false-positive rate of 70% in the current system is the problem this reduces **[VERIFY numbers against slide 14]**.

## 3. Task 2: Player segment classification (4 classes, macro F1)
**EDA.** New dataset in the 2026-11-12 release (same schema and label mix). Only `loyalty_tier` and `engagement_level` carry categorical information; missingness carries no signal; no gapped sub-populations.
**Model.** LightGBM with noise columns dropped and class weights tuned on OOF. Logistic regression (0.7506) equals LightGBM (0.7503), so the signal is linear and about 0.75 is the ceiling.
**Results.** OOF macro F1 0.7512 (0.7524 weighted). Test and train are statistically identical (adversarial AUC 0.50).
**Business meaning.** Segments (casual, competitive grinder, social, whale) allow targeted campaigns instead of generic ones (2-5% engagement now); whales are the top 5% of players and 60% of revenue **[VERIFY slide 16]**.

## 4. Task 3: Spending regression
**EDA.** `historical_spending = total_transactions x avg_transaction_value` exactly; `segment` is recoverable for 92% of rows when missing; the log-residuals are a 98%/2% mixture whose outliers are rows with missing or extreme history. Whales (segment 3) cause 79% of the absolute error. The positive part of spending is almost deterministic (log-space R2 0.99); whether spending is zero is mostly random (AUC 0.78 overall, 0.56-0.63 within segment).
**Model.** Two-stage model (lognormal median) blended with an exact-identity L1 LightGBM; median blend of six L1 LightGBMs as the fallback. OOF score 0.7896 (L1 blend 0.7854, two-stage alone 0.7884).
**Insight.** Knowing the zero outcome exactly would add only +0.008, so the target is close to its ceiling.
**Business meaning.** Predicted spending per player supports lifetime-value and monetisation planning **[VERIFY slide 18]**.

## 5. Task 4: Game screenshot classification (5 classes, macro F1)
**Data.** 230x120 images from the games 7 Knights, FiveN, Free Fried, PUBE, Roblock; train class shares 0.39/0.14/0.18/0.19/0.10; test images come from different sources than train (only 0.08% of test images have a train neighbour above 0.95 cosine, vs 2% within train), and are darker and more saturated.
**Format problems (common mistake).** Rounds 1-2 scored about 0.2 on Task 4 for weeks. Cause: the old `test.csv` listed image ids in a different order from the answer key; after the hosts fixed the file (keyed by ANS ids) predictions mapped by image id scored correctly. Lessons: map predictions by id, never by row position; check the answer key format before chasing model quality.
**Validation.** Held-out clusters of train images (DINOv2-small embeddings, 300 k-means clusters, 60 held out) give 0.965-0.971 macro F1 but do not reflect the test shift: the leaderboard, not the held-out score, tracked improvements.
**Model design (final).** Fine-tune pretrained backbones end to end on all training images at 224x448: ConvNeXt-Base (IN-22k), DINOv2-B/14, ConvNeXt-Small, ConvNeXt-Tiny; AdamW, one-cycle LR, label smoothing 0.1, class-balanced loss, fp16, 10-12 epochs; strong class-independent augmentation (random resized crop, flip, rotation to 45 degrees, colour jitter, channel shuffle, grayscale, equalize/autocontrast/posterize/solarize, blur, down-up resampling, noise, random erasing); test-time augmentation over 3 scales (x0.8, 1.0, 1.25) and flip. Ensemble: probability average, weights 2/2/1/1 (ConvNeXt-B, DINOv2-B, ConvNeXt-S, ConvNeXt-T).
**Key finding: augmentation bug.** Until 2026-11-15 the dataset class re-seeded the random generators from the image index, so every image got the same augmentation in every epoch (verified 16/16 identical images across epochs). Fixing it ("fresh augmentation") improved the score by +0.003 per replaced big model and made the old models dead weight: V22c 0.82473 -> V24a 0.82772 (ConvNeXt-B fresh) -> V26a 0.83096 (DINOv2-B fresh) -> V27a 0.83227 (old models dropped) -> V27c 0.83310 (small fresh models added).
**Other findings.** Higher input resolution and stronger augmentation raised the score (V14 0.81542 -> V18 0.81835); frozen CLIP features are useless here (zero-shot accuracy 0.265); a CLIP-pretrained ConvNeXt-B hurt with the flawed training (V22a 0.82221 vs 0.82473; retest with fresh augmentation pending); test class shares differ from train, and re-weighting probabilities by EM-estimated shares gives +0.001 (V19a 0.81941 vs 0.81835), optimum at exponent 1.
**Results.** **[ESTIMATE]** Task 4 about 0.90 on the test set (from 5 x 0.8331 minus the other tasks); gains per added model were +0.0003-0.0005 before the fix and +0.003 after.
**Business meaning.** Automatic game identification from screenshots (support tickets, user-submitted evidence, community moderation) without manual tagging **[VERIFY slide text]**.

## 6. Task 5: Account security monitoring (unsupervised, plain F3)
**Data.** 25,889 players x 122 features (4 time periods x 30 behaviour columns plus statics: `behavior_stability`, `password_changed`, `vpn_usage`, `suspicious_login_time`, `mass_item_sale`). No labels: only the test file exists.
**Metric insight.** Plain F3 = 10TP/(9T+n); from submissions T is about 2,070 (8.0% of rows) and our precision is about 100%; adding a flag helps whenever its anomaly probability exceeds F3/10 (about 9%).
**EDA / method.** Normal columns are uniform latent values plus per-period noise (kurtosis about -1.2). Anomalies come in six structured groups: (A) large period-4 change vs periods 1-3 (825 rows, change score gap between 150 and 221), and five persistent single-feature groups, each separated by an empty gap: low `behavior_stability` 246, low `ip_hash_entropy` + `login_time_variance` + `session_duration_std` 207 (same rows), low `communication_frequency` 263, low `activity_level` 265. 1,806 rows flagged (no overlap). Group A shows account-takeover patterns (location jumps, currency transfers, purchases; password change 40%, mass item sale 28%) and skill boosting/selling.
**Missing-value fix.** 4.9% of rows have `behavior_stability` NaN, invisible to the gap rule. The low-stability group has side effects (more purchases +1.3 sd, fewer logins -1.1 sd, older accounts +0.8 sd), so a gradient-boosting classifier (OOF AUC 0.982) scores the NaN rows; the 49 rows with probability above 0.09 are flagged. Leaderboard effect +0.00095 (Task 5 +0.0048, about 14 true anomalies where 8 were expected).
**What did not work.** Isolation Forest at 5% contamination (baseline); robust z-scores on bounded features (artefact); mixtures, gap scans on period means, slopes, within-row variance, lat/lon jumps and residual correlations found no further group. About 260 anomalies (T about 2,070 vs 1,806+49 flagged) probably remain unflagged.
**Business meaning.** The groups are distinct fraud/abuse patterns (account takeover, bots with low IP entropy and login-time variance, skill boosting) that could be routed to different review queues.

## 7. Insights and common mistakes (cross-task)
1. Read the metric definition from the leaderboard behaviour, not the page label (plain vs macro F-scores).
2. Verify the submission format end to end (Task 4 order/ID mismatch cost weeks).
3. A high held-out score can hide a train/test source shift; build validation that mimics the shift, and judge changes by the leaderboard with care.
4. Check data-loading code for silent bugs (fixed-per-image augmentation).
5. Ensembles help only when members are individually trained well; dropping weaker, flawed members improved the score.
6. Use the metric's structure (F-beta algebra) to decide thresholds and which rows to flag.
7. Keep test-derived steps out of the final pipeline or disclose them.

## 8. Team collaboration **[TEAM]**
Roles, contribution table, how work was split, Git workflow (branch per member, `runs/<name>` branches, PRs), takeaways. To be filled by the team.

## 9. Conclusion
Lessons learned, ML insights (feature/threshold tuning vs data/format correctness; augmentation and diversity beat plain ensembling; unsupervised structure discovery), and future improvements: more fresh-augmentation members (CLIP ConvNeXt-B, second DINOv2-B seed in progress), longer training of the big backbones, remaining Task 5 anomalies, systematic validation matching the test source.

## Appendix: reproducibility
Folders: `runs/wirachat/round-1/` (EDA, Tasks 1-3, Task 5 round 1), `round-2/` (Task 4 early work, Task 5 groups), `round-3/` (new data, Task 4 v2-v7, ensemble builders, Task 5 missing-value fix). Run order and commands in each `notes.md`. Final submission builders: `build_round3_t4_variant.py` (Task 4 ensemble), `t4_prior_adjust.py` (optional class-share step), `build_round3_t5_nanstab.py` (Task 5 flags).
