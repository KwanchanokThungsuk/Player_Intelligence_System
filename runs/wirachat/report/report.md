# CPE342 Karena Player Intelligence System: Technical Report

Course: CPE 342 Machine Learning (1/2026), KMUTT. Kaggle team ID: **[TEAM: ML0xx]**. Members: **[TEAM: names and student IDs]**. Written from the repository notes (`runs/wirachat/round-1..3/notes.md`) and the Kaggle scores recorded in them.

How to read the markers: **[TEAM]** = only the team can fill this in; **[VERIFY]** = check against the slides, the model card or the final code before submitting; **[ESTIMATE]** = derived from leaderboard arithmetic, not measured on labelled data. Numbers without a marker were measured (cross-validation) or are Kaggle public-leaderboard scores copied from the notes.

---

## 0. Summary

| | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 |
|---|---|---|---|---|---|
| Problem | cheater pre-filter, binary, F2 | player segment, 4 classes, macro F1 | 30-day spending, NMAE score | game title from a screenshot, 5 classes, macro F1 | account anomaly detection, unsupervised, F3 |
| Final approach | LightGBM (raw and engineered features, averaged), threshold tuned for plain F2 | LightGBM multiclass, noise columns dropped, class weights tuned on OOF | blend of two-stage lognormal-median models and an L1 LightGBM with exact-identity features | probability average of fine-tuned pretrained backbones (DINOv2-L, DINOv2-B, ConvNeXt-B/S/T) at 224x448, new fine-tuning recipe, class-prior (EM) step | gap-based discovery of six anomaly groups + a classifier for rows with missing values + two tail rows |
| Quality | OOF F2 **0.837** (threshold 0.18) | OOF macro F1 **0.7524** | OOF score **0.7907** | held-out clusters 0.97-0.98 (does not measure the test shift); **[ESTIMATE]** about 0.94 on the test set | **[ESTIMATE]** about 0.89 (F3 0.883 measured with 1,806 flags) |

**Leaderboard.** Final public score **0.84167** (V57a), 14th of 22 teams on 2026-11-24 (leader 0.87346; **[VERIFY final standing]**). The two submissions selected for the final ranking are **V57a (0.84167)** and **V53c (0.84113)**; both contain the class-prior (EM) adjustment of Task 4 (section 5.4). The same ensemble without it, V57b, scored 0.83980.

Task weights: the slides state a weighted mean `(w1 S1 + ... + w5 S5) / sum(w)` but the weights are not readable in the PDF text we extracted. Leaderboard arithmetic shows equal weights of 1/5 (section 1.3) **[VERIFY with the TAs]**. Task 4 estimate: 5 x 0.84167 = 4.208, minus Task 1 0.837, Task 2 0.752, Task 3 0.791 and Task 5 about 0.89 leaves about 0.94 **[ESTIMATE, assumes the OOF scores hold on the test set]**.

**Score history (Kaggle public).**

| Version | Score | What changed |
|---|---|---|
| baseline | 0.50879 | provided baseline pipelines |
| round 1 | 0.54617 | LightGBM Tasks 1-3, first Task 5 attempt |
| V1 | 0.56346 | Task 1 threshold for plain F2 (0.40 -> 0.18) |
| V3 | 0.65561 | Task 5 group discovery |
| V4/V10 | 0.69464 / 0.69578 | Task 4 integer ids; ConvNeXt predictions (Task 4 still about 0.2 because of the host's file) |
| V14 | 0.81542 | **host corrected the datasets** (Task 4 `test.csv` key, new Task 2 and Task 5 data) |
| V18 / V20 | 0.81835 / 0.81973 | 224x448 input, strong augmentation, multi-scale TTA; EM class-prior step |
| V22c | 0.82473 | big backbones (ConvNeXt-B 22k, DINOv2-B), weights 3 |
| V24a | 0.82772 | augmentation bug fixed (section 5) |
| V26a / V27c | 0.83096 / 0.83310 | all members retrained with fresh augmentation |
| V32a | 0.83558 | DINOv2-Large added |
| V36a | 0.83735 | four DINOv2-L seeds |
| V45a | 0.83820 | ten DINOv2-L seeds (the plain-seed curve is flat from here) |
| V50b | 0.83960 | **new fine-tuning recipe** (layer-wise lr decay, EMA, CutMix/MixUp) |
| V53a / V53c | 0.84106 / 0.84113 | four new-recipe models; plus a DINOv2-B member |
| **V57a** | **0.84167** | six new-recipe models, improved Task 3 blend, two extra Task 5 flags |

---

## 1. Common setup, rules and disclosures

### 1.1 Setup
- Data: only the provided Kaggle datasets (the host's corrected release of 2026-11-12). Seed 342; 5-fold stratified cross-validation for Tasks 1-3 (out-of-fold predictions, "OOF", are used for thresholds and blends). Train and validation data were combined for training.
- Tasks 1, 2, 3 and 5 were trained from scratch (no pretrained weights, no external data). Task 4 uses pretrained backbones, as the rules allow. No AutoML; hyperparameters were tuned by hand and by simple searches.
- Hardware: one laptop GPU (RTX 4060, 8 GB). This limited Task 4 to batch size 8 with gradient checkpointing for the large models, about 8 hours per DINOv2-Large run.
- Pretrained backbones (all from public hubs): DINOv2 ViT-L/14 and ViT-B/14 (`lvd142m`), ConvNeXt-Base (ImageNet-22k), ConvNeXt-Small/Tiny (ImageNet), a CLIP-pretrained ConvNeXt-B (tested and dropped). Licences **[VERIFY each model card]**.

### 1.2 Notes on data use and tooling
1. **Class-prior (EM) adjustment in Task 4.** The class mix of the test images differs from the training mix. After the ensemble has predicted the test images, the test class shares are estimated from the predicted probabilities with the Saerens EM algorithm (no labels involved) and the probabilities are multiplied by the estimated shares (alpha 1; a label-shift correction). No test label, pseudo-label or fine-tuning is used. It is worth +0.0017 to +0.0019 on the public leaderboard (same members with and without: V31a 0.83419 vs V31c 0.83248; V41a 0.83782 vs V41b 0.83615; V57a 0.84167 vs V57b 0.83980). Both final submissions (V57a, V53c) include it.
2. **Task 5 has no training file**, only `test.csv`, so all Task 5 analysis (gap discovery, the missing-value classifier) is unsupervised and fitted on the Task 5 test features. No labels are involved, but it is test-set analysis by nature.
3. **Leaderboard feedback was used as information, not only for scoring.** We used submissions to find out (a) that the Task 1 metric is the plain positive-class F2 and not macro F2, (b) that Task 5 is plain F3 on class 1, (c) that Task 4's file format was wrong (section 5), and (d) one diagnostic upload with Task 5 cut to a single flagged row, from which the number of true anomalies was inferred (section 6). No test labels were ever seen.
4. **Test-image statistics motivated augmentations.** In round 2 we compared brightness and colour statistics of test and train images (test is darker and more saturated), which motivated the colour and brightness augmentations in Task 4. This is a design decision informed by unlabeled test images.
5. **Kaggle API use.** Most submissions were uploaded by team members through the website. Six files were submitted through the Kaggle command line by the AI coding assistant used on this branch (V40a, V42a, V43a, V45a, V51a, V55a), each only after the branch owner gave explicit permission for exactly that file in that time window. **[TEAM: keep or reword.]**
6. **AI assistance.** The code on branch `runs/wirachat` was written and run with an AI coding assistant (Claude Code) under the branch owner's direction. **[TEAM: state the course policy on AI tools and the owner's own contribution.]**
7. **Shared daily quota.** The Kaggle limit of about five submissions per day is shared by the team; one teammate's upload on 2026-11-21 used the slot we had planned for a no-EM test.

### 1.3 Task weights from the leaderboard
The Task 1 threshold change (0.40 -> 0.18, plain F2) raised the public score by +0.0173, exactly the +0.098 predicted for Task 1 divided by five. Cutting Task 5 to one flagged row (diagnostic V16) lowered the score from 0.81742 to 0.64078, a drop of 0.17664, consistent with the F3 of 0.883 for 1,806 flags divided by five. Both agree with equal weights; this is an inference from scores, not a statement from the course **[VERIFY]**.

---

## 2. Task 1: Anti-cheat pre-filter (binary, F2)

**Business problem.** About 10,000 cheat reports per day arrive, the review team handles 500, and the current system has a 70% false-positive rate (slide 14). The model should rank reports so that the likely cheaters are reviewed first.

**EDA.** About 104k labelled rows and 32 predictive columns (kill/death ratio, headshot percentage, win rate, accuracy, reaction time, reports received, ...). About 13% of the training labels are missing and those rows were dropped. Test has a different per-column missingness pattern from train (for example headshot_percentage 2.7% missing in train, 14.1% in test; adversarial AUC 0.90) but is otherwise identical (KS about 0.01 per column, same correlation structure). Accounts with `reports_received` above 27.6 are never cheaters (1,011 rows).

**Model.** Two LightGBM binary models averaged: one on the raw columns, one on raw plus five engineered features (number of missing values, reports per account age, kill/death x headshot, total aim score, reaction x consistency). Parameters: learning rate 0.03, 31 leaves, min child samples 40, feature fraction 0.7, bagging 0.8, L2 5, early stopping 150 rounds (700-1,300 trees per fold). The test probability is the mean of the five fold models, then the mean of the two variants.

**Results.** OOF AUC 0.8911 (raw), 0.8906 (engineered); the OOF-optimal threshold for plain F2 is 0.18, giving F2 **0.837**. At 0.18 the model flags 60.3% of the test accounts.

**Checks.**
- XGBoost/CatBoost blends gave no gain (0.8370 vs 0.8373); a GPU MLP was clearly worse (AUC 0.8145 vs 0.8914); logistic regression 0.765.
- Four-leaf boosted trees already reach AUC 0.8899, so the task is saturated.
- Simulating the test missingness on validation folds costs only 0.0009 AUC, so the OOF score is credible.

**Metric insight and common mistake.** The Kaggle page says "F2 (Macro)" but the score behaves like plain positive-class F2. Round 1 tuned a macro-F2 threshold (0.40); switching to the plain-F2 threshold (0.18) raised the score by +0.0173. F2 weights recall four times as much as precision, which pushes the best threshold far below 0.5.

**Business meaning.** F2 rewards catching cheaters over avoiding false alarms. Flagging 60% of accounts maximises F2 but is far above the 500-case daily review capacity, so in production the probability should be used to **rank** the queue (review the top 500) and the 0.18 cut-off only to decide what must not be auto-dismissed. This is the practical meaning of "pre-filter".

Code: `runs/wirachat/round-1/task1_cv.py` (run with and without `--fe`), threshold step in `round-2/build_round2_variants.py`.

---

## 3. Task 2: Player segment classification (4 classes, macro F1)

**Business problem.** Generic campaigns reach only 2-5% engagement; whales (top 5% of players) give 60% of revenue (slide 16). A segment label (Casual, Grinder, Social, Whale) enables targeted campaigns.

**EDA.** The host replaced the Task 2 data on 2026-11-12 (new ids, same schema and label mix), so everything was redone. Only `loyalty_tier` and `engagement_level` are categorical; missingness carries no signal; there are no gapped sub-populations. Several `random_metric*` columns are pure noise and were dropped. Test and train are statistically identical (adversarial AUC 0.50).

**Model.** LightGBM multiclass, 5-fold, categorical columns as pandas categories, class weights multiplied onto the probabilities and tuned by coordinate search on the OOF predictions (same parameters as Task 1).

**Results.** OOF macro F1 **0.7512** (argmax) and **0.7524** with the tuned class weights. Logistic regression reaches 0.7506, so the signal is essentially linear and about 0.75 is the ceiling; a GPU MLP was worse (0.7253). Predicted test shares [0.416, 0.249, 0.195, 0.14].

**Insight.** When logistic regression equals gradient boosting, extra model complexity cannot help; the remaining effort is better spent on other tasks.

**Business meaning.** The segments allow different messages (tournaments for grinders, cosmetics for whales, event invitations for social players) instead of one generic campaign.

Code: `round-3/t2_new.py` (same method as `round-1/task2_cv.py`).

---

## 4. Task 3: Player monthly spending (regression, NMAE)

**Business problem.** Finance misses quarterly targets by 35-40% and VIP support cannot tell who deserves priority (slide 18). The model predicts spending in the next 30 days.

**EDA.** 48% of the targets are zero (zero rate by segment 72/48/31/7%) and the top 1% of rows hold 19% of the total spend. `historical_spending = total_transactions x avg_transaction_value` holds exactly, and the player `segment` can be recovered with 92% accuracy for rows where it is missing. Log-residuals of the positive part follow a 98%/2% mixture; whales (segment 3) cause 79% of the absolute error. The positive amount is almost deterministic (log-space R2 0.99), while whether the spend is zero is mostly random (AUC 0.78 overall, 0.56-0.63 within a segment).

**Metric.** `NMAE = 1 / (1 + sum|y - y_hat| / sum y)` is an absolute-error metric on the raw scale, so the best prediction is the **conditional median**, not the mean. Optimising mean squared error (or predicting the mean of a log-model) is a common scaling mistake the slide warns about.

**Model.**
1. Two-stage model (hint of slide 19): LightGBM P(spend = 0), LightGBM on log-spend for the positive rows, a per-segment lognormal sigma, combined into the conditional median (predict 0 if P(zero) >= 0.5).
2. The same two-stage model on the exact-identity features of model 3 (`t3_twostage_fe.py`).
3. L1 (absolute error) LightGBM with the identity features (`historical_spending` rebuilt from its factors, recovered segment).
4. Final: `0.55 x two-stage(FE) + 0.25 x two-stage(raw) + 0.20 x L1 identity`, clipped to [0, 300000].

**Results (5-fold OOF NMAE score).** Median of six L1 LightGBMs 0.7854; two-stage (raw) 0.7884; L1 identity 0.7863; two-stage with identity features 0.7900; **final blend 0.7907** (+0.0011 over the previous blend of 0.7896; worth about +0.0002 on the leaderboard, not measured separately). A finer blend search of the old members (0.78974), scaling the predictions and changing the clip gave nothing. A GPU MLP was worse (log-residual sd 0.242 vs 0.225).

**Insight.** Knowing the zero outcome exactly would add only about +0.008 on the leaderboard, so the target is near its ceiling given the information in the features.

**Business meaning.** Predicted spending ranks players for VIP treatment and discounts. Because whales cause most of the error, the model should be monitored on that segment first.

Code: `round-1/task3_cv.py`, `task3_tune.py`; `round-2/t3_twostage.py`, `t3_fe2.py`; `round-3/t3_twostage_fe.py`, `build_round3_t3t5_update.py`.

---

## 5. Task 4: Game title detection (5 classes, macro F1)

**Business problem.** Support receives 2,000 image tickets daily and spends 10+ minutes finding which game is shown; contests and moderation cannot be auto-tagged (slide 20).

**Data.** 230x120 screenshots from 7 Knights (class 0), FiveN, Free Fried, PUBE, Roblock; train class shares 0.39 / 0.14 / 0.18 / 0.19 / 0.10. The test images come from different sources than train: only 0.08% of test images have a train neighbour above 0.95 cosine similarity (2% within train), and they are darker and more saturated. DINOv2 nearest-neighbour cosine, test to train, has a median of 0.804 against 0.840 within train.

### 5.1 The file-format problem (common mistake)
Rounds 1-2 scored about 0.2 on Task 4 (chance level) for the first ten days, whatever model we used. On 2026-11-12 the hosts replaced `test.csv` with a version keyed by the answer ids; the old file listed image ids in a different order from the answer key. All our encodings, orderings and model changes had been correct but could not move the score. Lessons: map predictions by id, never by row position; verify the answer-key format before chasing model quality; and a high held-out score (0.97) cannot detect a key mismatch.

### 5.2 Validation
Held-out clusters: 300 k-means clusters of DINOv2-small embeddings, 60 held out (6,000 images). This gives macro F1 0.965-0.98 but does not reflect the test-source shift, so the leaderboard, not the held-out score, tracked real improvements. We used the held-out split only to compare training recipes (section 5.4), where the changes were small and consistent.

### 5.3 Model and training (baseline recipe)
End-to-end fine-tuning of pretrained backbones on all training images at 224x448 input (the images are wide): AdamW, one-cycle learning rate, label smoothing 0.1, class-balanced loss, fp16, 8-12 epochs, strong class-independent augmentation (random resized crop, flip, rotation to 45 degrees, colour jitter, channel shuffle, grayscale, equalize/autocontrast/posterize/solarize, blur, down-up resampling, noise, random erasing). Test-time augmentation over three scales (0.8, 1.0, 1.25) and horizontal flip. Backbones tried: ConvNeXt-Tiny/Small/Base, EfficientNet-V2-S (stopped: 5.6x slower), DINOv2-B and DINOv2-L, CLIP ConvNeXt-B.

### 5.4 Findings in the order they paid off
1. **Bigger and more diverse backbones** (V21a 0.82273, weights x2 V21b 0.82411, x3 V22c 0.82473) helped far more than more copies of small models (+0.0003-0.0005 each).
2. **Augmentation bug.** Until 2026-11-15 the dataset class re-seeded the random generators from the image index, so every image received the same augmentation in every epoch (verified: 16 of 16 images identical across two epochs). With `--fresh-aug` a new augmentation is drawn each epoch (0 of 16 identical). Replacing one model gave +0.0030 (V24a 0.82772), the fresh DINOv2-B another +0.0023 (V26a 0.83096), and the old fixed-augmentation members turned out to be dead weight (V27a, dropping them, +0.0013).
3. **Equal-weight strong members.** Four strong members at equal weight (ConvNeXt-B at 224x448, hi-res ConvNeXt-B, two DINOv2-B seeds) beat any single addition (V31a 0.83419). A weaker, different architecture hurt (CLIP ConvNeXt-B, V28b 0.83235 vs V27c 0.83310); resolution 288x576 was flat (V30b).
4. **DINOv2-Large** (304M parameters, gradient checkpointing): V32a 0.83558. Ten plain seeds raised the score step by step (V36a 0.83735 with four seeds, V43a 0.83803 with eight, V45a 0.83820 with ten); per-seed gains shrank to +0.0001 and the curve was flat at ten to eleven seeds (V46a 0.83816). A 12-epoch variant (V37a) and weight tweaks were flat or negative.
5. **New fine-tuning recipe** (layer-wise learning-rate decay 0.9 with top lr 6e-5, exponential moving average of the weights 0.999, CutMix/MixUp on half of the batches). On a DINOv2-B held-out test it was ahead at every check (final macro F1 0.9825 vs 0.9760; colour-stress 0.9744 vs 0.9686). One DINOv2-L trained with it, weighted like seven plain seeds, moved the public score by +0.0014 (V50b 0.83960), the largest step since V36a, although its test-time behaviour (agreement with the other models 91.7-92.6%) looked like another seed. More new-recipe models kept helping at a fixed weight share of about two thirds (two models 0.84073, four models 0.84106); a 75% share was lower (0.84060). A DINOv2-B member trained with it added +0.00007 (V53c 0.84113); the fifth L model plateaued (V55a 0.84109, within the public noise of about 0.0002).
6. **Class-prior (EM) adjustment** (section 1.2): +0.0017 to +0.0019, optimum at exponent 1 (alpha 2, and shrinking the shares half-way toward uniform, were worse).

### 5.5 Final ensemble (V57a)
Weighted average of test-time-augmented class probabilities, weights as integers summing to 316:
| Members | Weight each | Total weight | Share |
|---|---|---|---|
| six DINOv2-L, new recipe (seeds 201-206), all training images | 33 | 198 | 62.7% |
| DINOv2-B, new recipe (seed 301), all training images | 20 | 20 | 6.3% |
| ten DINOv2-L, baseline recipe, fresh augmentation | 7 | 70 | 22.2% |
| ConvNeXt-B 22k (224x448, seed 5, hi-res 288x576) and DINOv2-B (three seeds), fresh augmentation: six members | 4 | 24 | 7.6% |
| ConvNeXt-Small and ConvNeXt-Tiny, fresh augmentation | 2 | 4 | 1.3% |

The new-recipe members hold 218/316 = **69%** of the weight. The probabilities are averaged, then multiplied by the class shares estimated by EM on the ensemble itself (alpha 1); the predicted test shares after the step are about [0.135, 0.289, 0.139, 0.249, 0.187].

**Results.** The Task 4 score is not measured separately; **[ESTIMATE]** about 0.94 from the total. Gains per added model were +0.0003-0.0005 before the augmentation fix and +0.003 for the first fresh big models.

**Business meaning.** Reliable game identification routes tickets to the right specialist without manual work and allows verification of the screenshot campaign. The shift between sources (streams, tickets, forums) is the real risk in production: new image sources need monitoring of the predicted class mix, which is exactly what the EM step estimates.

Code: `round-3/t4_finetune2.py` (training and prediction), `run_t4_v*.sh` (exact commands per run), `t4_prior_adjust.py` (EM step), `build_round3_t4_variant.py` (ensemble builder), `build_final_v57.sh` (final build).

---

## 6. Task 5: Account security monitoring (unsupervised, plain F3)

**Business problem.** More than 5,000 confirmed account thefts a month, bot farms and account-selling rings; the rule-based system has a 60% false-positive rate (slide 22).

**Data.** 25,889 players x 122 features: four time periods x 30 behaviour columns plus statics (`behavior_stability`, `password_changed`, `vpn_usage`, `suspicious_login_time`, `mass_item_sale`). No labels exist; only `test.csv` is given. The host replaced the dataset on 2026-11-12 (same ids, new order, 99.8% of the values changed), so the analysis was redone.

**Metric insight.** The score behaves like plain F3 on class 1: `F3 = 10 TP / (9T + n)` with T true anomalies and n flagged rows. Adding a flag helps if and only if its probability of being an anomaly exceeds F3/10 (about 9%). So a flagged-row budget should be spent liberally on rows with even a modest probability, unlike in an F1 problem. From the diagnostic upload with a single flagged row, the F3 for our 1,806 flags is 0.883, which implies about 2,070 true anomalies (8.0% of the rows) at about 100% precision, or about 2,000 at 97%.

**Method.** Normal columns are uniform latent values plus per-period noise (kurtosis about -1.2). Anomalies appear as structured groups separated from the normal rows by empty gaps in the feature space:
- **Group A (825 rows):** large change of period 4 against periods 1-3. The score is the sum of capped squared robust z-scores of the change over all base features; sorted scores show a clean gap between about 150 and 221. These rows show account-takeover patterns (location jumps, currency transfers, purchases; password change 40%, mass item sale 28%) and skill boosting or selling.
- **Five persistent groups** (same rows in all four periods, each separated by an empty gap): low `behavior_stability` 246; low `ip_hash_entropy` + `login_time_variance` + `session_duration_std` 207 (bot-like, same rows in all three columns); low `communication_frequency` 263; low `activity_level` 265. No overlaps between groups. Total 1,806 rows (6.98%).
- **Missing-value flags (+49).** 4.9% of rows have `behavior_stability` missing, invisible to the gap rule. The low-stability group has side effects (purchase frequency +1.3 sd, logins -1.1 sd, older accounts +0.8 sd), so a gradient-boosting classifier that recognises the low-stability group from its side effects (OOF AUC 0.982) scores the missing rows; the 49 rows above 0.09 are flagged. Leaderboard effect +0.00095 (Task 5 +0.0048, about 14 true anomalies where 8 were expected).
- **Two tail rows (+2).** Two unflagged rows jump 34 and 57 degrees in login location and have onset scores (periods 3-4 vs 1-2) of 243 and 237, far above the next-largest 188; both have 13-14 missing values, which is why the period-4 rule missed them. Final flags: 1,857 (7.17%). Together with the improved Task 3 blend and a slightly different Task 4 ensemble this took V53c (0.84113) to V57a (0.84167, +0.00054); the parts were not scored separately.

**What did not work.** Isolation Forest (baseline level, smooth tails, finds only 6% of the activity-level group), PCA reconstruction, LOF, Gaussian mixtures (BIC picks eight components, only the known groups), gap scans on period means, slopes, within-row variance, location jumps and residual correlations, duplicates, discrete-value and NaN-pattern checks. About 200 anomalies (T about 2,070 vs 1,857 flagged) probably remain unflagged; this is the largest known loss in any task.

**Business meaning.** The groups are distinct fraud and abuse patterns (account takeover, low-entropy bot farms, skill boosting) that map to different review queues, which a single anomaly score would hide.

Code: `round-1/task5_*.py`, `round-2/task5_*.py` (history), `round-3/t5_new.py`, `t5_new2.py`, `t5_nanstab.py`, `build_round3_t5_nanstab.py`, `build_round3_t3t5_update.py` (tail rows).

---

## 7. Insights and common mistakes

1. **Read the metric from the leaderboard's behaviour, not the page label.** Task 1's "F2 (Macro)" is plain F2, Task 5 is plain F3; the wrong threshold cost 0.0173.
2. **Verify the submission format end to end.** The Task 4 key mismatch kept Task 4 at chance level for the first ten days; no modelling could have fixed it.
3. **A high held-out score can hide a source shift.** Held-out clusters said 0.97 while the test images came from other sources; judge changes by the leaderboard with care and use the held-out set only for controlled comparisons.
4. **Check data-loading code for silent bugs.** Fixed per-image augmentation looked like normal training but cost about +0.003 per big model.
5. **Ensembles help only when members are individually good.** Dropping the old weak members improved the score; a weak different architecture (CLIP ConvNeXt-B) hurt; seeds of one strong model helped, but with diminishing returns.
6. **A better training recipe beat more seeds.** Ten plain seeds saturated at 0.8382; one new-recipe model moved the score by +0.0014.
7. **Use the metric's algebra.** F-beta formulas decide thresholds (Task 1), which rows to flag (Task 5) and what to predict (the median for NMAE, Task 3).
8. **Public leaderboard noise is about 0.0002.** Many late "improvements" were inside it; we stopped at the plateau instead of spending submissions.
9. **Check for a shift in the class mix.** The test images contain more of some games than the training set (predicted shares differ from the training shares); correcting for it with the EM class-prior adjustment is worth +0.0018 and costs no extra training.

---

## 8. Team collaboration **[TEAM]**

Fill in: member names and roles, which tasks each member led, how results were shared, meetings, and what each member learned. Facts from the repository that can be used:
- Everyone works on a branch `runs/<name>` of the shared repository (rule: no pushes to `main` or another member's branch; code and notes only, no submission files).
- The daily Kaggle quota of about five submissions was shared across the team and was a real constraint (a teammate's upload used the slot planned for a test on 2026-11-21).
- This branch (`runs/wirachat`) holds the work described in this report, in three rounds (`round-1`: EDA and baselines, `round-2`: Task 3/5 methods and early Task 4, `round-3`: corrected data, Task 4 fine-tuning, final builds).

| Member | Role / tasks | Main contribution | Takeaway |
|---|---|---|---|
| [TEAM] | | | |

---

## 9. Conclusion

**Lessons learned.** Data and format correctness came first (the Task 4 key), then metric understanding (F2/F3 thresholds), then model work. After that, training quality (augmentation fix, new fine-tuning recipe) mattered more than ensembling more of the same model.

**ML insights.** Linear models matched boosting on Task 2, so the ceiling was the data, not the model; an unsupervised problem (Task 5) was solved by looking for structure (gaps) instead of a generic detector; the best use of a metric is to derive from its formula what to optimise.

**Limits.** Our final public score is 0.84167 against 0.87346 for the leading team; the 0.03 gap equals about 0.16 on the sum of five task scores. We could not locate it: Tasks 1-3 are at their train-CV ceilings, Task 4 plateaued, and about 200 Task 5 anomalies remain unflagged without any signal we could find. Public scores are noisy, and the private ranking may differ.

**Future improvements.** (1) Search for the unflagged Task 5 anomalies with a labelled proxy (simulate the six known group types and test detectors). (2) A held-out validation set that mimics the test-source shift for Task 4 (for example, a split by image source). (3) More new-recipe seeds at about two thirds of the weight, if GPU time allows; the gain per seed is now below the leaderboard noise. (4) Estimate the test class mix early (the EM class-prior adjustment) and train with that mix in mind, instead of correcting only at the end.

---

## Appendix A: Reproducibility

- Code lives in `runs/wirachat/round-1/` (EDA, Tasks 1-3, Task 5 first rounds), `round-2/` (Task 3 and Task 4 methods, Task 5 groups), `round-3/` (corrected data, Task 4 fine-tuning, final builds). A task-by-task index is in `runs/wirachat/final/README.md`. Each round's `notes.md` records every experiment with its Kaggle score.
- Data: the host's corrected release (`CPE342-DATA-v2`) is read through the environment variable `CPE342_DATA` (`round-1/r1_common.py`); the dataset tracked in the repository is the older release.
- Final files: `build_final_v57.sh` is the exact command sequence for V57a/V57b (Task 3 two-stage model -> `build_round3_t3t5_update.py` -> EM step `t4_prior_adjust.py` -> `build_round3_t4_variant.py` -> `build_round3_t5_nanstab.py`). V53c is built by the commands in the Task 4 section of `round-3/notes.md` (`build_best.sh` in the local folder `best/`).
- Model weights, probability arrays and submission files are not committed (size, and the repository rules).
- Software: Python 3.11, LightGBM, scikit-learn, PyTorch, timm, Windows 11, RTX 4060 Laptop GPU (8 GB). Seeds: 342 (CV), per-run seeds in the `run_t4_v*.sh` files.
- Not re-run end to end after the final cleanup; the scripts resolve their paths relative to their round folder. **[VERIFY by running one task from a clean checkout.]**

## Appendix B: Final submission files
| File | Public score | Task 4 | Task 3 / Task 5 |
|---|---|---|---|
| V57a `submission_v57a_new6x33_ball20_t3fe_em.csv` (**selected**) | **0.84167** | six new-recipe L + B, EM | improved blend; 1,857 flags |
| V53c `submission_v53c_new4x49_bnew20_em.csv` (**selected**) | 0.84113 | four new-recipe L + B (80% data), EM | earlier blend; 1,855 flags |
| V57b `submission_v57b_new6x33_ball20_t3fe_clean.csv` (not selected) | 0.83980 | same as V57a, no EM | same as V57a |
