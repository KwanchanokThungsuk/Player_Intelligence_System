# Task 1 improvement Round 3

Round 3 uses stratified five-fold out-of-fold (OOF) validation because Round 2 used one split and its small Macro F2 gain needs a stability check. Raw baseline and Round 2 engineered features use exactly the same folds, seed, model family, hyperparameters, and threshold grid. Preprocessing and models are fit inside each fold only on that fold's training rows; test rows are never used for feature or threshold decisions.

The baseline uses the original Task 1 numeric features. The engineered candidate adds row-local missing count/rate, log1p features for the non-negative observed columns `reports_received`, `headshot_percentage`, and `accuracy_score`, and two justified ratios: accuracy/reaction time and headshot percentage/accuracy. No temporal groups were present in the actual schema.

Thresholds maximize Macro F2 on complete OOF probabilities. Ties use the threshold closest to 0.5, then the lower threshold. The engineered candidate is selected only when its OOF Macro F2 is at least 0.005 above baseline and it wins at least three folds; otherwise the baseline is selected. The test output is a five-model fold ensemble, not a full-data model.

Comparison with Round 1 and Round 2, exact metrics, changed prediction counts, and the selection decision are recorded in `outputs/metrics/task1.json`. Kaggle recommendation: fill in after reviewing the recorded OOF gain and fold standard deviation; do not submit solely for a marginal gain.

For context, Round 1's single-split Macro F2 was 0.778565 and Round 2's single-split engineered Macro F2 was 0.781210. Round 3 recomputed its own five-fold reference at OOF Macro F2 0.776954 and found the engineered candidate at 0.777259, an improvement of only 0.000305 with wins in 2/5 folds. These validation designs are not identical, so the five-fold result is the stability check used for selection.

Kaggle score: ## Kaggle result

- Root baseline score: 0.50879
- Round 1 score: 0.49294
- Round 3 score: 0.49545
- Improvement over Round 1: +0.00251
- Difference from root baseline: -0.01334

The 5-fold OOF baseline improved upon Round 1, but it did not
outperform the original root baseline. The engineered features were
not selected because their OOF Macro F2 gain was only 0.000305 and
they outperformed the baseline in only 2 of 5 folds.

Round 3 is retained as an experiment and is not promoted as the
current best submission.
