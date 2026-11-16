# Improvement Round 4

## Official metrics

The project instruction PDF specifies binary F2 for Task 1, F1 for the four-class Task 2, NMAE `1 / (1 + sum(abs(y - prediction)) / sum(y))` for Task 3, Macro F1 for Task 4, and F3 for Task 5. The PDF does not state an averaging method for Task 2, so this round uses Macro F1 as required by the experiment brief. Task 5 supplies no labels and therefore has no local F3 validation score.

## Validation and leakage controls

Tasks 1 and 2 use the same five stratified folds for every candidate. Task 3 uses the same five shuffled KFold splits for every candidate. Every preprocessing pipeline is fit only on each fold's training rows, and OOF predictions are made only for held-out rows. Threshold selection for Task 1 uses complete OOF probabilities. No supervised choice uses test data or Kaggle feedback. Final supervised predictions use five fold-model probability/prediction averages.

CatBoost was not installed in the project environment. No dependency was installed; the experiment continued with HistGradientBoosting and ExtraTrees candidates.

## Results and selection

- Task 1: official binary F2. Candidates were raw/engineered HistGradientBoosting and raw/engineered balanced ExtraTrees. Conservative engineered features were missing count/rate, three non-negative log1p transforms, and two domain-motivated ratios. Raw HGB OOF F2 was 0.835337 (fold mean 0.835335 +/- 0.002295). Engineered HGB was best at 0.835677 (0.835676 +/- 0.003040), but the gain was only 0.000340. Although it won/tied 3/5 folds, it did not meet the explicit non-negligible gain of 0.001. The root baseline prediction was retained; changed rows: 0. OOF thresholds 0.17 and 0.38 were evaluated but not automatically preferred.
- Task 2: official multiclass F1, evaluated as Macro F1. HGB achieved OOF Macro F1 0.744579 (0.744574 +/- 0.002011); balanced ExtraTrees achieved 0.700562. HGB remained the best candidate, so the existing root prediction was retained; changed rows: 0.
- Task 3: official NMAE. Candidates were HGB absolute-error baseline, HGB squared-error, ExtraTrees, and log1p-target HGB/ExtraTrees. The absolute-error baseline was best at OOF NMAE 0.776204 (0.776224 +/- 0.003838). The root prediction was retained; changed rows: 0. Log-target predictions were transformed back with expm1 and all candidates were clipped only below zero.
- Task 5: official hidden-label F3, with no labeled local validation. Isolation Forest with median imputation, robust-preprocessed Isolation Forest, and transductive LOF were compared across contamination 0.03/0.05/0.07 and seeds 342/343/344 where randomness applies. Baseline and robust Isolation Forest had identical mean seed Jaccard 0.234773; LOF agreed weakly with the baseline (Jaccard 0.148050 at contamination 0.05). The robust version was not demonstrably more stable, so the root prediction was retained; changed rows: 0. LOF used `negative_outlier_factor_` because novelty scoring on the fitted rows would be invalid.
- Task 4 was not retrained and its root prediction values and ID order were preserved exactly.

All four Round 4 task predictions equal their root baseline predictions because no new candidate passed its selection rule. Consequently, all five generated submissions contain the same prediction values as the existing baseline composition.

## Submission recommendation

The designated order is:

1. `combined.csv` first.
2. If it improves the current best score, use task-only submissions only to isolate the likely contributing task.
3. If it does not improve, do not submit all files blindly; use validation evidence to choose at most one or two task-only files.

For this run, no Kaggle upload is recommended because `combined.csv` and every task-only file are prediction-equivalent to the root baseline. If an upload is nevertheless required for workflow verification, use `combined.csv` first.
