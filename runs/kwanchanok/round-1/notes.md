# Task 1 Improvement Round 1

## Configuration

- Model: `HistGradientBoostingClassifier`
- Preprocessing: baseline numeric median imputation with missing indicators
- Validation split: 80/20 stratified `train_test_split`
- Random seed: `342`
- Model parameters: `max_iter=120`, `learning_rate=0.10`, `max_leaf_nodes=31`, `l2_regularization=1.0`
- Threshold grid: `0.05` through `0.95` in increments of `0.005`
- Selection objective: `fbeta_score(..., beta=2, average="macro")`
- Baseline threshold: `0.17`

## Validation results

- Selected threshold: `0.39`
- Macro F2: `0.7785650289`
- Binary F2 for cheater class: `0.7451539688`
- Precision: `0.6726190476`
- Recall: `0.7657998424`
- Confusion matrix: `[[9454, 2365], [1486, 4859]]`
- Predicted positive rate: `0.3977097556`

Baseline threshold comparison on the same validation split:

- Threshold `0.17` Macro F2: `0.7038981809`
- Threshold `0.17` binary cheater F2: `0.8355668722`
- Threshold `0.17` predicted positive rate: `0.6567936578`

The macro-F2 objective selects a different threshold than the prior binary-cheater-F2 objective. These results are validation-only; no test labels or test outcomes were used.

## Outputs

- Task 1 predictions: `outputs/predictions/task1.csv`
- Round submission: `outputs/submission.csv`
- Metrics: `outputs/metrics/task1.json`
- Kaggle score: 0.49294

## Commands

```bash
OMP_NUM_THREADS=4 .venv/bin/python runs/kwanchanok/round-1/train_task1.py
python runs/kwanchanok/round-1/build_submission.py
```
