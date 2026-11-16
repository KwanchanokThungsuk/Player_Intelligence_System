# Task 1 Improvement Round 2: Feature Engineering

- Branch: `runs/kwanchanok`
- Seed: `342`
- Model and hyperparameters: same as Round 1 (`HistGradientBoostingClassifier`, `max_iter=120`, `learning_rate=0.10`, `max_leaf_nodes=31`, `l2_regularization=1.0`)
- Split: same 80/20 stratified split and threshold grid as Round 1
- Raw features: 31
- Added features: 7
- Temporal groups: none; Task 1 has no numeric repeated-period groups matching `base_1` to `base_4`
- Log features: `reports_received`, `headshot_percentage`, `accuracy_score`
- Interaction features: `accuracy_score_over_reaction_time_ms`, `headshot_percentage_over_accuracy_score`

## Validation comparison

| Metric | Round 1 reference | Round 2 engineered |
|---|---:|---:|
| Threshold | 0.39 | 0.40 |
| Macro F2 | 0.7785650289 | 0.7812101672 |
| Binary F2 (cheater) | 0.7451539688 | 0.7364329079 |
| Precision | 0.6726190476 | 0.6887311703 |
| Recall | 0.7657998424 | 0.7494089835 |
| Predicted positive rate | 0.3977097556 | 0.3800924906 |

Reference confusion matrix: `[[9454, 2365], [1486, 4859]]`

Engineered confusion matrix: `[[9670, 2149], [1590, 4755]]`

Macro F2 improvement: `0.0026451383`

Because engineered Macro F2 is higher, the final engineered pipeline was used for the round-2 test prediction. No test labels or test-based threshold selection were used.

## Outputs and commands

- Metrics: `outputs/metrics/task1.json`
- Task 1 prediction: `outputs/predictions/task1.csv`
- Round submission: `outputs/submission.csv`

```bash
OMP_NUM_THREADS=4 .venv/bin/python runs/kwanchanok/round-2/task1/train.py
python3 runs/kwanchanok/round-2/build_submission.py
```

Kaggle score: ____________________
