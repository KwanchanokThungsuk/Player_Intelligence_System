# Task 2: player segment (4 classes, macro F1)
`task2.ipynb`: LightGBM multiclass on the corrected data, noise columns (`random_metric*`) dropped, 5-fold CV, class weights tuned on the OOF probabilities (final weights [0.85, 0.9, 1.0, 1.0]). Writes `../outputs/task2_pred.npy`.
OOF macro F1 0.7512 (argmax), 0.7524 (weighted). Run: `jupyter nbconvert --execute --inplace task2.ipynb` (needs `CPE342_DATA`).
