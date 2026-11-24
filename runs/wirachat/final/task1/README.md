# Task 1: anti-cheat pre-filter (binary, F2)
`task1.ipynb`: two LightGBM models (raw columns; raw + 5 engineered features), 5-fold CV, test probability = mean of the fold models and of the two variants; threshold 0.18 for the plain positive-class F2. Writes `../outputs/task1_pred.npy`.
OOF AUC 0.891, OOF F2 0.837. Run: `jupyter nbconvert --execute --inplace task1.ipynb` (needs `CPE342_DATA`, see `../README.md`).
