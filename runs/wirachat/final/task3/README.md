# Task 3: monthly spending (regression, NMAE)
`task3.ipynb`: recovers the missing segment, builds exact-identity features (`historical_spending = total_transactions x avg_transaction_value`), trains three LightGBM models (two-stage lognormal median on raw features, L1 on identity features, two-stage on identity features) and blends them 0.25 / 0.20 / 0.55. Writes `../outputs/task3_pred.npy` and the members (used for the second selected file V53c).
OOF NMAE score: 0.7884 / 0.7863 / 0.7900, blend 0.7907. Run: `jupyter nbconvert --execute --inplace task3.ipynb` (needs `CPE342_DATA`).
