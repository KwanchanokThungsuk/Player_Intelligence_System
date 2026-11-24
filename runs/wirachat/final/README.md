# CPE342 Karena Player Intelligence System: final source code

This folder holds **only the final code that produced the two files selected for the Kaggle ranking**: V57a (public score **0.84167**) and V53c (0.84113). The trial-and-error work (other models, diagnostics, weight sweeps) is kept apart in `../round-1`, `../round-2` and `../round-3` and is not needed to reproduce the submission. Technical report: `../report/`.

```
final/
  README.md              this file
  requirements.txt       packages (Python 3.11)
  run_all.py             runs all notebooks in order
  task1/task1.ipynb      anti-cheat pre-filter      LightGBM raw + engineered, threshold 0.18 for F2
  task2/task2.ipynb      player segment             LightGBM multiclass, tuned class weights
  task3/task3.ipynb      monthly spending           two-stage lognormal-median + L1 LightGBM blend
  task4/                 game title detection       train_backbone.py + train_all_members.sh (training), task4_ensemble.ipynb (ensemble + EM class prior)
  task5/task5.ipynb      account anomaly detection  unsupervised gap groups + missing-value classifier
  make_submission/make_submission.ipynb   combines the per-task outputs into submission_v57a.csv and submission_v53c.csv
  outputs/               written by the notebooks (not committed)
```

## How to run
1. `pip install -r requirements.txt` (Task 4 training also needs an NVIDIA GPU and the CUDA build of PyTorch).
2. Download the competition data from Kaggle (the corrected release of 2026-11-12) and point `CPE342_DATA` at the folder that contains `01_binary_classification/`, ..., `05_anomaly_detection/` and `sample_submission.csv`:
   ```bash
   export CPE342_DATA=/path/to/cpe342-kaggle-main            # Windows PowerShell: $env:CPE342_DATA = "C:/path/to/cpe342-kaggle-main"
   ```
   (The dataset tracked in this repository is the older release, do not use it for Tasks 2 and 5.)
3. Run the notebooks, either from Jupyter (open each notebook, Run All; the working directory must be the notebook's own folder) or all at once:
   ```bash
   python run_all.py                  # task1, task2, task3, task5, task4 (ensemble), make_submission
   python run_all.py task1 task5      # only some
   ```
   Tasks 1, 2, 3 and 5 take a few minutes each on a 16-thread CPU (set `NJOBS` to your core count).
4. **Task 4.** The 25 member models need about 130 GPU hours, see `task4/README.md`. To only reproduce the final file, put the saved member probabilities in `outputs/task4_probs/` (supplied with the submission archive, not in git) and set `CPE342_T4_PROBS`, `CPE342_T4_CSV_DIR` as described there.
5. `make_submission.ipynb` writes `outputs/submission_v57a.csv` (the final file) and `outputs/submission_v53c.csv`. Set `REF_V57A` / `REF_V53C` to the files uploaded to Kaggle to compare.

## Verification
Run on the original machine with the saved Task 4 probabilities: every column of `submission_v57a.csv` and `submission_v53c.csv` equals the file that was uploaded to Kaggle (Tasks 1, 2, 4, 5: 100% identical rows; Task 3: mean absolute difference 0.000). Out-of-fold scores: Task 1 F2 0.837, Task 2 macro F1 0.7524, Task 3 0.7907 (see each notebook).

## Rules followed
Provided datasets only; train and validation data combined; Tasks 1, 2, 3, 5 trained from scratch (no pretrained weights); Task 4 uses pretrained backbones; no AutoML; no test labels and no pseudo-labelling. Task 5 has no training file, so its analysis is unsupervised on the test features. The class-prior step in Task 4 estimates the test class shares from the ensemble's own predictions (no labels).
