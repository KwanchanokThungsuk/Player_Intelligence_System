# CPE342 Karena Player Intelligence System

Team: [TEAM: team ID and members]

This repository holds our solution of the CPE342 Kaggle project (five tasks: anti-cheat pre-filter, player segment, monthly spending, game-title detection, account anomaly detection). The **final code that produced the two files selected for the Kaggle ranking** is organised per task at the repository root, as on slide 6 of the project instruction. Final public score **0.84167** (file V57a); the second selected file is V53c (0.84113).

## Final solution

```text
README.md                 this file
requirements.txt          packages (Python 3.11)
run_all.py                runs all notebooks below in order
task1/task1.ipynb         anti-cheat pre-filter      LightGBM raw + engineered features, threshold 0.18 for F2
task2/task2.ipynb         player segment             LightGBM multiclass, tuned class weights
task3/task3.ipynb         monthly spending           two-stage lognormal-median + L1 LightGBM blend
task4/                    game title detection       train_backbone.py + train_all_members.sh (training), task4_ensemble.ipynb (ensemble + class-prior step), README.md
task5/task5.ipynb         account anomaly detection  unsupervised gap groups + missing-value classifier + tail rows
make_submission/make_submission.ipynb   combines the per-task outputs into submission_v57a.csv and submission_v53c.csv
outputs/                  written by the notebooks (not committed)
```

### How to run
1. `pip install -r requirements.txt` (Task 4 training also needs an NVIDIA GPU and the CUDA build of PyTorch).
2. Download the competition data from Kaggle (the **corrected release of 2026-11-12**) and point `CPE342_DATA` at the folder that contains `01_binary_classification/`, ..., `05_anomaly_detection/` and `sample_submission.csv`:
   ```bash
   export CPE342_DATA=/path/to/the/kaggle/download           # Windows PowerShell: $env:CPE342_DATA = "C:/path/to/the/kaggle/download"
   ```
   The dataset tracked in `cpe342-kaggle-main/` is the **older release**: do not use it for Tasks 2 and 5 (the notebooks only read the original `04_image_classification/test.csv` from it, see `task4/README.md`).
3. Run the notebooks from Jupyter (Run All; the working directory must be the notebook's own folder) or all at once:
   ```bash
   python run_all.py                  # task1, task2, task3, task5, task4 (ensemble), make_submission
   python run_all.py task1 task5      # only some
   ```
   Tasks 1, 2, 3 and 5 take a few minutes each on a 16-thread CPU (set `NJOBS` to your core count).
4. **Task 4.** The 25 member models need about 130 GPU hours (see `task4/README.md`). To only reproduce the final file, put the saved member probabilities in `outputs/task4_probs/` (supplied with the submission archive, not committed to git) and set `CPE342_T4_CSV_DIR` if the corrected `test.csv` is not in `$CPE342_DATA/04_image_classification`.
5. `make_submission.ipynb` writes `outputs/submission_v57a.csv` (the final file) and `outputs/submission_v53c.csv`. Set `REF_V57A` / `REF_V53C` to the files uploaded to Kaggle to compare.

### Verification
Run on the original machine with the saved Task 4 probabilities: every column of `submission_v57a.csv` and `submission_v53c.csv` equals the file that was uploaded to Kaggle (Tasks 1, 2, 4, 5: 100% identical rows; Task 3: mean absolute difference below 0.0005). Out-of-fold scores: Task 1 F2 0.837, Task 2 macro F1 0.7524, Task 3 0.7907 (see each notebook).

### Rules followed
Provided datasets only; train and validation data combined; Tasks 1, 2, 3, 5 trained from scratch (no pretrained weights); Task 4 uses pretrained backbones; no AutoML; no test labels and no pseudo-labelling. Task 5 has no training file, so its analysis is unsupervised on the test features. The class-prior step in Task 4 estimates the test class shares from the ensemble's own predictions (no labels).

### Experiments
Each member's trial-and-error work (other models, diagnostics, weight sweeps, every Kaggle score in `notes.md`) stays on the member branches `runs/<name>`, for example `runs/wirachat/round-1`, `round-2`, `round-3`. It is kept apart from the final code and is not needed to reproduce the submission.

## Baseline pipelines (original team baseline, unchanged)

CPU baseline pipelines are separated by task. Existing artifacts and predictions are preserved; rerun commands only when a new fit is intended.

### Structure

```text
src/common.py
src/task1/train.py                  # anti-cheat classification
src/task2/train.py                  # player segment classification
src/task3/train.py                  # monthly spending regression
notebooks/eda/task1_eda.ipynb       # EDA notebooks for Tasks 1-5
notebooks/training/task4_kaggle_train.ipynb
src/task5/train.py                  # unsupervised Isolation Forest
src/build_submission.py             # submission assembly and validation
```

### Commands

Install dependencies:

```bash
pip install -r requirements.txt
```

Authenticate Kaggle using the Kaggle-supported method for your environment (for example, `KAGGLE_API_TOKEN` or a local `kaggle.json`). Do not commit credentials.

Data resolution order is: `--data-root`, `PLAYER_INTELLIGENCE_DATA_DIR`, local `cpe342-kaggle-main/`, then `kagglehub.competition_download` only when `--download` is supplied. Example:

```bash
python -m src.task1.train --download
```

The download path returned by kagglehub is used directly; no Kaggle cache path is hardcoded. Existing local data remains the default fallback.

```bash
python -m src.task1.train
python -m src.task2.train
python -m src.task3.train
python -m src.task5.train
python src/build_submission.py
```

Task 4 is run on Kaggle or a local environment with PyTorch/torchvision using `notebooks/training/task4_kaggle_train.ipynb`, or locally with `python -m src.task4.train`. The notebook and entry point write `task4_test_predictions.csv` and the class mapping before the final submission command.

### Preserved baseline results

- Task 1: HistGradientBoostingClassifier, F2 `0.8355668722`, precision `0.5226320`, recall `0.9826635`, threshold `0.17`.
- Task 2: HistGradientBoostingClassifier, macro F1 `0.7491372`, weighted F1 `0.7626101`.
- Task 3: HistGradientBoostingRegressor with absolute-error loss, MAE `3055.2187`, NMAE `0.7778654`.
- Task 5: Isolation Forest, contamination hypothesis `0.05`, anomaly rate `0.0500212`; no F3 is reported because labels are unavailable.

The reported Kaggle baseline score is `0.50879`.

Model artifacts (`artifacts/*.joblib` and `artifacts/*.pt`) are ignored and should not be committed. No Kaggle upload, commit, or push is performed by these commands.
