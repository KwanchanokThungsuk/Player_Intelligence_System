# Player Intelligence System

CPU baseline pipelines are separated by task. Existing artifacts and predictions are preserved; rerun commands only when a new fit is intended.

## Structure

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

## Commands

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

## Preserved baseline results

- Task 1: HistGradientBoostingClassifier, F2 `0.8355668722`, precision `0.5226320`, recall `0.9826635`, threshold `0.17`.
- Task 2: HistGradientBoostingClassifier, macro F1 `0.7491372`, weighted F1 `0.7626101`.
- Task 3: HistGradientBoostingRegressor with absolute-error loss, MAE `3055.2187`, NMAE `0.7778654`.
- Task 5: Isolation Forest, contamination hypothesis `0.05`, anomaly rate `0.0500212`; no F3 is reported because labels are unavailable.

The reported Kaggle baseline score is `0.50879`.

Model artifacts (`artifacts/*.joblib` and `artifacts/*.pt`) are ignored and should not be committed. No Kaggle upload, commit, or push is performed by these commands.
