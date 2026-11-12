"""Round 1 helpers (own code; does not touch the shared src/ files)."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("CPE342_DATA", ROOT / "cpe342-kaggle-main"))   # set CPE342_DATA to use a different (e.g. re-downloaded) data folder
OUT = Path(__file__).resolve().parent  # runs/wirachat/round-1
SEED = 342
DIRS = {1: "01_binary_classification", 2: "02_multiclass_classification", 3: "03_regression",
        4: "04_image_classification", 5: "05_anomaly_detection"}


def load(task, name):
    return pd.read_csv(DATA / DIRS[task] / f"{name}.csv")


def macro_fbeta(y, pred, beta):
    return float(fbeta_score(y, pred, beta=beta, average="macro", zero_division=0))


def best_threshold(y, prob, beta, grid=None, macro=True):
    grid = np.linspace(0.02, 0.98, 193) if grid is None else grid
    f = (lambda t: macro_fbeta(y, prob >= t, beta)) if macro else (lambda t: float(fbeta_score(y, prob >= t, beta=beta, zero_division=0)))
    scores = np.array([f(t) for t in grid])
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i]), grid, scores


def nmae_score(y, pred):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    return float(1.0 / (1.0 + np.abs(y - pred).sum() / y.sum()))
