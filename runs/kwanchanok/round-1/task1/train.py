"""Task 1 improvement round 1: select threshold with macro F2."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.common import SEED, features, preprocessing
DATA = ROOT / "cpe342-kaggle-main"
ROUND = Path(__file__).resolve().parents[1]
MAX_ITER = 120
LEARNING_RATE = 0.10
MAX_LEAF_NODES = 31
L2_REGULARIZATION = 1.0
BASELINE_THRESHOLD = 0.17
THRESHOLD_GRID = np.linspace(0.05, 0.95, 181)

def make_model():
    return HistGradientBoostingClassifier(max_iter=MAX_ITER, learning_rate=LEARNING_RATE, max_leaf_nodes=MAX_LEAF_NODES, l2_regularization=L2_REGULARIZATION, random_state=SEED)

def main():
    train = pd.read_csv(DATA / "01_binary_classification/train.csv")
    test = pd.read_csv(DATA / "01_binary_classification/test.csv")
    train = train.dropna(subset=["is_cheater"]).copy()
    y = train.pop("is_cheater").astype(int)
    X = features(train, "is_cheater")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    pipe = Pipeline([("prep", preprocessing(X, [])), ("model", make_model())])
    pipe.fit(X_train, y_train)
    probabilities = pipe.predict_proba(X_val)[:, 1]
    macro_scores = [fbeta_score(y_val, probabilities >= threshold, beta=2, average="macro", zero_division=0) for threshold in THRESHOLD_GRID]
    threshold = float(THRESHOLD_GRID[int(np.argmax(macro_scores))])
    predictions = (probabilities >= threshold).astype(int)
    baseline_predictions = (probabilities >= BASELINE_THRESHOLD).astype(int)
    metrics = {
        "task": 1, "round": "kwanchanok/round-1", "model": "HistGradientBoostingClassifier", "seed": SEED,
        "validation_rows": int(len(y_val)), "threshold": threshold, "baseline_threshold": BASELINE_THRESHOLD,
        "macro_f2": float(fbeta_score(y_val, predictions, beta=2, average="macro", zero_division=0)),
        "binary_f2_cheater": float(fbeta_score(y_val, predictions, beta=2, average="binary", zero_division=0)),
        "precision": float(precision_score(y_val, predictions, zero_division=0)),
        "recall": float(recall_score(y_val, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_val, predictions).tolist(),
        "predicted_positive_rate": float(predictions.mean()),
        "baseline_macro_f2": float(fbeta_score(y_val, baseline_predictions, beta=2, average="macro", zero_division=0)),
        "baseline_binary_f2_cheater": float(fbeta_score(y_val, baseline_predictions, beta=2, average="binary", zero_division=0)),
        "baseline_predicted_positive_rate": float(baseline_predictions.mean()),
    }
    final = Pipeline([("prep", preprocessing(X, [])), ("model", make_model())])
    final.fit(X, y)
    output = pd.DataFrame({"id": test.player_id, "task1": (final.predict_proba(features(test, "is_cheater"))[:, 1] >= threshold).astype(int)})
    output.to_csv(ROUND / "outputs/predictions/task1.csv", index=False)
    (ROUND / "outputs/metrics/task1.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
