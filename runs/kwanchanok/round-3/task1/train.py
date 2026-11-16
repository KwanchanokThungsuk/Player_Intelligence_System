"""Task 1 Round 3: five-fold OOF comparison of raw and engineered features."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from src.common import SEED, preprocessing

DATA = ROOT / "cpe342-kaggle-main"
ROUND = Path(__file__).resolve().parents[1]
N_SPLITS = 5
MAX_ITER = 120
LEARNING_RATE = 0.10
MAX_LEAF_NODES = 31
L2_REGULARIZATION = 1.0
THRESHOLD_GRID = np.linspace(0.05, 0.95, 181)
LOG_CANDIDATES = ["reports_received", "headshot_percentage", "accuracy_score"]
INTERACTION_CANDIDATES = [("accuracy_score", "reaction_time_ms"), ("headshot_percentage", "accuracy_score")]


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic, row-local features without fitting on any data."""
    out = frame.drop(columns=["player_id", "is_cheater"], errors="ignore").copy()
    numeric = out.select_dtypes(include=np.number)
    out["missing_count"] = out.isna().sum(axis=1)
    out["missing_rate"] = out.isna().mean(axis=1)
    groups = {}
    for column in numeric.columns:
        match = re.match(r"^(.*)_([1-4])$", column)
        if match:
            groups.setdefault(match.group(1), []).append(column)
    for base, columns in sorted(groups.items()):
        columns = sorted(columns, key=lambda c: int(c.rsplit("_", 1)[1]))
        if len(columns) < 2:
            continue
        values = out[columns]
        out[f"{base}_mean"] = values.mean(axis=1)
        out[f"{base}_std"] = values.std(axis=1)
        out[f"{base}_min"] = values.min(axis=1)
        out[f"{base}_max"] = values.max(axis=1)
        out[f"{base}_range"] = values.max(axis=1) - values.min(axis=1)
        out[f"{base}_change"] = values.iloc[:, -1] - values.iloc[:, 0]
    for column in LOG_CANDIDATES:
        if column in out and out[column].dropna().ge(0).all():
            out[f"{column}_log1p"] = np.log1p(out[column])
    for numerator, denominator in INTERACTION_CANDIDATES:
        if numerator in out and denominator in out:
            out[f"{numerator}_over_{denominator}"] = out[numerator] / (out[denominator].abs() + 1e-6)
    return out


def make_model():
    return HistGradientBoostingClassifier(
        max_iter=MAX_ITER, learning_rate=LEARNING_RATE, max_leaf_nodes=MAX_LEAF_NODES,
        l2_regularization=L2_REGULARIZATION, random_state=SEED,
    )


def make_pipeline(frame):
    return Pipeline([("prep", preprocessing(frame, [])), ("model", make_model())])


def choose_threshold(y, probabilities):
    scores = np.array([
        fbeta_score(y, probabilities >= threshold, beta=2, average="macro", zero_division=0)
        for threshold in THRESHOLD_GRID
    ])
    best = scores.max()
    tied = THRESHOLD_GRID[np.isclose(scores, best, rtol=0, atol=1e-12)]
    return float(sorted(tied, key=lambda value: (abs(value - 0.5), value))[0])


def score(y, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "macro_f2": float(fbeta_score(y, predictions, beta=2, average="macro", zero_division=0)),
        "binary_f2_cheater": float(fbeta_score(y, predictions, beta=2, average="binary", zero_division=0)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
        "predicted_positive_rate": float(predictions.mean()),
    }


def evaluate(frame, y, folds):
    oof = np.full(len(y), np.nan)
    fold_scores = []
    fold_sizes = []
    fold_positive_rates = []
    for train_idx, valid_idx in folds:
        pipeline = make_pipeline(frame)
        pipeline.fit(frame.iloc[train_idx], y.iloc[train_idx])
        oof[valid_idx] = pipeline.predict_proba(frame.iloc[valid_idx])[:, 1]
        fold_sizes.append(int(len(valid_idx)))
        fold_positive_rates.append(float(y.iloc[valid_idx].mean()))
    assert np.isfinite(oof).all()
    threshold = choose_threshold(y.to_numpy(), oof)
    overall = score(y.to_numpy(), oof, threshold)
    for (_, valid_idx) in folds:
        fold_scores.append(score(y.iloc[valid_idx].to_numpy(), oof[valid_idx], threshold)["macro_f2"])
    overall.update({"threshold": threshold, "oof_probabilities": oof, "fold_macro_f2": fold_scores,
                    "fold_sizes": fold_sizes, "fold_positive_label_rates": fold_positive_rates,
                    "fold_macro_f2_mean": float(np.mean(fold_scores)),
                    "fold_macro_f2_std": float(np.std(fold_scores, ddof=1))})
    return overall


def final_ensemble(frame, y, test_frame, folds, threshold):
    test_probabilities = []
    for train_idx, _ in folds:
        pipeline = make_pipeline(frame)
        pipeline.fit(frame.iloc[train_idx], y.iloc[train_idx])
        test_probabilities.append(pipeline.predict_proba(test_frame)[:, 1])
    return (np.mean(test_probabilities, axis=0) >= threshold).astype(int)


def compare_predictions(new, path):
    previous = pd.read_csv(path)
    if len(previous) != len(new) or not previous["id"].equals(new["id"]):
        raise ValueError(f"Prediction order mismatch: {path}")
    changed = int((previous["task1"].astype(int).to_numpy() != new["task1"].to_numpy()).sum())
    return {"count": changed, "percentage": float(100 * changed / len(new))}


def main():
    train = pd.read_csv(DATA / "01_binary_classification/train.csv").dropna(subset=["is_cheater"]).copy()
    test = pd.read_csv(DATA / "01_binary_classification/test.csv")
    y = train.pop("is_cheater").astype(int)
    raw = train.drop(columns=["player_id"])
    engineered = add_features(train)
    folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(raw, y))
    reference = evaluate(raw, y, folds)
    engineered_metrics = evaluate(engineered, y, folds)
    better_folds = sum(b > a for a, b in zip(reference["fold_macro_f2"], engineered_metrics["fold_macro_f2"]))
    improvement = engineered_metrics["macro_f2"] - reference["macro_f2"]
    selected = "engineered" if improvement >= 0.005 and better_folds >= 3 else "baseline"
    reason = (f"engineered OOF Macro F2 improvement {improvement:.6f} >= 0.005 and "
              f"better in {better_folds}/5 folds") if selected == "engineered" else (
              f"engineered improvement {improvement:.6f} and {better_folds}/5 better folds; "
              "selection rule requires both improvement >= 0.005 and at least 3 better folds")
    selected_frame = engineered if selected == "engineered" else raw
    selected_metrics = engineered_metrics if selected == "engineered" else reference
    test_frame = add_features(test) if selected == "engineered" else test.drop(columns=["player_id"])
    predictions = final_ensemble(selected_frame, y, test_frame, folds, selected_metrics["threshold"])
    output = pd.DataFrame({"id": test["player_id"], "task1": predictions})
    if len(output) != 25889 or output["id"].isna().any() or output["id"].duplicated().any() or set(output["task1"]) - {0, 1}:
        raise ValueError("Invalid Task 1 output")
    (ROUND / "outputs/predictions").mkdir(parents=True, exist_ok=True)
    (ROUND / "outputs/metrics").mkdir(parents=True, exist_ok=True)
    output.to_csv(ROUND / "outputs/predictions/task1.csv", index=False)
    metrics = {
        "seed": SEED, "fold_configuration": {"n_splits": N_SPLITS, "shuffle": True, "random_state": SEED},
        "model": "HistGradientBoostingClassifier",
        "model_hyperparameters": {"max_iter": MAX_ITER, "learning_rate": LEARNING_RATE, "max_leaf_nodes": MAX_LEAF_NODES, "l2_regularization": L2_REGULARIZATION},
        "threshold_grid": [float(x) for x in THRESHOLD_GRID], "train_rows": int(len(train)), "test_rows": int(len(test)),
        "raw_feature_count": int(raw.shape[1]), "engineered_feature_count": int(engineered.shape[1]),
        "added_feature_count": int(engineered.shape[1] - raw.shape[1]),
        "added_feature_names": [c for c in engineered if c not in raw],
        "temporal_feature_groups": sorted({m.group(1) for c in raw.select_dtypes(include=np.number).columns if (m := re.match(r"^(.*)_([1-4])$", c))}),
        "log_transformed_columns": [c for c in LOG_CANDIDATES if f"{c}_log1p" in engineered],
        "interaction_features": [f"{a}_over_{b}" for a, b in INTERACTION_CANDIDATES if f"{a}_over_{b}" in engineered],
        "baseline": {k: v for k, v in reference.items() if k != "oof_probabilities"},
        "engineered": {k: v for k, v in engineered_metrics.items() if k != "oof_probabilities"},
        "selected_pipeline": selected, "selection_rule": "engineered must improve OOF Macro F2 by >= 0.005 and win at least 3/5 folds",
        "selection_reason": reason, "selected_threshold": selected_metrics["threshold"],
        "oof_macro_f2": selected_metrics["macro_f2"], "fold_macro_f2": selected_metrics["fold_macro_f2"],
        "fold_macro_f2_mean": selected_metrics["fold_macro_f2_mean"], "fold_macro_f2_std": selected_metrics["fold_macro_f2_std"],
        "precision": selected_metrics["precision"], "recall": selected_metrics["recall"],
        "binary_f2_cheater": selected_metrics["binary_f2_cheater"], "confusion_matrix": selected_metrics["confusion_matrix"],
        "oof_predicted_positive_rate": selected_metrics["predicted_positive_rate"],
        "test_predicted_positive_rate": float(predictions.mean()), "better_folds": better_folds, "macro_f2_improvement": float(improvement),
        "changed_vs_root_baseline": compare_predictions(output, ROOT / "predictions/task1_test_predictions.csv"),
        "changed_vs_round1": compare_predictions(output, ROUND.parent / "round-1/outputs/predictions/task1.csv"),
        "changed_vs_round2": compare_predictions(output, ROUND.parent / "round-2/outputs/predictions/task1.csv"),
    }
    (ROUND / "outputs/metrics/task1.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
