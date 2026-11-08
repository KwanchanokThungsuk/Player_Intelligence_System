"""Validate and build the positional Kaggle submission."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
try:
    from src.common import resolve_data_root
except ModuleNotFoundError:  # direct `python src/build_submission.py`
    from common import resolve_data_root

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description="Build and validate the competition submission")
parser.add_argument("--data-root", default=None, help="Local competition data directory")
parser.add_argument("--download", action="store_true", help="Download via kagglehub if local data is unavailable")
args, _unknown = parser.parse_known_args()
DATA_ROOT = resolve_data_root(args.data_root, args.download)
SAMPLE_PATH = DATA_ROOT / "sample_submission.csv"
EXPECTED_ROWS = 25_889
PREDICTION_DIR = ROOT / "predictions"
header = SAMPLE_PATH.read_text(encoding="utf-8").splitlines()[0]
template_sep = "|" if "|" in header and "," not in header else ","
sample = pd.read_csv(SAMPLE_PATH, sep=template_sep)
if sample.shape != (EXPECTED_ROWS, 6) or sample.columns.tolist() != ["id", "task1", "task2", "task3", "task4", "task5"]:
    raise ValueError(f"Unexpected template shape/columns: {sample.shape}, {sample.columns.tolist()}")
if sample.id.isna().any() or sample.id.eq("").any() or sample.id.duplicated().any():
    raise ValueError("Template IDs must be non-empty and unique")
task_dirs = {1:"01_binary_classification",2:"02_multiclass_classification",3:"03_regression",4:"04_image_classification",5:"05_anomaly_detection"}
valid_task4 = set(pd.read_csv(DATA_ROOT / task_dirs[4] / "train.csv")["class"].dropna().astype(str))
result = pd.DataFrame({"id": sample.id})
for task in range(1, 6):
    name = f"task{task}"
    pred = pd.read_csv(PREDICTION_DIR / f"{name}_test_predictions.csv")
    test = pd.read_csv(DATA_ROOT / task_dirs[task] / "test.csv")
    if len(pred) != EXPECTED_ROWS or len(test) != EXPECTED_ROWS:
        raise ValueError(f"{name} row count mismatch: prediction={len(pred)}, test={len(test)}")
    if pred.columns.tolist() != ["id", name] or pred.id.isna().any() or pred.id.eq("").any() or pred.id.duplicated().any():
        raise ValueError(f"Invalid {name} prediction schema or IDs")
    source_id = "id" if task == 4 else "player_id"
    if pred.id.tolist() != test[source_id].astype(str).tolist():
        raise ValueError(f"{name} prediction IDs are not in test.csv order")
    values = pred[name]
    if task in (1,5) and not set(values).issubset({0,1}): raise ValueError(f"{name} domain invalid")
    if task == 2 and not set(values).issubset({0,1,2,3}): raise ValueError("task2 domain invalid")
    if task == 3 and (not np.isfinite(values).all() or (values < 0).any()): raise ValueError("task3 domain invalid")
    if task == 4 and not set(values.astype(str)).issubset(valid_task4): raise ValueError("task4 domain invalid")
    result[name] = values.to_numpy()
if result.isna().any().any() or not np.isfinite(result.task3).all(): raise ValueError("Submission contains NaN/infinity")
submission_path = ROOT / "submissions" / "submission_baseline_v1.csv"
submission_path.parent.mkdir(exist_ok=True)
result.to_csv(submission_path, sep=template_sep, index=False)
round_trip = pd.read_csv(submission_path, sep=template_sep)
if round_trip.shape != (EXPECTED_ROWS,6) or round_trip.columns.tolist() != sample.columns.tolist() or round_trip.id.tolist() != sample.id.tolist() or round_trip.isna().any().any(): raise ValueError("Round-trip validation failed")
if not set(round_trip.task1).issubset({0,1}) or not set(round_trip.task5).issubset({0,1}) or not set(round_trip.task2).issubset({0,1,2,3}) or not np.isfinite(round_trip.task3).all() or (round_trip.task3 < 0).any() or not set(round_trip.task4.astype(str)).issubset(valid_task4): raise ValueError("Round-trip domains failed")
print(
    f"Template delimiter: {repr(template_sep)}; "
    f"output delimiter: {repr(template_sep)}"
)
print(f"Wrote and validated: {submission_path}")
