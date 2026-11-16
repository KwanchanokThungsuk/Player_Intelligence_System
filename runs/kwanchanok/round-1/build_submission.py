"""Build a round-local submission using round 1 Task 1 and read-only baselines."""
from pathlib import Path
import numpy as np
import pandas as pd

ROUND = Path(__file__).resolve().parent
ROOT = ROUND.parents[2]
DATA = ROOT / "cpe342-kaggle-main"
sample_path = DATA / "sample_submission.csv"
header = sample_path.read_text().splitlines()[0]
sep = "|" if "|" in header and "," not in header else ","
sample = pd.read_csv(sample_path, sep=sep)
if sample.shape != (25889, 6):
    raise ValueError(f"Unexpected sample shape: {sample.shape}")
result = pd.DataFrame({"id": sample.id})
expected_columns = [
    "id",
    "task1",
    "task2",
    "task3",
    "task4",
    "task5",
]

if (
    sample.shape != (25889, 6)
    or sample.columns.tolist() != expected_columns
):
    raise ValueError(
        f"Unexpected sample format: "
        f"{sample.shape}, {sample.columns.tolist()}"
    )

if (
    sample["id"].isna().any()
    or sample["id"].astype(str).str.strip().eq("").any()
    or sample["id"].duplicated().any()
):
    raise ValueError("Sample IDs must be non-empty and unique")
task_dirs = {1:"01_binary_classification",2:"02_multiclass_classification",3:"03_regression",4:"04_image_classification",5:"05_anomaly_detection"}
for task in range(1, 6):
    name = f"task{task}"
    prediction_path = ROUND / "outputs/predictions/task1.csv" if task == 1 else ROOT / "predictions" / f"{name}_test_predictions.csv"
    prediction = pd.read_csv(prediction_path)
    test = pd.read_csv(DATA / task_dirs[task] / "test.csv")
    if len(prediction) != len(sample) or len(test) != len(sample):
        raise ValueError(f"{name} row count mismatch")
    if prediction.columns.tolist() != ["id", name] or prediction.id.isna().any() or prediction.id.duplicated().any():
        raise ValueError(f"Invalid {name} prediction file")
    source_id = "id" if task == 4 else "player_id"
    if prediction.id.astype(str).tolist() != test[source_id].astype(str).tolist():
        raise ValueError(f"{name} IDs do not preserve test order")
    values = prediction[name]
    if task in (1, 5) and not set(values).issubset({0, 1}): raise ValueError(f"{name} domain invalid")
    if task == 2 and not set(values).issubset({0, 1, 2, 3}): raise ValueError("task2 domain invalid")
    if task == 3 and (not np.isfinite(values).all() or (values < 0).any()): raise ValueError("task3 domain invalid")
    if task == 4:
        valid = set(pd.read_csv(DATA / "04_image_classification/train.csv")["class"].astype(str))
        if not set(values.astype(str)).issubset(valid): raise ValueError("task4 domain invalid")
    result[name] = values.to_numpy()
submission_path = ROUND / "outputs/submission.csv"
submission_path.parent.mkdir(parents=True, exist_ok=True)

result.to_csv(
    submission_path,
    sep=sep,
    index=False,
)

check = pd.read_csv(
    submission_path,
    sep=sep,
)

if (
    check.shape != (25889, 6)
    or check.columns.tolist() != sample.columns.tolist()
    or check["id"].tolist() != sample["id"].tolist()
    or check.isna().any().any()
    or not np.isfinite(check["task3"]).all()
):
    raise ValueError("Round submission validation failed")

print(f"Delimiter: {repr(sep)}")
print(f"Wrote: {submission_path}")