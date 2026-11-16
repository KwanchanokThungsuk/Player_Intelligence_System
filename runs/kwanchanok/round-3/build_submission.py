"""Build and validate the Round 3 submission."""
import csv
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
ROUND = Path(__file__).resolve().parent
N_ROWS = 25889


def delimiter(path):
    sample = path.read_text(encoding="utf-8")[:4096]
    return csv.Sniffer().sniff(sample, delimiters=",|\t;").delimiter


def main():
    sample_path = ROOT / "cpe342-kaggle-main/sample_submission.csv"
    sep = delimiter(sample_path)
    sample = pd.read_csv(sample_path, sep=sep)
    expected = ["id", "task1", "task2", "task3", "task4", "task5"]
    if sample.columns.tolist() != expected or len(sample) != N_ROWS:
        raise ValueError("Unexpected sample submission")
    files = {1: ROUND / "outputs/predictions/task1.csv", 2: ROOT / "predictions/task2_test_predictions.csv", 3: ROOT / "predictions/task3_test_predictions.csv", 4: ROOT / "predictions/task4_test_predictions.csv", 5: ROOT / "predictions/task5_test_predictions.csv"}
    parts = {}
    test_paths = {
        2: ROOT / "cpe342-kaggle-main/02_multiclass_classification/test.csv",
        3: ROOT / "cpe342-kaggle-main/03_regression/test.csv",
        4: ROOT / "cpe342-kaggle-main/04_image_classification/test.csv",
        5: ROOT / "cpe342-kaggle-main/05_anomaly_detection/test.csv",
    }
    for task, path in files.items():
        frame = pd.read_csv(path)
        id_column = "id" if "id" in frame.columns else "player_id"
        if len(frame) != N_ROWS or frame[id_column].isna().any() or frame[id_column].astype(str).str.strip().eq("").any() or frame[id_column].duplicated().any():
            raise ValueError(f"Invalid Task {task} prediction IDs")
        if task == 1:
            expected_ids = pd.read_csv(ROOT / "cpe342-kaggle-main/01_binary_classification/test.csv")["player_id"]
        else:
            test_ids = pd.read_csv(test_paths[task])
            expected_ids = test_ids["id"] if "id" in test_ids.columns else test_ids["player_id"]
        if not frame[id_column].astype(str).reset_index(drop=True).equals(expected_ids.astype(str).reset_index(drop=True)):
            raise ValueError(f"Task {task} prediction IDs are not in test order")
        parts[task] = frame.iloc[:, -1].to_numpy()
    train4 = pd.read_csv(ROOT / "cpe342-kaggle-main/04_image_classification/train.csv")
    target4 = "game_title" if "game_title" in train4.columns else "class"
    classes = set(train4[target4].dropna().astype(str))
    task3_values = parts[3].astype(float)
    if set(parts[1]) - {0, 1} or set(parts[2]) - {0, 1, 2, 3} or set(parts[5]) - {0, 1} or not np.isfinite(task3_values).all() or np.any(task3_values < 0) or set(parts[4]) - classes:
        raise ValueError("Prediction domain validation failed")
    out = pd.DataFrame({"id": sample["id"], **{f"task{task}": parts[task] for task in range(1, 6)}})
    if out.isna().any().any() or out.shape != (N_ROWS, 6) or not out["id"].equals(sample["id"]):
        raise ValueError("Submission validation failed")
    output_path = ROUND / "outputs/submission.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, sep=sep, index=False)
    reread = pd.read_csv(output_path, sep=sep)
    if reread.shape != (N_ROWS, 6) or reread.columns.tolist() != expected or not reread["id"].equals(sample["id"]):
        raise ValueError("Written submission validation failed")
    print(f"wrote {output_path} rows={len(reread)} delimiter={sep!r}")


if __name__ == "__main__":
    main()
