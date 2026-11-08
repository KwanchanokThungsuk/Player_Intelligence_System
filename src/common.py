"""Shared paths, preprocessing, and output helpers."""
import json
import os
import argparse
from pathlib import Path
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "cpe342-kaggle-main"
COMPETITION_SLUG = "player-intelligence-system-cpe-342-ml"
ARTIFACTS = ROOT / "artifacts"
PREDICTIONS = ROOT / "predictions"
SEED = 342

def resolve_data_root(data_root=None, download=False):
    """Resolve competition data without downloading during import."""
    if data_root:
        candidate = Path(data_root).expanduser().resolve()
    elif os.environ.get("PLAYER_INTELLIGENCE_DATA_DIR"):
        candidate = Path(os.environ["PLAYER_INTELLIGENCE_DATA_DIR"]).expanduser().resolve()
    elif DATA.exists():
        candidate = DATA.resolve()
    elif download:
        import kagglehub
        candidate = Path(kagglehub.competition_download(COMPETITION_SLUG)).resolve()
    else:
        raise FileNotFoundError("Competition data not found. Use --data-root, PLAYER_INTELLIGENCE_DATA_DIR, or --download.")
    if not candidate.exists():
        raise FileNotFoundError(f"Resolved data root does not exist: {candidate}")
    return candidate

def add_data_arguments(parser):
    parser.add_argument("--data-root", default=None, help="Local competition data directory")
    parser.add_argument("--download", action="store_true", help="Download via kagglehub if local data is unavailable")
    return parser

def features(frame, target):
    return frame.drop(columns=["player_id", target], errors="ignore")

def preprocessing(frame, categorical):
    numeric = [c for c in frame.columns if c not in categorical]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True))]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ], remainder="drop")

def save_outputs(name, pipeline, predictions, metrics):
    ARTIFACTS.mkdir(exist_ok=True); PREDICTIONS.mkdir(exist_ok=True)
    joblib.dump(pipeline, ARTIFACTS / f"{name}.joblib")
    predictions.to_csv(PREDICTIONS / f"{name}_test_predictions.csv", index=False)
    (ARTIFACTS / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
