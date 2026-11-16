"""Round 4 Task 1: five-fold OOF binary-F2 model comparison."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(ROOT))
from src.common import SEED, preprocessing
DATA = ROOT / "cpe342-kaggle-main/01_binary_classification"; ROUND = Path(__file__).resolve().parents[1]
GRID = np.linspace(.05, .95, 181); FOLDS = 5; MIN_OOF_GAIN = .001

def add_features(frame):
    out = frame.drop(columns=["player_id", "is_cheater"], errors="ignore").copy()
    out["missing_count"] = out.isna().sum(axis=1); out["missing_rate"] = out.isna().mean(axis=1)
    for c in ["reports_received", "headshot_percentage", "accuracy_score"]:
        if c in out and out[c].dropna().ge(0).all(): out[f"{c}_log1p"] = np.log1p(out[c])
    for a, b in [("accuracy_score", "reaction_time_ms"), ("headshot_percentage", "accuracy_score")]:
        if a in out and b in out: out[f"{a}_over_{b}"] = out[a] / (out[b].abs() + 1e-6)
    return out

def model(kind):
    if kind == "hgb": return HistGradientBoostingClassifier(max_iter=120, learning_rate=.1, max_leaf_nodes=31, l2_regularization=1., random_state=SEED)
    return ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=-1)

def threshold(y, p):
    scores = np.array([fbeta_score(y, p >= t, beta=2, zero_division=0) for t in GRID]); tied = GRID[np.isclose(scores, scores.max(), atol=1e-12, rtol=0)]
    return float(sorted(tied, key=lambda x: (abs(x-.5), x))[0])

def summary(y, p, t, splits):
    q = (p >= t).astype(int); fold = [float(fbeta_score(y[v], q[v], beta=2, zero_division=0)) for _, v in splits]
    return {"oof_f2": float(fbeta_score(y, q, beta=2, zero_division=0)), "binary_f2": float(fbeta_score(y, q, beta=2, zero_division=0)),
            "macro_f2_diagnostic": float(fbeta_score(y, q, beta=2, average="macro", zero_division=0)),
            "precision": float(precision_score(y, q, zero_division=0)), "recall": float(recall_score(y, q, zero_division=0)),
            "confusion_matrix": confusion_matrix(y, q).tolist(), "positive_rate": float(q.mean()), "threshold": t,
            "fold_scores": fold, "fold_mean": float(np.mean(fold)), "fold_std": float(np.std(fold, ddof=1))}

def main():
    train = pd.read_csv(DATA / "train.csv").dropna(subset=["is_cheater"]).copy(); test = pd.read_csv(DATA / "test.csv")
    y = train["is_cheater"].astype(int).to_numpy(); raw = train.drop(columns=["player_id", "is_cheater"]); eng = add_features(train)
    test_raw = test.drop(columns=["player_id"]); test_eng = add_features(test)
    splits = list(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(raw, y))
    candidates = {"hgb_raw": ("hgb", raw), "hgb_engineered": ("hgb", eng), "extra_trees_raw": ("extra", raw), "extra_trees_engineered": ("extra", eng)}
    results = {}; oofs = {}
    for name, (kind, X) in candidates.items():
        p = np.full(len(y), np.nan)
        for tr, va in splits:
            pipe = Pipeline([("prep", preprocessing(X, [])), ("model", model(kind))]); pipe.fit(X.iloc[tr], y[tr]); p[va] = pipe.predict_proba(X.iloc[va])[:, 1]
        t = threshold(y, p); results[name] = summary(y, p, t, splits); oofs[name] = p
    baseline = "hgb_raw"; best = max(results, key=lambda k: results[k]["oof_f2"])
    wins = sum(a >= b for a, b in zip(results[best]["fold_scores"], results[baseline]["fold_scores"]))
    gain = results[best]["oof_f2"] - results[baseline]["oof_f2"]
    passed = best != baseline and gain >= MIN_OOF_GAIN and wins >= 3; selected = best if passed else baseline
    reason = (f"{best} improved OOF F2 by at least {MIN_OOF_GAIN} and won/tied {wins}/5 folds" if passed else f"retained baseline; best={best}, wins/ties={wins}/5, gain={gain:.6f}, required gain={MIN_OOF_GAIN:.3f}")
    selected_kind, selected_X = candidates[selected]; selected_test = test_eng if "engineered" in selected else test_raw; probs = []
    for tr, _ in splits:
        pipe = Pipeline([("prep", preprocessing(selected_X, [])), ("model", model(selected_kind))]); pipe.fit(selected_X.iloc[tr], y[tr]); probs.append(pipe.predict_proba(selected_test)[:, 1])
    proposed = (np.mean(probs, axis=0) >= results[selected]["threshold"]).astype(int); root = pd.read_csv(ROOT / "predictions/task1_test_predictions.csv")
    output = pd.DataFrame({"id": test.player_id, "task1": proposed}) if passed else root.copy()
    if len(output) != 25889 or not output.id.equals(test.player_id) or set(output.task1) - {0, 1}: raise ValueError("invalid Task 1 output")
    changed = int((output.task1.to_numpy() != root.task1.to_numpy()).sum())
    metrics = {"task": 1, "official_metric": "binary F2", "seed": SEED, "fold_configuration": {"n_splits": 5, "shuffle": True, "random_state": SEED},
               "candidate_configurations":{"hgb":{"max_iter":120,"learning_rate":.1,"max_leaf_nodes":31,"l2_regularization":1.,"random_state":SEED},"extra_trees":{"n_estimators":200,"min_samples_leaf":2,"max_features":"sqrt","class_weight":"balanced","random_state":SEED}},
               "threshold_grid": [float(x) for x in GRID], "candidates": results, "baseline_candidate": baseline, "root_validation_f2": 0.835566872152238,
               "best_candidate": best, "selected_model": selected, "selection_passed": passed, "minimum_oof_gain": MIN_OOF_GAIN, "selection_reason": reason, "selected_threshold": results[selected]["threshold"],
               "comparison_thresholds": {"0.17": summary(y, oofs[selected], .17, splits), "0.38": summary(y, oofs[selected], .38, splits)},
               "changed_test_rows": changed, "test_positive_rate": float(output.task1.mean()), "catboost_omitted": "not installed; no dependency was added"}
    (ROUND / "outputs/predictions").mkdir(parents=True, exist_ok=True); (ROUND / "outputs/metrics").mkdir(parents=True, exist_ok=True)
    output.to_csv(ROUND / "outputs/predictions/task1.csv", index=False); (ROUND / "outputs/metrics/task1.json").write_text(json.dumps(metrics, indent=2)); print(json.dumps(metrics, indent=2))

if __name__ == "__main__": main()
