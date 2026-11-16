"""Round 4 Task 2: five-fold OOF multiclass comparison."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
ROOT = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(ROOT))
from src.common import SEED, preprocessing
DATA = ROOT / "cpe342-kaggle-main/02_multiclass_classification"; ROUND = Path(__file__).resolve().parents[1]

def model(kind):
    if kind == "hgb": return HistGradientBoostingClassifier(max_iter=120, learning_rate=.1, max_leaf_nodes=31, l2_regularization=1., random_state=SEED)
    return ExtraTreesClassifier(n_estimators=240, min_samples_leaf=2, max_features="sqrt", class_weight="balanced", random_state=SEED, n_jobs=-1)

def summarize(y, pred, splits):
    fold = [float(f1_score(y[v], pred[v], average="macro", zero_division=0)) for _, v in splits]
    return {"oof_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "per_class_f1": f1_score(y, pred, average=None, labels=[0,1,2,3], zero_division=0).tolist(),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0,1,2,3]).tolist(), "fold_scores": fold,
            "fold_mean": float(np.mean(fold)), "fold_std": float(np.std(fold, ddof=1))}

def main():
    train = pd.read_csv(DATA / "train.csv"); test = pd.read_csv(DATA / "test.csv"); y = train.segment.astype(int).to_numpy()
    X = train.drop(columns=["player_id", "segment"]); Xt = test.drop(columns=["player_id"]); cats = X.select_dtypes(exclude=np.number).columns.tolist()
    splits = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y)); results = {}; oofs = {}
    for name, kind in {"hgb_baseline":"hgb", "extra_trees_balanced":"extra"}.items():
        probabilities = np.zeros((len(y), 4))
        for tr, va in splits:
            pipe = Pipeline([("prep", preprocessing(X, cats)), ("model", model(kind))]); pipe.fit(X.iloc[tr], y[tr]); probabilities[va] = pipe.predict_proba(X.iloc[va])
        pred = probabilities.argmax(axis=1); results[name] = summarize(y, pred, splits); oofs[name] = probabilities
    baseline = "hgb_baseline"; best = max(results, key=lambda k: results[k]["oof_macro_f1"])
    wins = sum(a >= b for a,b in zip(results[best]["fold_scores"], results[baseline]["fold_scores"]))
    passed = best != baseline and results[best]["oof_macro_f1"] > results[baseline]["oof_macro_f1"] and wins >= 3; selected = best if passed else baseline
    reason = (f"{best} improved OOF Macro F1 and won/tied {wins}/5 folds" if passed else f"retained baseline; best={best}, wins/ties={wins}/5, gain={results[best]['oof_macro_f1']-results[baseline]['oof_macro_f1']:.6f}")
    kind = "extra" if selected.startswith("extra") else "hgb"; test_probs = []
    for tr, _ in splits:
        pipe = Pipeline([("prep", preprocessing(X, cats)), ("model", model(kind))]); pipe.fit(X.iloc[tr], y[tr]); test_probs.append(pipe.predict_proba(Xt))
    proposed = np.mean(test_probs, axis=0).argmax(axis=1); root = pd.read_csv(ROOT / "predictions/task2_test_predictions.csv")
    output = pd.DataFrame({"id":test.player_id, "task2":proposed}) if passed else root.copy()
    if len(output)!=25889 or not output.id.equals(test.player_id) or set(output.task2)-{0,1,2,3}: raise ValueError("invalid Task 2 output")
    changed = int((output.task2.to_numpy()!=root.task2.to_numpy()).sum())
    metrics = {"task":2, "official_metric":"multiclass F1; Macro F1 used because official PDF does not state averaging", "seed":SEED,
               "candidate_configurations":{"hgb":{"max_iter":120,"learning_rate":.1,"max_leaf_nodes":31,"l2_regularization":1.,"random_state":SEED},"extra_trees":{"n_estimators":240,"min_samples_leaf":2,"max_features":"sqrt","class_weight":"balanced","random_state":SEED}},
               "fold_configuration":{"n_splits":5,"shuffle":True,"random_state":SEED}, "candidates":results, "baseline_candidate":baseline,
               "root_validation_macro_f1":0.7491371752784082, "best_candidate":best, "selected_model":selected, "selection_passed":passed,
               "selection_reason":reason, "changed_test_rows":changed, "catboost_omitted":"not installed; no dependency was added"}
    (ROUND/"outputs/predictions").mkdir(parents=True,exist_ok=True); (ROUND/"outputs/metrics").mkdir(parents=True,exist_ok=True)
    output.to_csv(ROUND/"outputs/predictions/task2.csv",index=False); (ROUND/"outputs/metrics/task2.json").write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))

if __name__=="__main__": main()
