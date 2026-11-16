"""Round 5 Task 5: reuse Round 4 stability evidence and retain root baseline."""
import hashlib, json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[4]; ROUND=Path(__file__).resolve().parents[1]

def digest(frame): return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
def main():
    prior=json.loads((ROUND.parent/"round-4/outputs/metrics/task5.json").read_text()); root=pd.read_csv(ROOT/"predictions/task5_test_predictions.csv")
    output=root.copy()
    if len(output)!=25889 or output.id.isna().any() or output.id.duplicated().any() or set(output.task5)-{0,1}: raise ValueError("invalid Task 5 baseline")
    report={"task":5,"official_metric":"F3 on hidden labels; no labeled local validation metric","selection_policy":"retain root unless an alternative has clearly stronger stability evidence",
            "round4_evidence":{"baseline_mean_seed_jaccard":prior["candidates"]["isolation_forest_baseline"]["mean_seed_jaccard"],"robust_mean_seed_jaccard":prior["candidates"]["isolation_forest_robust"]["mean_seed_jaccard"],
                               "lof_baseline_jaccard":prior["candidates"]["local_outlier_factor_transductive"]["jaccard_with_baseline_seed342_contamination005"],"contamination_values":prior["contamination_values"],"seeds":prior["seeds"]},
            "additional_ensemble_evaluated":False,"reason":"Round 4 robust preprocessing did not improve stability; without labels, another rank ensemble is not sufficiently justified by Kaggle-independent evidence",
            "selected_model":"root_isolation_forest_baseline","selection_passed":False,"anomaly_rate":float(output.task5.mean()),"changed_test_rows":0,"root_prediction_sha256":digest(root),"output_prediction_sha256":digest(output)}
    (ROUND/"outputs/predictions").mkdir(parents=True,exist_ok=True); (ROUND/"outputs/metrics").mkdir(parents=True,exist_ok=True); output.to_csv(ROUND/"outputs/predictions/task5.csv",index=False); (ROUND/"outputs/metrics/task5.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
