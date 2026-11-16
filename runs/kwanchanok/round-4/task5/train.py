"""Round 4 Task 5: unlabeled stability analysis; conservative baseline retention."""
import json, sys
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
ROOT=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(ROOT))
from src.common import SEED
DATA=ROOT/"cpe342-kaggle-main/05_anomaly_detection"; ROUND=Path(__file__).resolve().parents[1]
SEEDS=[SEED,SEED+1,SEED+2]; CONTAMINATIONS=[.03,.05,.07]

def labels(scores,contamination):
    n=max(1,int(np.ceil(len(scores)*contamination))); idx=np.argsort(scores,kind="stable")[:n]; out=np.zeros(len(scores),dtype=int); out[idx]=1; return out

def jaccard(a,b):
    a=a.astype(bool); b=b.astype(bool); union=np.logical_or(a,b).sum(); return float(np.logical_and(a,b).sum()/union) if union else 1.

def quantiles(scores): return {str(q):float(np.quantile(scores,q)) for q in [0,.01,.05,.25,.5,.75,.95,.99,1]}

def summarize(label_sets,score_sets):
    central=[label_sets[(s,.05)] for s in SEEDS]
    seed_pairs=[jaccard(a,b) for a,b in combinations(central,2)]
    sensitivity=[]
    for s in SEEDS:
        sensitivity.extend([jaccard(label_sets[(s,.05)],label_sets[(s,.03)]),jaccard(label_sets[(s,.05)],label_sets[(s,.07)])])
    return {"anomaly_rates":{f"seed_{s}_contamination_{c}":float(label_sets[(s,c)].mean()) for s in SEEDS for c in CONTAMINATIONS},
            "seed_jaccards_at_0.05":seed_pairs,"mean_seed_jaccard":float(np.mean(seed_pairs)),
            "contamination_sensitivity_jaccards":sensitivity,"mean_contamination_jaccard":float(np.mean(sensitivity)),
            "score_quantiles_by_seed":{str(s):quantiles(score_sets[s]) for s in SEEDS}}

def main():
    raw=pd.read_csv(DATA/"test.csv"); ids=raw.player_id.copy(); X=raw.drop(columns=["player_id"])
    if X.select_dtypes(exclude=np.number).shape[1]: raise ValueError("unexpected categorical Task 5 features")
    candidate_labels={}; candidate_scores={}
    for name,robust in [("isolation_forest_baseline",False),("isolation_forest_robust",True)]:
        ls={}; ss={}
        for seed in SEEDS:
            steps=[("imputer",SimpleImputer(strategy="median",add_indicator=True))]
            if robust: steps.append(("scaler",RobustScaler()))
            steps.append(("model",IsolationForest(n_estimators=100,contamination="auto",random_state=seed,n_jobs=-1)))
            pipe=Pipeline(steps); pipe.fit(X); scores=pipe.score_samples(X); ss[seed]=scores
            for contamination in CONTAMINATIONS: ls[(seed,contamination)]=labels(scores,contamination)
        candidate_labels[name]=ls; candidate_scores[name]=ss
    transformed=Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),("scaler",RobustScaler())]).fit_transform(X)
    lof=LocalOutlierFactor(n_neighbors=35,contamination="auto",n_jobs=-1); lof.fit_predict(transformed); lof_scores=lof.negative_outlier_factor_
    lof_ls={}; lof_ss={s:lof_scores for s in SEEDS}
    for s in SEEDS:
        for c in CONTAMINATIONS: lof_ls[(s,c)]=labels(lof_scores,c)
    candidate_labels["local_outlier_factor_transductive"]=lof_ls; candidate_scores["local_outlier_factor_transductive"]=lof_ss
    summaries={name:summarize(candidate_labels[name],candidate_scores[name]) for name in candidate_labels}
    base_labels=candidate_labels["isolation_forest_baseline"][(SEED,.05)]
    for name in summaries:
        summaries[name]["jaccard_with_baseline_seed342_contamination005"]=jaccard(base_labels,candidate_labels[name][(SEED,.05)])
    base_stability=summaries["isolation_forest_baseline"]["mean_seed_jaccard"]
    robust_stability=summaries["isolation_forest_robust"]["mean_seed_jaccard"]
    passed=robust_stability>base_stability+.02 and summaries["isolation_forest_robust"]["jaccard_with_baseline_seed342_contamination005"]>=.8
    selected="isolation_forest_robust" if passed else "root_isolation_forest_baseline"
    reason=("robust Isolation Forest had materially higher seed stability and strong baseline agreement" if passed else
            f"retained root baseline: robust mean seed Jaccard {robust_stability:.6f} was not > baseline {base_stability:.6f} by 0.02; no labels are available")
    root=pd.read_csv(ROOT/"predictions/task5_test_predictions.csv")
    proposed=pd.DataFrame({"id":ids,"task5":candidate_labels["isolation_forest_robust"][(SEED,.05)]})
    output=proposed if passed else root.copy()
    if len(output)!=25889 or not output.id.equals(ids) or set(output.task5)-{0,1}: raise ValueError("invalid Task 5 output")
    changed=int((output.task5.to_numpy()!=root.task5.to_numpy()).sum())
    metrics={"task":5,"official_metric":"F3 on hidden labels; no labeled validation metric is available locally","seed":SEED,
             "candidate_configurations":{"isolation_forest":{"n_estimators":100,"contamination_for_scores":"auto","preprocessing":["median_imputation","optional_robust_scaling"]},"local_outlier_factor":{"n_neighbors":35,"mode":"transductive negative_outlier_factor_","preprocessing":["median_imputation","robust_scaling"]}},
             "seeds":SEEDS,"contamination_values":CONTAMINATIONS,"features":X.columns.tolist(),"missing_values":int(X.isna().sum().sum()),
             "candidates":summaries,"selected_model":selected,"selection_passed":passed,"selection_reason":reason,
             "changed_test_rows":changed,"selected_anomaly_rate":float(output.task5.mean()),
             "lof_note":"transductive negative_outlier_factor_ used; novelty scoring on fitted samples would be invalid",
             "robust_covariance_omitted":"122 dimensions with missing values and 25,889 rows make covariance assumptions and cost less defensible"}
    (ROUND/"outputs/predictions").mkdir(parents=True,exist_ok=True); (ROUND/"outputs/metrics").mkdir(parents=True,exist_ok=True)
    output.to_csv(ROUND/"outputs/predictions/task5.csv",index=False); (ROUND/"outputs/metrics/task5.json").write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))

if __name__=="__main__": main()
