"""Round 4 Task 3: five-fold OOF NMAE regression comparison."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
ROOT=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(ROOT))
from src.common import SEED, preprocessing
DATA=ROOT/"cpe342-kaggle-main/03_regression"; ROUND=Path(__file__).resolve().parents[1]

def model(kind):
    if kind=="hgb_abs": return HistGradientBoostingRegressor(loss="absolute_error",max_iter=120,learning_rate=.1,max_leaf_nodes=31,l2_regularization=1.,random_state=SEED)
    if kind=="hgb_sq": return HistGradientBoostingRegressor(loss="squared_error",max_iter=120,learning_rate=.1,max_leaf_nodes=31,l2_regularization=1.,random_state=SEED)
    return ExtraTreesRegressor(n_estimators=180,min_samples_leaf=2,max_features=.8,random_state=SEED,n_jobs=-1)

def nmae(y,p):
    denom=float(np.sum(y)); return float(1/(1+np.abs(y-p).sum()/denom)) if denom>0 else None

def summarize(y,p,splits):
    fold=[nmae(y[v],p[v]) for _,v in splits]
    qs=[0,.25,.5,.75,.9,.99,1]
    return {"oof_nmae":nmae(y,p),"mae":float(mean_absolute_error(y,p)),"fold_scores":fold,"fold_mean":float(np.mean(fold)),"fold_std":float(np.std(fold,ddof=1)),
            "target_quantiles":{str(q):float(np.quantile(y,q)) for q in qs},"prediction_quantiles":{str(q):float(np.quantile(p,q)) for q in qs}}

def main():
    train=pd.read_csv(DATA/"train.csv"); test=pd.read_csv(DATA/"test.csv"); y=train.spending_30d.to_numpy(float)
    X=train.drop(columns=["player_id","spending_30d"]); Xt=test.drop(columns=["player_id"]); splits=list(KFold(5,shuffle=True,random_state=SEED).split(X))
    specs={"hgb_absolute_baseline":("hgb_abs",False),"hgb_squared_raw":("hgb_sq",False),"extra_trees_raw":("extra",False),
           "hgb_squared_log1p_target":("hgb_sq",True),"extra_trees_log1p_target":("extra",True)}
    results={}; oofs={}
    for name,(kind,log_target) in specs.items():
        p=np.full(len(y),np.nan)
        for tr,va in splits:
            pipe=Pipeline([("prep",preprocessing(X,[])),("model",model(kind))]); fit_y=np.log1p(y[tr]) if log_target else y[tr]
            pipe.fit(X.iloc[tr],fit_y); fold_p=pipe.predict(X.iloc[va]); p[va]=np.expm1(fold_p) if log_target else fold_p
        p=np.clip(p,0,None); results[name]=summarize(y,p,splits); oofs[name]=p
    baseline="hgb_absolute_baseline"; best=max(results,key=lambda k:results[k]["oof_nmae"])
    wins=sum(a>=b for a,b in zip(results[best]["fold_scores"],results[baseline]["fold_scores"]))
    passed=best!=baseline and results[best]["oof_nmae"]>results[baseline]["oof_nmae"] and wins>=3; selected=best if passed else baseline
    reason=(f"{best} improved OOF NMAE and won/tied {wins}/5 folds" if passed else f"retained baseline; best={best}, wins/ties={wins}/5, gain={results[best]['oof_nmae']-results[baseline]['oof_nmae']:.6f}")
    kind,log_target=specs[selected]; preds=[]
    for tr,_ in splits:
        pipe=Pipeline([("prep",preprocessing(X,[])),("model",model(kind))]); pipe.fit(X.iloc[tr],np.log1p(y[tr]) if log_target else y[tr]); p=pipe.predict(Xt); preds.append(np.expm1(p) if log_target else p)
    proposed=np.clip(np.mean(preds,axis=0),0,None); root=pd.read_csv(ROOT/"predictions/task3_test_predictions.csv")
    output=pd.DataFrame({"id":test.player_id,"task3":proposed}) if passed else root.copy()
    vals=output.task3.to_numpy(float)
    if len(output)!=25889 or not output.id.equals(test.player_id) or not np.isfinite(vals).all() or (vals<0).any(): raise ValueError("invalid Task 3 output")
    changed=int((~np.isclose(vals,root.task3.to_numpy(float),rtol=0,atol=1e-12)).sum())
    metrics={"task":3,"official_metric":"NMAE = 1/(1 + sum(abs(y-pred))/sum(y)); higher is better","seed":SEED,
             "candidate_configurations":{"hgb":{"max_iter":120,"learning_rate":.1,"max_leaf_nodes":31,"l2_regularization":1.,"random_state":SEED,"losses":["absolute_error","squared_error"]},"extra_trees":{"n_estimators":180,"min_samples_leaf":2,"max_features":.8,"random_state":SEED},"target_modes":["raw","log1p_then_expm1"]},
             "fold_configuration":{"n_splits":5,"shuffle":True,"random_state":SEED},"candidates":results,"baseline_candidate":baseline,
             "root_validation_nmae":0.7778654405623588,"best_candidate":best,"selected_model":selected,"selection_passed":passed,
             "selection_reason":reason,"changed_test_rows":changed,"test_prediction_quantiles":{str(q):float(np.quantile(vals,q)) for q in [0,.25,.5,.75,.9,.99,1]},
             "catboost_omitted":"not installed; no dependency was added"}
    (ROUND/"outputs/predictions").mkdir(parents=True,exist_ok=True); (ROUND/"outputs/metrics").mkdir(parents=True,exist_ok=True)
    output.to_csv(ROUND/"outputs/predictions/task3.csv",index=False); (ROUND/"outputs/metrics/task3.json").write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))

if __name__=="__main__": main()
