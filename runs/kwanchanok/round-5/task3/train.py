"""Round 5 Task 3: HGB/ExtraTrees/CatBoost OOF NMAE comparison and blending."""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import catboost
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
ROOT=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(ROOT))
from src.common import SEED, preprocessing
DATA=ROOT/"cpe342-kaggle-main/03_regression"; ROUND=Path(__file__).resolve().parents[1]; MIN_GAIN=.001
CAT_RAW={"loss_function":"MAE","iterations":600,"depth":7,"learning_rate":.05,"l2_leaf_reg":8,"random_seed":SEED,"verbose":False,"allow_writing_files":False,"thread_count":-1}
CAT_LOG={**CAT_RAW,"loss_function":"RMSE"}

def add_features(frame):
    out=frame.drop(columns=["player_id","spending_30d"],errors="ignore").copy(); out["missing_count"]=out.isna().sum(axis=1); out["missing_rate"]=out.isna().mean(axis=1)
    for c in ["historical_spending","prev_month_spending","total_transactions","avg_transaction_value","purchase_frequency","account_age_days"]:
        if c in out and out[c].dropna().ge(0).all() and abs(out[c].skew(skipna=True))>1: out[f"{c}_log1p"]=np.log1p(out[c])
    for a,b in [("prev_month_spending","historical_spending"),("historical_spending","account_age_days"),("total_transactions","account_age_days")]:
        if a in out and b in out: out[f"{a}_over_{b}"]=(out[a]/out[b].abs().replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
    if "purchase_frequency" in out and "avg_transaction_value" in out: out["estimated_purchase_value_rate"]=out.purchase_frequency*out.avg_transaction_value
    return out
def hgb(): return HistGradientBoostingRegressor(loss="absolute_error",max_iter=120,learning_rate=.1,max_leaf_nodes=31,l2_regularization=1.,random_state=SEED)
def extra(): return ExtraTreesRegressor(n_estimators=200,min_samples_leaf=2,max_features=.8,random_state=SEED,n_jobs=-1)
def nmae(y,p): return float(1/(1+np.abs(y-p).sum()/y.sum()))
def summarize(y,rawp,splits,runtime):
    negative=int((rawp<0).sum()); p=np.clip(rawp,0,None); fn=[nmae(y[v],p[v]) for _,v in splits]; fm=[float(mean_absolute_error(y[v],p[v])) for _,v in splits]
    q25,q90=np.quantile(y,[.25,.9]); masks={"low":y<=q25,"medium":(y>q25)&(y<=q90),"high":y>q90}; residual={k:{"rows":int(m.sum()),"mae":float(mean_absolute_error(y[m],p[m])),"median_error":float(np.median(p[m]-y[m])),"mean_error":float(np.mean(p[m]-y[m]))} for k,m in masks.items()}
    return {"oof_nmae":nmae(y,p),"mae":float(mean_absolute_error(y,p)),"fold_nmae":fn,"fold_mae":fm,"fold_nmae_mean":float(np.mean(fn)),"fold_nmae_std":float(np.std(fn,ddof=1)),"fold_mae_mean":float(np.mean(fm)),"fold_mae_std":float(np.std(fm,ddof=1)),
            "negative_predictions_before_clipping":negative,"prediction_quantiles":{str(q):float(np.quantile(p,q)) for q in [0,.25,.5,.75,.9,.99,1]},"residual_ranges":residual,"runtime_seconds":runtime}

def main():
    train=pd.read_csv(DATA/"train.csv"); test=pd.read_csv(DATA/"test.csv"); y=train.spending_30d.to_numpy(float); raw=train.drop(columns=["player_id","spending_30d"]); eng=add_features(train)
    frames={"raw":raw,"engineered":eng}; testframes={"raw":test.drop(columns=["player_id"]),"engineered":add_features(test)}; splits=list(KFold(5,shuffle=True,random_state=SEED).split(raw))
    specs={"hgb_raw_baseline":("hgb","raw",False),"hgb_engineered":("hgb","engineered",False),"extra_trees_raw":("extra","raw",False),"catboost_raw_target":("cat","raw",False),"catboost_log1p_target":("cat","raw",True)}
    results={}; oof={}
    for name,(kind,features,logtarget) in specs.items():
        start=time.perf_counter(); p=np.full(len(y),np.nan); X=frames[features]
        for tr,va in splits:
            fit_y=np.log1p(y[tr]) if logtarget else y[tr]
            if kind=="hgb": m=Pipeline([("prep",preprocessing(X,[])),("model",hgb())])
            elif kind=="extra": m=Pipeline([("prep",preprocessing(X,[])),("model",extra())])
            else: m=CatBoostRegressor(**(CAT_LOG if logtarget else CAT_RAW))
            m.fit(X.iloc[tr],fit_y); fp=m.predict(X.iloc[va]); p[va]=np.expm1(fp) if logtarget else fp
        if not np.isfinite(p).all(): raise ValueError(f"invalid OOF {name}")
        oof[name]=p; results[name]=summarize(y,p,splits,time.perf_counter()-start)
    best_nonlog=max((n for n,s in specs.items() if not s[2]),key=lambda n:results[n]["oof_nmae"]); best_log=max((n for n,s in specs.items() if s[2]),key=lambda n:results[n]["oof_nmae"])
    for w in [.25,.5,.75]:
        name=f"blend_log_weight_{w:.2f}"; start=time.perf_counter(); p=(1-w)*np.clip(oof[best_nonlog],0,None)+w*np.clip(oof[best_log],0,None); oof[name]=p; results[name]=summarize(y,p,splits,time.perf_counter()-start); specs[name]=("blend",(best_nonlog,best_log,w),False)
    baseline="hgb_raw_baseline"; best=max(results,key=lambda n:results[n]["oof_nmae"]); gain=results[best]["oof_nmae"]-results[baseline]["oof_nmae"]
    delta=np.array(results[best]["fold_nmae"])-np.array(results[baseline]["fold_nmae"]); wins=int((delta>=0).sum()); high_stable=results[best]["prediction_quantiles"]["0.99"]<=2*np.quantile(y,.99) and results[best]["prediction_quantiles"]["1"]<=2*y.max()
    passed=best!=baseline and gain>=MIN_GAIN and wins>=3 and high_stable; selected=best if passed else baseline
    def fold_test(candidate,tr):
        kind,info,logtarget=specs[candidate]
        if kind=="blend":
            a,b,w=info; return (1-w)*np.clip(fold_test(a,tr),0,None)+w*np.clip(fold_test(b,tr),0,None)
        X=frames[info]; Xt=testframes[info]; fit_y=np.log1p(y[tr]) if logtarget else y[tr]
        if kind=="hgb": m=Pipeline([("prep",preprocessing(X,[])),("model",hgb())])
        elif kind=="extra": m=Pipeline([("prep",preprocessing(X,[])),("model",extra())])
        else: m=CatBoostRegressor(**(CAT_LOG if logtarget else CAT_RAW))
        m.fit(X.iloc[tr],fit_y); p=m.predict(Xt); return np.expm1(p) if logtarget else p
    proposed=np.clip(np.mean([fold_test(selected,tr) for tr,_ in splits],axis=0),0,None); root=pd.read_csv(ROOT/"predictions/task3_test_predictions.csv"); output=pd.DataFrame({"id":test.player_id,"task3":proposed}) if passed else root.copy(); vals=output.task3.to_numpy(float)
    if len(output)!=25889 or not output.id.equals(test.player_id) or not np.isfinite(vals).all() or (vals<0).any(): raise ValueError("invalid Task 3 output")
    report={"task":3,"official_metric":"NMAE=1/(1+sum(abs(y-pred))/sum(y)); higher is better","seed":SEED,"catboost_version":catboost.__version__,"folds":{"n_splits":5,"shuffle":True,"random_state":SEED},
            "candidate_configurations":{"hgb":{"loss":"absolute_error","max_iter":120,"learning_rate":.1,"max_leaf_nodes":31,"l2_regularization":1.},"extra_trees":{"n_estimators":200,"min_samples_leaf":2,"max_features":.8},"catboost_raw":CAT_RAW,"catboost_log":CAT_LOG,"blend_log_weights":[.25,.5,.75]},
            "feature_sets":{k:list(v.columns) for k,v in frames.items()},"candidates":results,"baseline_candidate":baseline,"historical_root_validation_nmae":.7778654405623588,"best_candidate":best,"selected_model":selected,"selection_passed":passed,
            "selection_rule":{"minimum_gain":MIN_GAIN,"minimum_winning_folds":3,"high_value_limit":"OOF p99 and max <= 2x target"},"selection_reason":f"gain={gain:.6f}, wins/ties={wins}/5, high_stable={high_stable}",
            "selected_blend_weight":specs[selected][1][2] if specs[selected][0]=="blend" else None,"changed_test_rows":int((~np.isclose(vals,root.task3.to_numpy(float),rtol=0,atol=1e-12)).sum())}
    (ROUND/"outputs/predictions").mkdir(parents=True,exist_ok=True); (ROUND/"outputs/metrics").mkdir(parents=True,exist_ok=True); output.to_csv(ROUND/"outputs/predictions/task3.csv",index=False); (ROUND/"outputs/metrics/task3.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
