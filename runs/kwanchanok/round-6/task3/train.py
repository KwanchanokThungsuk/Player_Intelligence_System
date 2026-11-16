"""Task 3 Round 6: OOF regression and two-stage spending experiment."""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import catboost
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
ROOT=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(ROOT))
from src.common import SEED, preprocessing
DATA=ROOT/"cpe342-kaggle-main/03_regression"; ROUND=Path(__file__).resolve().parents[1]; MIN_GAIN=.0005
CAT_RAW={"loss_function":"MAE","iterations":700,"depth":7,"learning_rate":.04,"l2_leaf_reg":8,"random_seed":SEED,"verbose":False,"allow_writing_files":False,"thread_count":-1}
CAT_LOG={**CAT_RAW,"loss_function":"RMSE"}
GATE_GRID=np.linspace(.10,.90,81)

def add_features(frame):
    out=frame.drop(columns=["player_id","spending_30d"],errors="ignore").copy(); out["missing_count"]=out.isna().sum(axis=1); out["missing_rate"]=out.isna().mean(axis=1)
    for c in ["historical_spending","prev_month_spending","total_transactions","avg_transaction_value","purchase_frequency","account_age_days"]:
        if c in out and out[c].dropna().ge(0).all() and abs(out[c].skew(skipna=True))>1: out[f"{c}_log1p"]=np.log1p(out[c])
    for a,b in [("prev_month_spending","historical_spending"),("historical_spending","account_age_days"),("total_transactions","account_age_days"),("purchase_frequency","sessions_per_week")]:
        if a in out and b in out: out[f"{a}_over_{b}"]=(out[a]/out[b].abs().replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
    if "purchase_frequency" in out and "avg_transaction_value" in out: out["estimated_purchase_value_rate"]=out.purchase_frequency*out.avg_transaction_value
    return out
def hgb_r(): return HistGradientBoostingRegressor(loss="absolute_error",max_iter=120,learning_rate=.1,max_leaf_nodes=31,l2_regularization=1.,random_state=SEED)
def hgb_c(): return HistGradientBoostingClassifier(max_iter=120,learning_rate=.1,max_leaf_nodes=31,l2_regularization=1.,random_state=SEED)
def extra(): return ExtraTreesRegressor(n_estimators=240,min_samples_leaf=2,max_features=.8,random_state=SEED,n_jobs=-1)
def nmae(y,p): return float(1/(1+np.abs(y-p).sum()/y.sum()))
def residual_report(y,p):
    p=np.clip(p,0,None); q25,q90=np.quantile(y,[.25,.9]); masks={"zero":y==0,"low":(y>0)&(y<=q25),"medium":(y>q25)&(y<=q90),"high":y>q90,"top5":y>=np.quantile(y,.95),"top1":y>=np.quantile(y,.99)}
    return {k:{"rows":int(m.sum()),"mae":float(mean_absolute_error(y[m],p[m])) if m.any() else None,"median_error":float(np.median(p[m]-y[m])) if m.any() else None,"mean_error":float(np.mean(p[m]-y[m])) if m.any() else None} for k,m in masks.items()}
def summarize(y,rawp,splits,runtime,threshold=None):
    before=np.asarray(rawp); p=np.clip(before,0,None); fn=[nmae(y[v],p[v]) for _,v in splits]; fm=[float(mean_absolute_error(y[v],p[v])) for _,v in splits]
    return {"oof_nmae":nmae(y,p),"oof_mae":float(mean_absolute_error(y,p)),"fold_nmae":fn,"fold_mae":fm,"fold_nmae_mean":float(np.mean(fn)),"fold_nmae_std":float(np.std(fn,ddof=1)),"fold_mae_mean":float(np.mean(fm)),"fold_mae_std":float(np.std(fm,ddof=1)),"negative_before_clipping":int((before<0).sum()),"zero_prediction_rate":float((p==0).mean()),"prediction_quantiles":{str(q):float(np.quantile(p,q)) for q in [0,.25,.5,.75,.9,.95,.99,.995,1]},"residual_quantiles":{str(q):float(np.quantile(p-y,q)) for q in [0,.01,.05,.5,.95,.99,1]},"residual_by_band":residual_report(y,p),"threshold":threshold,"runtime_seconds":runtime}
def direct_fit(kind,X,y,tr,va,target_log=False):
    fit_y=np.log1p(y[tr]) if target_log else y[tr]
    if kind=="hgb": m=Pipeline([("prep",preprocessing(X,[])),("model",hgb_r())])
    elif kind=="extra": m=Pipeline([("prep",preprocessing(X,[])),("model",extra())])
    else: m=CatBoostRegressor(**(CAT_LOG if target_log else CAT_RAW))
    m.fit(X.iloc[tr],fit_y); p=m.predict(X.iloc[va]); return np.expm1(p) if target_log else p
def direct_test(kind,X,Xt,y,tr,target_log=False):
    fit_y=np.log1p(y[tr]) if target_log else y[tr]
    if kind=="hgb": m=Pipeline([("prep",preprocessing(X,[])),("model",hgb_r())])
    elif kind=="extra": m=Pipeline([("prep",preprocessing(X,[])),("model",extra())])
    else: m=CatBoostRegressor(**(CAT_LOG if target_log else CAT_RAW))
    m.fit(X.iloc[tr],fit_y); p=m.predict(Xt); return np.expm1(p) if target_log else p
def two_stage_fold(X,y,tr,va,mode,gate=None):
    clf=Pipeline([("prep",preprocessing(X,[])),("model",hgb_c())]); clf.fit(X.iloc[tr],(y[tr]>0).astype(int)); pp=clf.predict_proba(X.iloc[va])[:,1]
    pos=tr[y[tr]>0]; reg=Pipeline([("prep",preprocessing(X,[])),("model",hgb_r())]); reg.fit(X.iloc[pos],y[pos]); cond=np.clip(reg.predict(X.iloc[va]),0,None)
    if mode=="expected": return pp*cond
    return np.where(pp>=gate,cond,0.)
def two_stage_test(X,Xt,y,tr,mode,gate=None):
    clf=Pipeline([("prep",preprocessing(X,[])),("model",hgb_c())]); clf.fit(X.iloc[tr],(y[tr]>0).astype(int)); pp=clf.predict_proba(Xt)[:,1]; pos=tr[y[tr]>0]
    reg=Pipeline([("prep",preprocessing(X,[])),("model",hgb_r())]); reg.fit(X.iloc[pos],y[pos]); cond=np.clip(reg.predict(Xt),0,None)
    return pp*cond if mode=="expected" else np.where(pp>=gate,cond,0.)
def main():
    train=pd.read_csv(DATA/"train.csv"); test=pd.read_csv(DATA/"test.csv"); y=train.spending_30d.to_numpy(float); raw=train.drop(columns=["player_id","spending_30d"]); eng=add_features(train); frames={"raw":raw,"engineered":eng}; testframes={"raw":test.drop(columns=["player_id"]),"engineered":add_features(test)}
    splits=list(KFold(5,shuffle=True,random_state=SEED).split(raw)); fold_id=np.full(len(y),-1); [fold_id.__setitem__(va,k) for k,(_,va) in enumerate(splits)]
    specs={"hgb_raw_baseline":("direct","hgb","raw",False),"hgb_engineered":("direct","hgb","engineered",False),"catboost_raw_target":("direct","cat","raw",False),"catboost_log1p_target":("direct","cat","raw",True),"extra_trees_raw":("direct","extra","raw",False)}; results={}; oof={}
    for name,(group,kind,features,logtarget) in specs.items():
        st=time.perf_counter(); p=np.full(len(y),np.nan); X=frames[features]
        for tr,va in splits: p[va]=direct_fit(kind,X,y,tr,va,logtarget)
        results[name]=summarize(y,p,splits,time.perf_counter()-st); oof[name]=p
    # Two-stage OOF: expected value and hard gate threshold chosen from OOF only.
    st=time.perf_counter(); pp=np.full(len(y),np.nan); cond=np.full(len(y),np.nan)
    for tr,va in splits:
        clf=Pipeline([("prep",preprocessing(eng,[])),("model",hgb_c())]); clf.fit(eng.iloc[tr],(y[tr]>0).astype(int)); pp[va]=clf.predict_proba(eng.iloc[va])[:,1]; pos=tr[y[tr]>0]; reg=Pipeline([("prep",preprocessing(eng,[])),("model",hgb_r())]); reg.fit(eng.iloc[pos],y[pos]); cond[va]=np.clip(reg.predict(eng.iloc[va]),0,None)
    expected=pp*cond; oof["two_stage_expected"]=expected; results["two_stage_expected"]=summarize(y,expected,splits,time.perf_counter()-st)
    gate_scores=[nmae(y,np.where(pp>=g,cond,0)) for g in GATE_GRID]; gate= float(GATE_GRID[int(np.argmax(gate_scores))]); hard=np.where(pp>=gate,cond,0.); oof["two_stage_hard_gate"]=hard; results["two_stage_hard_gate"]=summarize(y,hard,splits,time.perf_counter()-st,gate)
    best_raw=max([n for n,s in specs.items() if not s[3]],key=lambda n:results[n]["oof_nmae"]); best_log="catboost_log1p_target"
    for w in [.25,.5,.75]:
        name=f"blend_log_weight_{w:.2f}"; p=(1-w)*np.clip(oof[best_raw],0,None)+w*np.clip(oof[best_log],0,None); oof[name]=p; results[name]=summarize(y,p,splits,0.0)
    baseline="hgb_raw_baseline"; best=max(results,key=lambda n:results[n]["oof_nmae"]); gain=results[best]["oof_nmae"]-results[baseline]["oof_nmae"]; delta=np.array(results[best]["fold_nmae"])-np.array(results[baseline]["fold_nmae"]); wins=int((delta>=0).sum()); severe=bool((delta<-.02).any()); high_stable=bool(results[best]["prediction_quantiles"]["0.99"]<=2*np.quantile(y,.99) and results[best]["prediction_quantiles"]["1"]<=2*y.max()); passed=bool(best!=baseline and gain>=MIN_GAIN and wins>=3 and not severe and high_stable); selected=best if passed else baseline
    def final_fold(name,tr):
        if name.startswith("blend_log_weight"):
            w=float(name.rsplit("_",1)[1]); return (1-w)*np.clip(final_fold(best_raw,tr),0,None)+w*np.clip(final_fold(best_log,tr),0,None)
        if name=="two_stage_expected": return two_stage_test(eng,testframes["engineered"],y,tr,"expected")
        if name=="two_stage_hard_gate": return two_stage_test(eng,testframes["engineered"],y,tr,"hard",gate)
        group,kind,features,logtarget=specs[name]; return direct_test(kind,frames[features],testframes[features],y,tr,logtarget)
    root=pd.read_csv(ROOT/"predictions/task3_test_predictions.csv"); proposed=np.clip(np.mean([final_fold(selected,tr) for tr,_ in splits],axis=0),0,None); output=pd.DataFrame({"id":test.player_id,"task3":proposed}) if passed else root.copy(); vals=output.task3.to_numpy(float)
    if len(output)!=25889 or not output.id.equals(test.player_id) or not np.isfinite(vals).all() or (vals<0).any(): raise ValueError("invalid Task 3 output")
    target_stats={"count":int(len(y)),"min":float(y.min()),"max":float(y.max()),"mean":float(y.mean()),"median":float(np.median(y)),"std":float(y.std()),"quantiles":{str(q):float(np.quantile(y,q)) for q in [0,.25,.5,.75,.9,.95,.99,.995,1]},"zero_count":int((y==0).sum()),"zero_rate":float((y==0).mean()),"near_zero_le_1_count":int((y<=1).sum()),"near_zero_le_1_rate":float((y<=1).mean()),"skewness":float(pd.Series(y).skew()),"two_stage_justified":bool((y==0).mean()>.1)}
    num=raw.select_dtypes(include=np.number).columns; diffs=[]
    for c in num:
        a=train[c].mean(); b=test[c].mean(); sd=train[c].std(); diffs.append({"feature":c,"train_mean":float(a),"test_mean":float(b),"standardized_mean_difference":float((b-a)/(sd+1e-12))})
    diffs=sorted(diffs,key=lambda d:abs(d["standardized_mean_difference"]),reverse=True)[:10]
    duplicate_conflicts=0; dup_rows=0
    hashes=pd.util.hash_pandas_object(raw,index=False); groups=pd.DataFrame({"h":hashes,"y":y}).groupby("h")["y"]; duplicate_conflicts=int(sum(s.nunique()>1 for _,s in groups)); dup_rows=int(sum(len(s) for _,s in groups if len(s)>1))
    report={"task":3,"official_metric":"NMAE=1/(1+sum(abs(y-pred))/sum(y)); higher is better","seed":SEED,"catboost_version":catboost.__version__,"data_analysis":target_stats,"top_train_test_mean_differences":diffs,"duplicate_feature_rows":dup_rows,"duplicate_feature_groups_with_conflicting_targets":duplicate_conflicts,"folds":{"n_splits":5,"shuffle":True,"random_state":SEED,"fold_assignment_sha256":hashlib.sha256(fold_id.tobytes()).hexdigest(),"fold_sizes":np.bincount(fold_id).tolist()},"candidate_configurations":{"hgb":{"loss":"absolute_error","max_iter":120,"learning_rate":.1,"max_leaf_nodes":31,"l2_regularization":1.},"extra_trees":{"n_estimators":240,"min_samples_leaf":2,"max_features":.8},"catboost_raw":CAT_RAW,"catboost_log":CAT_LOG,"gate_grid":[.1,.9,81],"blend_log_weights":[.25,.5,.75]},"feature_sets":{k:list(v.columns) for k,v in frames.items()},"candidates":results,"baseline_candidate":baseline,"round5_comparable_hgb_nmae":.7762036900104093,"best_candidate":best,"selected_model":selected,"selection_passed":passed,"selection_rule":{"minimum_gain":MIN_GAIN,"minimum_winning_folds":3,"severe_fold_delta":-.02,"high_value_limit":"OOF p99 and max <= 2x target"},"selection_reason":f"gain={gain:.6f}, wins/ties={wins}/5, severe={severe}, high_stable={high_stable}","selected_gate_threshold":gate if selected=="two_stage_hard_gate" else None,"changed_vs_root":int((~np.isclose(vals,root.task3.to_numpy(float),rtol=0,atol=1e-12)).sum()),"changed_vs_round5":int((~np.isclose(vals,pd.read_csv(ROOT/"runs/kwanchanok/round-5/outputs/predictions/task3.csv").task3.to_numpy(float),rtol=0,atol=1e-12)).sum())}
    (ROUND/"outputs/predictions").mkdir(parents=True,exist_ok=True); (ROUND/"outputs/metrics").mkdir(parents=True,exist_ok=True); output.to_csv(ROUND/"outputs/predictions/task3.csv",index=False); (ROUND/"outputs/metrics/task3.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
