"""Round 5 Task 1: HGB/CatBoost OOF Binary-F2 comparison and blending."""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import catboost
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
ROOT=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(ROOT))
from src.common import SEED, preprocessing
DATA=ROOT/"cpe342-kaggle-main/01_binary_classification"; ROUND=Path(__file__).resolve().parents[1]
GRID=np.linspace(.05,.95,361); MIN_GAIN=.003

def add_features(frame):
    out=frame.drop(columns=["player_id","is_cheater"],errors="ignore").copy(); out["missing_count"]=out.isna().sum(axis=1); out["missing_rate"]=out.isna().mean(axis=1)
    for c in ["reports_received","headshot_percentage","accuracy_score","account_age_days"]:
        if c in out and out[c].dropna().ge(0).all() and abs(out[c].skew(skipna=True))>1: out[f"{c}_log1p"]=np.log1p(out[c])
    for a,b in [("accuracy_score","reaction_time_ms"),("headshot_percentage","accuracy_score"),("reports_received","account_age_days")]:
        if a in out and b in out: out[f"{a}_over_{b}"]=(out[a]/out[b].abs().replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
    return out

def hgb(): return HistGradientBoostingClassifier(max_iter=120,learning_rate=.1,max_leaf_nodes=31,l2_regularization=1.,random_state=SEED)
CAT_CFG={"loss_function":"Logloss","eval_metric":"F:beta=2","iterations":400,"depth":6,"learning_rate":.06,"l2_leaf_reg":8,"random_seed":SEED,"verbose":False,"allow_writing_files":False,"thread_count":-1}
def threshold(y,p):
    s=np.array([fbeta_score(y,p>=t,beta=2,average="binary",pos_label=1,zero_division=0) for t in GRID]); tied=GRID[np.isclose(s,s.max(),atol=1e-12,rtol=0)]; return float(sorted(tied,key=lambda x:(abs(x-.5),x))[0])
def metrics(y,p,t,splits,runtime):
    q=(p>=t).astype(int); folds=[float(fbeta_score(y[v],q[v],beta=2,zero_division=0)) for _,v in splits]
    return {"oof_binary_f2":float(fbeta_score(y,q,beta=2,zero_division=0)),"threshold":t,"fold_binary_f2":folds,"fold_mean":float(np.mean(folds)),"fold_std":float(np.std(folds,ddof=1)),
            "precision":float(precision_score(y,q,zero_division=0)),"recall":float(recall_score(y,q,zero_division=0)),"macro_f2_diagnostic":float(fbeta_score(y,q,beta=2,average="macro",zero_division=0)),
            "confusion_matrix":confusion_matrix(y,q).tolist(),"predicted_positive_rate":float(q.mean()),"runtime_seconds":runtime}
def fit_predict(kind,X,y,tr,va):
    if kind=="hgb":
        pipe=Pipeline([("prep",preprocessing(X,[])),("model",hgb())]); pipe.fit(X.iloc[tr],y[tr]); return pipe.predict_proba(X.iloc[va])[:,1]
    model=CatBoostClassifier(**CAT_CFG); model.fit(X.iloc[tr],y[tr]); return model.predict_proba(X.iloc[va])[:,1]

def main():
    train=pd.read_csv(DATA/"train.csv").dropna(subset=["is_cheater"]).copy(); test=pd.read_csv(DATA/"test.csv"); y=train.is_cheater.astype(int).to_numpy()
    frames={"raw":train.drop(columns=["player_id","is_cheater"]),"engineered":add_features(train)}; testframes={"raw":test.drop(columns=["player_id"]),"engineered":add_features(test)}
    splits=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(frames["raw"],y)); specs={"hgb_raw":("hgb","raw"),"hgb_engineered":("hgb","engineered"),"catboost_raw":("cat","raw"),"catboost_engineered":("cat","engineered")}
    results={}; oof={}
    for name,(kind,features) in specs.items():
        start=time.perf_counter(); p=np.full(len(y),np.nan)
        for tr,va in splits: p[va]=fit_predict(kind,frames[features],y,tr,va)
        if not np.isfinite(p).all(): raise ValueError(f"invalid OOF {name}")
        oof[name]=p; t=threshold(y,p); results[name]=metrics(y,p,t,splits,time.perf_counter()-start)
    best_hgb=max((n for n in specs if n.startswith("hgb")),key=lambda n:results[n]["oof_binary_f2"]); best_cat=max((n for n in specs if n.startswith("cat")),key=lambda n:results[n]["oof_binary_f2"])
    for w in [.25,.5,.75]:
        name=f"blend_cat_weight_{w:.2f}"; start=time.perf_counter(); p=(1-w)*oof[best_hgb]+w*oof[best_cat]; oof[name]=p; t=threshold(y,p); results[name]=metrics(y,p,t,splits,time.perf_counter()-start); specs[name]=("blend",(best_hgb,best_cat,w))
    baseline="hgb_raw"; best=max(results,key=lambda n:results[n]["oof_binary_f2"]); gain=results[best]["oof_binary_f2"]-results[baseline]["oof_binary_f2"]
    fold_delta=np.array(results[best]["fold_binary_f2"])-np.array(results[baseline]["fold_binary_f2"]); wins=int((fold_delta>=0).sum()); catastrophic=bool((fold_delta<-.02).any())
    passed=best!=baseline and gain>=MIN_GAIN and wins>=3 and not catastrophic; selected=best if passed else baseline
    test_probs=[]
    def fold_test(candidate,tr):
        kind,info=specs[candidate]
        if kind=="hgb":
            X=frames[info]; pipe=Pipeline([("prep",preprocessing(X,[])),("model",hgb())]); pipe.fit(X.iloc[tr],y[tr]); return pipe.predict_proba(testframes[info])[:,1]
        if kind=="cat":
            X=frames[info]; m=CatBoostClassifier(**CAT_CFG); m.fit(X.iloc[tr],y[tr]); return m.predict_proba(testframes[info])[:,1]
        h,c,w=info; return (1-w)*fold_test(h,tr)+w*fold_test(c,tr)
    for tr,_ in splits: test_probs.append(fold_test(selected,tr))
    proposed=(np.mean(test_probs,axis=0)>=results[selected]["threshold"]).astype(int); root=pd.read_csv(ROOT/"predictions/task1_test_predictions.csv")
    output=pd.DataFrame({"id":test.player_id,"task1":proposed}) if passed else root.copy(); changed=int((output.task1.to_numpy()!=root.task1.to_numpy()).sum())
    if len(output)!=25889 or not output.id.equals(test.player_id) or set(output.task1)-{0,1}: raise ValueError("invalid Task 1 output")
    report={"task":1,"official_metric":"Binary F2 class 1","seed":SEED,"catboost_version":catboost.__version__,"folds":{"n_splits":5,"shuffle":True,"random_state":SEED},"threshold_grid":{"start":.05,"stop":.95,"count":361},
            "candidate_configurations":{"hgb":{"max_iter":120,"learning_rate":.1,"max_leaf_nodes":31,"l2_regularization":1.},"catboost":CAT_CFG,"blend_cat_weights":[.25,.5,.75]},"feature_sets":{k:list(v.columns) for k,v in frames.items()},
            "candidates":results,"baseline_candidate":baseline,"historical_root_validation_f2":.835566872152238,"best_candidate":best,"selected_model":selected,"selection_passed":passed,
            "selection_rule":{"minimum_gain":MIN_GAIN,"minimum_winning_folds":3,"catastrophic_fold_delta":-.02},"selection_reason":f"gain={gain:.6f}, wins/ties={wins}/5, catastrophic={catastrophic}",
            "selected_threshold":results[selected]["threshold"],"selected_blend_weight":specs[selected][1][2] if specs[selected][0]=="blend" else None,"changed_test_rows":changed,"test_positive_rate":float(output.task1.mean())}
    (ROUND/"outputs/predictions").mkdir(parents=True,exist_ok=True); (ROUND/"outputs/metrics").mkdir(parents=True,exist_ok=True); output.to_csv(ROUND/"outputs/predictions/task1.csv",index=False); (ROUND/"outputs/metrics/task1.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
