"""Round 5 Task 2: HGB/CatBoost OOF Macro-F1 comparison and blending."""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import catboost
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
ROOT=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(ROOT))
from src.common import SEED, preprocessing
DATA=ROOT/"cpe342-kaggle-main/02_multiclass_classification"; ROUND=Path(__file__).resolve().parents[1]; MIN_GAIN=.003
CAT_CFG={"loss_function":"MultiClass","eval_metric":"TotalF1:average=Macro","iterations":500,"depth":7,"learning_rate":.06,"l2_leaf_reg":8,"random_seed":SEED,"verbose":False,"allow_writing_files":False,"thread_count":-1}

def add_features(frame):
    out=frame.drop(columns=["player_id","segment"],errors="ignore").copy(); out["missing_count"]=out.isna().sum(axis=1); out["missing_rate"]=out.isna().mean(axis=1)
    for c in ["total_spending_thb","avg_monthly_spending","spending_frequency","total_playtime_hours","friend_count"]:
        if c in out and out[c].dropna().ge(0).all() and abs(out[c].skew(skipna=True))>1: out[f"{c}_log1p"]=np.log1p(out[c])
    for a,b in [("total_spending_thb","account_age_days"),("avg_monthly_spending","play_frequency"),("team_play_percentage","friend_count")]:
        if a in out and b in out: out[f"{a}_over_{b}"]=(out[a]/out[b].abs().replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
    return out
def hgb(): return HistGradientBoostingClassifier(max_iter=120,learning_rate=.1,max_leaf_nodes=31,l2_regularization=1.,random_state=SEED)
def cat_frame(X):
    out=X.copy(); cats=out.select_dtypes(exclude=np.number).columns.tolist()
    for c in cats: out[c]=out[c].fillna("__MISSING__").astype(str)
    return out,cats
def summarize(y,pred,splits,runtime):
    fm=[float(f1_score(y[v],pred[v],average="macro",zero_division=0)) for _,v in splits]; fw=[float(f1_score(y[v],pred[v],average="weighted",zero_division=0)) for _,v in splits]
    pr,rc,f1,_=precision_recall_fscore_support(y,pred,labels=[0,1,2,3],zero_division=0)
    return {"oof_macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),"oof_weighted_f1":float(f1_score(y,pred,average="weighted",zero_division=0)),
            "fold_macro_f1":fm,"fold_weighted_f1":fw,"fold_macro_mean":float(np.mean(fm)),"fold_macro_std":float(np.std(fm,ddof=1)),"fold_weighted_mean":float(np.mean(fw)),"fold_weighted_std":float(np.std(fw,ddof=1)),
            "per_class":{"precision":pr.tolist(),"recall":rc.tolist(),"f1":f1.tolist()},"confusion_matrix":confusion_matrix(y,pred,labels=[0,1,2,3]).tolist(),
            "predicted_class_distribution":np.bincount(pred,minlength=4).tolist(),"true_class_distribution":np.bincount(y,minlength=4).tolist(),"runtime_seconds":runtime}

def main():
    train=pd.read_csv(DATA/"train.csv"); test=pd.read_csv(DATA/"test.csv"); y=train.segment.astype(int).to_numpy(); raw=train.drop(columns=["player_id","segment"]); eng=add_features(train)
    frames={"raw":raw,"engineered":eng}; testframes={"raw":test.drop(columns=["player_id"]),"engineered":add_features(test)}; splits=list(StratifiedKFold(5,shuffle=True,random_state=SEED).split(raw,y))
    specs={"hgb_raw":("hgb","raw"),"hgb_engineered":("hgb","engineered"),"catboost_raw":("cat","raw"),"catboost_engineered":("cat","engineered")}; results={}; oof={}
    for name,(kind,features) in specs.items():
        start=time.perf_counter(); p=np.zeros((len(y),4)); X=frames[features]
        for tr,va in splits:
            if kind=="hgb":
                cats=X.select_dtypes(exclude=np.number).columns.tolist(); m=Pipeline([("prep",preprocessing(X,cats)),("model",hgb())]); m.fit(X.iloc[tr],y[tr]); p[va]=m.predict_proba(X.iloc[va])
            else:
                Xc,cats=cat_frame(X); m=CatBoostClassifier(**CAT_CFG); m.fit(Xc.iloc[tr],y[tr],cat_features=cats); p[va]=m.predict_proba(Xc.iloc[va])
        if not np.isfinite(p).all(): raise ValueError(f"invalid OOF {name}")
        oof[name]=p; results[name]=summarize(y,p.argmax(1),splits,time.perf_counter()-start)
    best_hgb=max((n for n in specs if n.startswith("hgb")),key=lambda n:results[n]["oof_macro_f1"]); best_cat=max((n for n in specs if n.startswith("cat")),key=lambda n:results[n]["oof_macro_f1"])
    for w in [.25,.5,.75]:
        name=f"blend_cat_weight_{w:.2f}"; start=time.perf_counter(); p=(1-w)*oof[best_hgb]+w*oof[best_cat]; oof[name]=p; results[name]=summarize(y,p.argmax(1),splits,time.perf_counter()-start); specs[name]=("blend",(best_hgb,best_cat,w))
    baseline="hgb_raw"; best=max(results,key=lambda n:results[n]["oof_macro_f1"]); gain=results[best]["oof_macro_f1"]-results[baseline]["oof_macro_f1"]
    delta=np.array(results[best]["fold_macro_f1"])-np.array(results[baseline]["fold_macro_f1"]); wins=int((delta>=0).sum()); minority_ok=min(results[best]["per_class"]["f1"])>=min(results[baseline]["per_class"]["f1"])-.03
    passed=best!=baseline and gain>=MIN_GAIN and wins>=3 and minority_ok; selected=best if passed else baseline
    def fold_test(candidate,tr):
        kind,info=specs[candidate]
        if kind=="hgb":
            X=frames[info]; cats=X.select_dtypes(exclude=np.number).columns.tolist(); m=Pipeline([("prep",preprocessing(X,cats)),("model",hgb())]); m.fit(X.iloc[tr],y[tr]); return m.predict_proba(testframes[info])
        if kind=="cat":
            X,cats=cat_frame(frames[info]); Xt,_=cat_frame(testframes[info]); m=CatBoostClassifier(**CAT_CFG); m.fit(X.iloc[tr],y[tr],cat_features=cats); return m.predict_proba(Xt)
        h,c,w=info; return (1-w)*fold_test(h,tr)+w*fold_test(c,tr)
    proposed=np.mean([fold_test(selected,tr) for tr,_ in splits],axis=0).argmax(1); root=pd.read_csv(ROOT/"predictions/task2_test_predictions.csv"); output=pd.DataFrame({"id":test.player_id,"task2":proposed}) if passed else root.copy()
    if len(output)!=25889 or not output.id.equals(test.player_id) or set(output.task2)-{0,1,2,3}: raise ValueError("invalid Task 2 output")
    report={"task":2,"official_metric_assumption":"Macro F1 primary; official PDF says multiclass F1 without averaging","seed":SEED,"catboost_version":catboost.__version__,"folds":{"n_splits":5,"shuffle":True,"random_state":SEED},
            "candidate_configurations":{"hgb":{"max_iter":120,"learning_rate":.1,"max_leaf_nodes":31,"l2_regularization":1.},"catboost":CAT_CFG,"blend_cat_weights":[.25,.5,.75]},"feature_sets":{k:list(v.columns) for k,v in frames.items()},"candidates":results,
            "baseline_candidate":baseline,"historical_root_validation_macro_f1":.7491371752784082,"best_candidate":best,"selected_model":selected,"selection_passed":passed,
            "selection_rule":{"minimum_gain":MIN_GAIN,"minimum_winning_folds":3,"minority_f1_max_drop":.03},"selection_reason":f"gain={gain:.6f}, wins/ties={wins}/5, minority_ok={minority_ok}",
            "selected_blend_weight":specs[selected][1][2] if specs[selected][0]=="blend" else None,"changed_test_rows":int((output.task2.to_numpy()!=root.task2.to_numpy()).sum())}
    (ROUND/"outputs/predictions").mkdir(parents=True,exist_ok=True); (ROUND/"outputs/metrics").mkdir(parents=True,exist_ok=True); output.to_csv(ROUND/"outputs/predictions/task2.csv",index=False); (ROUND/"outputs/metrics/task2.json").write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
