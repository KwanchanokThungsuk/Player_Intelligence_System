"""Build the primary Round 6 submission and diagnostic root-base variant."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]; ROUND=Path(__file__).resolve().parent; N=25889; COLS=["id","task1","task2","task3","task4","task5"]
FOLDERS={1:"01_binary_classification",2:"02_multiclass_classification",3:"03_regression",4:"04_image_classification",5:"05_anomaly_detection"}
def delimiter(p): return csv.Sniffer().sniff(p.read_text()[:4096],delimiters=",|\t;").delimiter
def readp(task,path):
    d=pd.read_csv(path); t=pd.read_csv(ROOT/"cpe342-kaggle-main"/FOLDERS[task]/"test.csv"); ic="id" if "id" in d else "player_id"; tc="id" if "id" in t else "player_id"
    if d.columns.tolist()!=[ic,f"task{task}"] or len(d)!=N or d[ic].isna().any() or d[ic].astype(str).str.strip().eq("").any() or d[ic].duplicated().any(): raise ValueError(f"Task {task} invalid prediction file")
    if not d[ic].astype(str).reset_index(drop=True).equals(t[tc].astype(str).reset_index(drop=True)): raise ValueError(f"Task {task} order mismatch")
    return d.reset_index(drop=True)
def validate(d,sample,classes):
    if d.shape!=(N,6) or d.columns.tolist()!=COLS or not d.id.astype(str).equals(sample.id.astype(str)): raise ValueError("submission shape/order failure")
    if d.isna().any().any() or d.id.astype(str).str.strip().eq("").any() or d.id.duplicated().any(): raise ValueError("submission IDs/null failure")
    if set(d.task1)-{0,1} or set(d.task2)-{0,1,2,3} or set(d.task5)-{0,1}: raise ValueError("classification domain failure")
    v=pd.to_numeric(d.task3,errors="coerce").to_numpy()
    if not np.isfinite(v).all() or (v<0).any() or set(d.task4.astype(str))-classes: raise ValueError("Task 3/4 domain failure")
def main():
    sp=ROOT/"cpe342-kaggle-main/sample_submission.csv"; s=delimiter(sp); sample=pd.read_csv(sp,sep=s)
    root={t:readp(t,ROOT/f"predictions/task{t}_test_predictions.csv") for t in [1,3,4,5]}; r5t2=readp(2,ROOT/"runs/kwanchanok/round-5/outputs/predictions/task2.csv"); root2=readp(2,ROOT/"predictions/task2_test_predictions.csv")
    diff2=int((r5t2.task2.to_numpy()!=root2.task2.to_numpy()).sum())
    if diff2!=1593: raise ValueError(f"Round 5 Task 2 differs by {diff2}, expected 1593")
    r6t3=readp(3,ROUND/"outputs/predictions/task3.csv"); tr4=pd.read_csv(ROOT/"cpe342-kaggle-main/04_image_classification/train.csv"); target="game_title" if "game_title" in tr4 else "class"; classes=set(tr4[target].astype(str))
    chosen_primary={1:root[1],2:r5t2,3:r6t3,4:root[4],5:root[5]}; chosen_diag={1:root[1],2:root2,3:r6t3,4:root[4],5:root[5]}
    root_out=pd.DataFrame({"id":sample.id,**{f"task{t}":(root2 if t==2 else root[t])[f"task{t}"].to_numpy() for t in [1,2,3,4,5]}}); validate(root_out,sample,classes)
    outdir=ROUND/"outputs/submissions"; outdir.mkdir(parents=True,exist_ok=True); report={}
    for name,chosen in [("task3_on_round5_best",chosen_primary),("task3_only_root_base",chosen_diag)]:
        out=pd.DataFrame({"id":sample.id,**{f"task{t}":chosen[t][f"task{t}"].to_numpy() for t in [1,2,3,4,5]}}); validate(out,sample,classes); p=outdir/f"{name}.csv"; out.to_csv(p,sep=s,index=False); back=pd.read_csv(p,sep=s); validate(back,sample,classes)
        report[name]={"path":str(p),"shape":list(back.shape),"delimiter":s,"round5_task2_changed_rows":diff2,"changed_vs_root":{"task1":int((out.task1.to_numpy()!=root[1].task1.to_numpy()).sum()),"task2":int((out.task2.to_numpy()!=root2.task2.to_numpy()).sum()),"task3":int((out.task3.to_numpy()!=root[3].task3.to_numpy()).sum()),"task4":int((out.task4.to_numpy()!=root[4].task4.to_numpy()).sum()),"task5":int((out.task5.to_numpy()!=root[5].task5.to_numpy()).sum())},"readback_valid":True,"task4_sha256":hashlib.sha256(out.task4.astype(str).to_csv(index=False).encode()).hexdigest()}
    report["task4_identical_to_root"]=all(v["changed_vs_root"]["task4"]==0 for v in report.values() if isinstance(v,dict)); report["task2_round5_retained_in_primary"]=report["task3_on_round5_best"]["changed_vs_root"]["task2"]==1593; print(json.dumps(report,indent=2))
if __name__=="__main__": main()
