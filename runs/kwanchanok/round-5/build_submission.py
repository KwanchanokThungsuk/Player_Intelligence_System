"""Build and validate five Round 5 submissions with equivalence diagnostics."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]; ROUND=Path(__file__).resolve().parent; N=25889
COLS=["id","task1","task2","task3","task4","task5"]
FOLDERS={1:"01_binary_classification",2:"02_multiclass_classification",3:"03_regression",4:"04_image_classification",5:"05_anomaly_detection"}
def sep(path): return csv.Sniffer().sniff(path.read_text()[:4096],delimiters=",|\t;").delimiter
def read_pred(task,path):
    d=pd.read_csv(path); test=pd.read_csv(ROOT/"cpe342-kaggle-main"/FOLDERS[task]/"test.csv"); idcol="id" if "id" in d else "player_id"; tid="id" if "id" in test else "player_id"
    if d.columns.tolist()!=[idcol,f"task{task}"] or len(d)!=N or d[idcol].isna().any() or d[idcol].astype(str).str.strip().eq("").any() or d[idcol].duplicated().any(): raise ValueError(f"Task {task} invalid prediction schema/IDs")
    if not d[idcol].astype(str).reset_index(drop=True).equals(test[tid].astype(str).reset_index(drop=True)): raise ValueError(f"Task {task} test order mismatch")
    return d.reset_index(drop=True)
def validate(d,sample,classes):
    if d.shape!=(N,6) or d.columns.tolist()!=COLS or not d.id.astype(str).equals(sample.id.astype(str)): raise ValueError("submission shape/ID failure")
    if d.isna().any().any() or set(d.task1)-{0,1} or set(d.task5)-{0,1} or set(d.task2)-{0,1,2,3}: raise ValueError("null/domain failure")
    v=pd.to_numeric(d.task3,errors="coerce").to_numpy();
    if not np.isfinite(v).all() or (v<0).any() or set(d.task4.astype(str))-classes: raise ValueError("Task 3/4 domain failure")
def value_hash(values): return hashlib.sha256(pd.DataFrame(values).to_csv(index=False).encode()).hexdigest()
def main():
    sample_path=ROOT/"cpe342-kaggle-main/sample_submission.csv"; delimiter=sep(sample_path); sample=pd.read_csv(sample_path,sep=delimiter)
    root={t:read_pred(t,ROOT/f"predictions/task{t}_test_predictions.csv") for t in range(1,6)}; new={t:read_pred(t,ROUND/f"outputs/predictions/task{t}.csv") for t in [1,2,3,5]}; new[4]=root[4]
    if not new[4].equals(root[4]): raise AssertionError("Task 4 changed")
    train4=pd.read_csv(ROOT/"cpe342-kaggle-main/04_image_classification/train.csv"); target="game_title" if "game_title" in train4 else "class"; classes=set(train4[target].astype(str))
    baseline=pd.DataFrame({"id":sample.id,**{f"task{t}":root[t][f"task{t}"].to_numpy() for t in range(1,6)}}); validate(baseline,sample,classes)
    plans={"combined":{1,2,3,5},"task1_only":{1},"task2_only":{2},"task3_only":{3},"task5_only":{5}}; report={}; outdir=ROUND/"outputs/submissions"; outdir.mkdir(parents=True,exist_ok=True)
    for name,use in plans.items():
        d=pd.DataFrame({"id":sample.id,**{f"task{t}":(new[t] if t in use else root[t])[f"task{t}"].to_numpy() for t in range(1,6)}}); validate(d,sample,classes)
        changed={f"task{t}":int((d[f"task{t}"].to_numpy()!=baseline[f"task{t}"].to_numpy()).sum()) for t in range(1,6)}
        if name.endswith("_only"):
            allowed=int(name[4]); assert all(v==0 for k,v in changed.items() if k!=f"task{allowed}")
        path=outdir/f"{name}.csv"; d.to_csv(path,sep=delimiter,index=False); reread=pd.read_csv(path,sep=delimiter); validate(reread,sample,classes)
        report[name]={"path":str(path),"shape":list(reread.shape),"changed_rows":changed,"value_equivalent_to_root_baseline":bool(reread.equals(baseline)),"byte_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"readback_valid":True}
    print(json.dumps({"delimiter":delimiter,"task4_unchanged":True,"task4_value_sha256":value_hash(root[4]),"submissions":report},indent=2))
if __name__=="__main__": main()
