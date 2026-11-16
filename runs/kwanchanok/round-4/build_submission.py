"""Build and validate Round 4 combined and task-isolation submissions."""
import csv, json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]; ROUND=Path(__file__).resolve().parent; N=25889
EXPECTED=["id","task1","task2","task3","task4","task5"]
FOLDERS={1:"01_binary_classification",2:"02_multiclass_classification",3:"03_regression",4:"04_image_classification",5:"05_anomaly_detection"}

def delimiter(path): return csv.Sniffer().sniff(path.read_text()[:4096],delimiters=",|\t;").delimiter

def read_prediction(task,path):
    frame=pd.read_csv(path); test=pd.read_csv(ROOT/"cpe342-kaggle-main"/FOLDERS[task]/"test.csv")
    expected_ids=test["id"] if "id" in test else test["player_id"]; idcol="id" if "id" in frame else "player_id"
    if frame.columns.tolist()!=[idcol,f"task{task}"] or len(frame)!=N: raise ValueError(f"Task {task} schema/row failure: {path}")
    if frame[idcol].isna().any(): raise ValueError(f"Task {task} null IDs: {path}")
    ids=frame[idcol].astype(str)
    if ids.str.strip().eq("").any() or ids.duplicated().any() or not ids.reset_index(drop=True).equals(expected_ids.astype(str).reset_index(drop=True)):
        raise ValueError(f"Task {task} ID/order failure: {path}")
    return frame.reset_index(drop=True)

def validate(out,sample,classes):
    if out.shape!=(N,6) or out.columns.tolist()!=EXPECTED or not out.id.astype(str).equals(sample.id.astype(str)): raise ValueError("submission shape/columns/ID failure")
    if out.isna().any().any() or out.astype(str).apply(lambda s:s.str.strip().eq("")).any().any(): raise ValueError("submission blank/null")
    if set(out.task1)-{0,1} or set(out.task5)-{0,1} or set(out.task2)-{0,1,2,3}: raise ValueError("classification domain failure")
    task3=pd.to_numeric(out.task3,errors="coerce").to_numpy();
    if not np.isfinite(task3).all() or (task3<0).any() or set(out.task4.astype(str))-classes: raise ValueError("regression/image domain failure")

def main():
    sample_path=ROOT/"cpe342-kaggle-main/sample_submission.csv"; sep=delimiter(sample_path); sample=pd.read_csv(sample_path,sep=sep)
    if sample.shape!=(N,6) or sample.columns.tolist()!=EXPECTED: raise ValueError("unexpected sample")
    root={t:read_prediction(t,ROOT/f"predictions/task{t}_test_predictions.csv") for t in range(1,6)}
    round4={t:read_prediction(t,ROUND/f"outputs/predictions/task{t}.csv") for t in [1,2,3,5]}; round4[4]=root[4]
    if not round4[4].equals(root[4]): raise AssertionError("Task 4 values/order changed")
    train4=pd.read_csv(ROOT/"cpe342-kaggle-main/04_image_classification/train.csv"); target4="game_title" if "game_title" in train4 else "class"; classes=set(train4[target4].astype(str))
    plans={"combined":{1,2,3,5},"task1_only":{1},"task2_only":{2},"task3_only":{3},"task5_only":{5}}
    report={}
    outdir=ROUND/"outputs/submissions"; outdir.mkdir(parents=True,exist_ok=True)
    for name,use_round in plans.items():
        chosen={t:(round4[t] if t in use_round else root[t]) for t in range(1,6)}
        out=pd.DataFrame({"id":sample.id,**{f"task{t}":chosen[t][f"task{t}"].to_numpy() for t in range(1,6)}}); validate(out,sample,classes)
        changed={f"task{t}":int((out[f"task{t}"].to_numpy()!=root[t][f"task{t}"].to_numpy()).sum()) for t in range(1,6)}
        if name.endswith("_only"):
            allowed=int(name[4]); assert all(v==0 for k,v in changed.items() if k!=f"task{allowed}")
        path=outdir/f"{name}.csv"; out.to_csv(path,sep=sep,index=False); reread=pd.read_csv(path,sep=sep); validate(reread,sample,classes)
        report[name]={"path":str(path),"shape":list(reread.shape),"delimiter":sep,"changed_rows":changed,"readback_valid":True}
    print(json.dumps({"task4_unchanged":True,"submissions":report},indent=2))

if __name__=="__main__": main()
