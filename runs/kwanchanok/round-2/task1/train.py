"""Task 1 round 2: baseline versus deterministic feature engineering."""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
ROOT=Path(__file__).resolve().parents[4]; sys.path.insert(0,str(ROOT))
from src.common import SEED, preprocessing
DATA=ROOT/'cpe342-kaggle-main'; ROUND=Path(__file__).resolve().parents[1]
MAX_ITER=120; LEARNING_RATE=.10; MAX_LEAF_NODES=31; L2_REGULARIZATION=1.; THRESHOLD_GRID=np.linspace(.05,.95,181)
LOG_CANDIDATES=['reports_received','headshot_percentage','accuracy_score']; INTERACTION_CANDIDATES=[('accuracy_score','reaction_time_ms'),('headshot_percentage','accuracy_score')]

def add_features(frame: pd.DataFrame)->pd.DataFrame:
    out=frame.drop(columns=['player_id','is_cheater'],errors='ignore').copy(); numeric=out.select_dtypes(include=np.number); out['missing_count']=out.isna().sum(axis=1); out['missing_rate']=out.isna().mean(axis=1); groups={}
    for c in numeric.columns:
        m=re.match(r'^(.*)_([1-4])$',c)
        if m: groups.setdefault(m.group(1),[]).append(c)
    for base,cols in sorted(groups.items()):
        cols=sorted(cols,key=lambda c:int(c.rsplit('_',1)[1]))
        if len(cols)<2: continue
        v=out[cols]; out[f'{base}_mean']=v.mean(axis=1); out[f'{base}_std']=v.std(axis=1); out[f'{base}_min']=v.min(axis=1); out[f'{base}_max']=v.max(axis=1); out[f'{base}_range']=v.max(axis=1)-v.min(axis=1); out[f'{base}_change']=v.iloc[:,-1]-v.iloc[:,0]
    for c in LOG_CANDIDATES:
        if c in out and out[c].dropna().ge(0).all(): out[f'{c}_log1p']=np.log1p(out[c])
    for a,b in INTERACTION_CANDIDATES:
        if a in out and b in out: out[f'{a}_over_{b}']=out[a]/(out[b].abs()+1e-6)
    return out

def model(): return HistGradientBoostingClassifier(max_iter=MAX_ITER,learning_rate=LEARNING_RATE,max_leaf_nodes=MAX_LEAF_NODES,l2_regularization=L2_REGULARIZATION,random_state=SEED)
def choose(y,p): return float(THRESHOLD_GRID[int(np.argmax([fbeta_score(y,p>=t,beta=2,average='macro',zero_division=0) for t in THRESHOLD_GRID]))])
def score(y,p,t):
    q=(p>=t).astype(int); return {'macro_f2':float(fbeta_score(y,q,beta=2,average='macro',zero_division=0)),'binary_f2_cheater':float(fbeta_score(y,q,beta=2,average='binary',zero_division=0)),'precision':float(precision_score(y,q,zero_division=0)),'recall':float(recall_score(y,q,zero_division=0)),'confusion_matrix':confusion_matrix(y,q).tolist(),'predicted_positive_rate':float(q.mean())}
def main():
    train=pd.read_csv(DATA/'01_binary_classification/train.csv').dropna(subset=['is_cheater']).copy(); test=pd.read_csv(DATA/'01_binary_classification/test.csv'); y=train.pop('is_cheater').astype(int); raw=train.drop(columns=['player_id']); engineered=add_features(train); tr,va=train_test_split(np.arange(len(train)),test_size=.2,random_state=SEED,stratify=y)
    def fit(X):
        p=Pipeline([('prep',preprocessing(X,[])),('model',model())]); p.fit(X.iloc[tr],y.iloc[tr]); return p,p.predict_proba(X.iloc[va])[:,1]
    ref_pipe,ref_p=fit(raw); eng_pipe,eng_p=fit(engineered); ref_t=choose(y.iloc[va],ref_p); eng_t=choose(y.iloc[va],eng_p); ref=score(y.iloc[va],ref_p,ref_t); eng=score(y.iloc[va],eng_p,eng_t)
    groups=sorted({m.group(1) for c in raw.select_dtypes(include=np.number).columns if (m:=re.match(r'^(.*)_([1-4])$',c))}); logs=[c for c in LOG_CANDIDATES if f'{c}_log1p' in engineered]; interactions=[f'{a}_over_{b}' for a,b in INTERACTION_CANDIDATES if f'{a}_over_{b}' in engineered]
    metrics={'seed':SEED,'train_rows':int(len(tr)),'validation_rows':int(len(va)),'raw_feature_count':int(raw.shape[1]),'engineered_feature_count':int(engineered.shape[1]),'added_feature_count':int(engineered.shape[1]-raw.shape[1]),'added_feature_names':[c for c in engineered if c not in raw],'temporal_feature_groups':groups,'log_transformed_columns':logs,'interaction_features':interactions,'reference_threshold':ref_t,'reference_macro_f2':ref['macro_f2'],'reference_binary_f2_cheater':ref['binary_f2_cheater'],'reference_precision':ref['precision'],'reference_recall':ref['recall'],'reference_confusion_matrix':ref['confusion_matrix'],'reference_predicted_positive_rate':ref['predicted_positive_rate'],'engineered_threshold':eng_t,'engineered_macro_f2':eng['macro_f2'],'engineered_binary_f2_cheater':eng['binary_f2_cheater'],'engineered_precision':eng['precision'],'engineered_recall':eng['recall'],'engineered_confusion_matrix':eng['confusion_matrix'],'engineered_predicted_positive_rate':eng['predicted_positive_rate'],'macro_f2_improvement':eng['macro_f2']-ref['macro_f2']}
    if eng['macro_f2']>ref['macro_f2']:
        final=Pipeline([('prep',preprocessing(engineered,[])),('model',model())]); final.fit(engineered,y); out=pd.DataFrame({'id':test.player_id,'task1':(final.predict_proba(add_features(test))[:,1]>=eng_t).astype(int)}); (ROUND/'outputs/predictions').mkdir(parents=True,exist_ok=True); out.to_csv(ROUND/'outputs/predictions/task1.csv',index=False); metrics['selected_for_test']='engineered'
    else: metrics['selected_for_test']='none_engineered_not_better'
    (ROUND/'outputs/metrics').mkdir(parents=True,exist_ok=True); (ROUND/'outputs/metrics/task1.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
