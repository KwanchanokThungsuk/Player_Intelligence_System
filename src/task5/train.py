from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.common import ROOT, SEED, add_data_arguments, resolve_data_root

CONTAMINATION=0.05; N_ESTIMATORS=100
def main(argv=None):
    parser=add_data_arguments(argparse.ArgumentParser(description='Fit Task 5 Isolation Forest baseline')); args=parser.parse_args(argv); data_root=resolve_data_root(args.data_root,args.download)
    raw=pd.read_csv(data_root/'05_anomaly_detection/test.csv'); ids=raw['player_id'].copy(); X=raw.drop(columns=['player_id']); categorical=X.select_dtypes(exclude=np.number).columns.tolist()
    if ids.isna().any() or ids.duplicated().any(): raise ValueError('Task 5 IDs must be non-null and unique')
    if categorical: raise ValueError(f'Unexpected categorical features: {categorical}')
    pipe=Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),('model',IsolationForest(n_estimators=N_ESTIMATORS,contamination=CONTAMINATION,random_state=SEED,n_jobs=-1))]); pipe.fit(X); labels=pipe.predict(X); scores=pipe.decision_function(X)
    out=pd.DataFrame({'id':ids,'task5':(labels==-1).astype(int)}); (ROOT/'predictions').mkdir(exist_ok=True); (ROOT/'artifacts').mkdir(exist_ok=True); out.to_csv(ROOT/'predictions/task5_test_predictions.csv',index=False); joblib.dump(pipe,ROOT/'artifacts/task5_isolation_forest.joblib')
    config={'seed':SEED,'contamination':CONTAMINATION,'n_estimators':N_ESTIMATORS,'features':X.columns.tolist(),'id_column':'player_id','output_mapping':{'1':0,'-1':1}}; (ROOT/'artifacts/task5_config.json').write_text(json.dumps(config,indent=2))
    metrics={'task':5,'model':'IsolationForest','seed':SEED,'contamination':CONTAMINATION,'hypothesis':'5% anomaly rate because labels are unavailable','rows':len(out),'features':X.shape[1],'missing_values_before_imputation':int(X.isna().sum().sum()),'anomaly_count':int(out.task5.sum()),'anomaly_rate':float(out.task5.mean()),'decision_score_min':float(scores.min()),'decision_score_p01':float(np.quantile(scores,.01)),'decision_score_median':float(np.median(scores)),'decision_score_p99':float(np.quantile(scores,.99)),'decision_score_max':float(scores.max()),'f3':None}; (ROOT/'artifacts/task5_metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
