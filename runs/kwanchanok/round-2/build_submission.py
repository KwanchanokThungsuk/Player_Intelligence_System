"""Round 2 submission builder: local Task 1, read-only baseline Tasks 2-5."""
from pathlib import Path
import numpy as np, pandas as pd
ROUND=Path(__file__).resolve().parent; ROOT=ROUND.parents[2]; DATA=ROOT/'cpe342-kaggle-main'; sample=pd.read_csv(DATA/'sample_submission.csv'); result=pd.DataFrame({'id':sample.id}); dirs={1:'01_binary_classification',2:'02_multiclass_classification',3:'03_regression',4:'04_image_classification',5:'05_anomaly_detection'}
for task in range(1,6):
    name=f'task{task}'; path=ROUND/'outputs/predictions/task1.csv' if task==1 else ROOT/'predictions'/f'{name}_test_predictions.csv'; pred=pd.read_csv(path); test=pd.read_csv(DATA/dirs[task]/'test.csv')
    if len(pred)!=25889 or len(test)!=25889 or pred.columns.tolist()!=['id',name] or pred.id.isna().any() or pred.id.duplicated().any(): raise ValueError(f'{name} schema/row failure')
    source='id' if task==4 else 'player_id'
    if pred.id.astype(str).tolist()!=test[source].astype(str).tolist(): raise ValueError(f'{name} order failure')
    values=pred[name]
    if task in (1,5) and not set(values).issubset({0,1}): raise ValueError(f'{name} domain failure')
    if task==2 and not set(values).issubset({0,1,2,3}): raise ValueError('task2 domain failure')
    if task==3 and (not np.isfinite(values).all() or (values<0).any()): raise ValueError('task3 domain failure')
    if task==4 and not set(values.astype(str)).issubset(set(pd.read_csv(DATA/'04_image_classification/train.csv')['class'].astype(str))): raise ValueError('task4 domain failure')
    result[name]=values.to_numpy()
result.to_csv(ROUND/'outputs/submission.csv',sep='|',index=False); check=pd.read_csv(ROUND/'outputs/submission.csv',sep='|')
if check.shape!=(25889,6) or check.isna().any().any() or not np.isfinite(check.task3).all(): raise ValueError('submission validation failure')
print(ROUND/'outputs/submission.csv')
