import json
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.common import DATA, SEED, add_data_arguments, features, preprocessing, resolve_data_root, save_outputs

MAX_ITER=120; LEARNING_RATE=.10; MAX_LEAF_NODES=31; L2_REGULARIZATION=1.; THRESHOLD_GRID=np.linspace(.05,.95,181)

def model():
    return HistGradientBoostingClassifier(max_iter=MAX_ITER, learning_rate=LEARNING_RATE, max_leaf_nodes=MAX_LEAF_NODES, l2_regularization=L2_REGULARIZATION, random_state=SEED)

def main(argv=None):
    parser=add_data_arguments(argparse.ArgumentParser(description='Train Task 1 baseline')); args=parser.parse_args(argv); data_root=resolve_data_root(args.data_root,args.download)
    train=pd.read_csv(data_root/'01_binary_classification/train.csv'); test=pd.read_csv(data_root/'01_binary_classification/test.csv')
    train=train.dropna(subset=['is_cheater']).copy(); y=train.pop('is_cheater').astype(int); X=features(train,'is_cheater')
    Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)
    pipe=Pipeline([('prep',preprocessing(X,[])),('model',model())]); pipe.fit(Xtr,ytr); prob=pipe.predict_proba(Xv)[:,1]
    scores=[fbeta_score(yv,prob>=t,beta=2,zero_division=0) for t in THRESHOLD_GRID]; threshold=float(THRESHOLD_GRID[int(np.argmax(scores))]); pred=(prob>=threshold).astype(int)
    metrics={'task':1,'model':'HistGradientBoostingClassifier','seed':SEED,'validation_rows':len(yv),'threshold':threshold,'f2':float(fbeta_score(yv,pred,beta=2)),'precision':float(precision_score(yv,pred,zero_division=0)),'recall':float(recall_score(yv,pred,zero_division=0)),'confusion_matrix':confusion_matrix(yv,pred).tolist()}
    final=Pipeline([('prep',preprocessing(X,[])),('model',model())]); final.fit(X,y); out=pd.DataFrame({'id':test.player_id,'task1':(final.predict_proba(features(test,'is_cheater'))[:,1]>=threshold).astype(int)})
    save_outputs('task1',final,out,metrics); (data_root.parent/'artifacts/task1_threshold.json').write_text(json.dumps({'task':1,'threshold':threshold,'selection_metric':'F2','validation_seed':SEED},indent=2)); print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()
