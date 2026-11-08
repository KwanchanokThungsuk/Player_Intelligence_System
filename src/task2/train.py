import json
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.common import DATA, SEED, add_data_arguments, features, preprocessing, resolve_data_root, save_outputs

MAX_ITER=120; LEARNING_RATE=.10; MAX_LEAF_NODES=31; L2_REGULARIZATION=1.
def model(): return HistGradientBoostingClassifier(max_iter=MAX_ITER,learning_rate=LEARNING_RATE,max_leaf_nodes=MAX_LEAF_NODES,l2_regularization=L2_REGULARIZATION,random_state=SEED)
def main(argv=None):
    parser=add_data_arguments(argparse.ArgumentParser(description='Train Task 2 baseline')); args=parser.parse_args(argv); data_root=resolve_data_root(args.data_root,args.download)
    train=pd.read_csv(data_root/'02_multiclass_classification/train.csv'); test=pd.read_csv(data_root/'02_multiclass_classification/test.csv'); y=train.pop('segment').astype(int); X=features(train,'segment'); cats=X.select_dtypes(exclude=np.number).columns.tolist(); Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=y)
    pipe=Pipeline([('prep',preprocessing(X,cats)),('model',model())]); pipe.fit(Xtr,ytr); pred=pipe.predict(Xv); metrics={'task':2,'model':'HistGradientBoostingClassifier','seed':SEED,'validation_rows':len(yv),'f1_macro':float(f1_score(yv,pred,average='macro')),'f1_weighted':float(f1_score(yv,pred,average='weighted')),'f1_per_class':f1_score(yv,pred,average=None,labels=[0,1,2,3]).tolist()}
    final=Pipeline([('prep',preprocessing(X,cats)),('model',model())]); final.fit(X,y); out=pd.DataFrame({'id':test.player_id,'task2':final.predict(features(test,'segment')).astype(int)}); save_outputs('task2',final,out,metrics); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
