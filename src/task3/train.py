import json
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from src.common import DATA, SEED, add_data_arguments, features, preprocessing, resolve_data_root, save_outputs

MAX_ITER=120; LEARNING_RATE=.10; MAX_LEAF_NODES=31; L2_REGULARIZATION=1.
def model(): return HistGradientBoostingRegressor(loss='absolute_error',max_iter=MAX_ITER,learning_rate=LEARNING_RATE,max_leaf_nodes=MAX_LEAF_NODES,l2_regularization=L2_REGULARIZATION,random_state=SEED)
def main(argv=None):
    parser=add_data_arguments(argparse.ArgumentParser(description='Train Task 3 baseline')); args=parser.parse_args(argv); data_root=resolve_data_root(args.data_root,args.download)
    train=pd.read_csv(data_root/'03_regression/train.csv'); test=pd.read_csv(data_root/'03_regression/test.csv'); y=train.pop('spending_30d'); X=features(train,'spending_30d'); Xtr,Xv,ytr,yv=train_test_split(X,y,test_size=.2,random_state=SEED)
    pipe=Pipeline([('prep',preprocessing(X,[])),('model',model())]); pipe.fit(Xtr,ytr); pred=np.clip(pipe.predict(Xv),0,None); mae=mean_absolute_error(yv,pred); nmae=1/(1+np.abs(yv-pred).sum()/yv.sum()) if yv.sum() else None; metrics={'task':3,'model':'HistGradientBoostingRegressor_absolute_error','seed':SEED,'validation_rows':len(yv),'mae':float(mae),'nmae':float(nmae) if nmae is not None else None}
    final=Pipeline([('prep',preprocessing(X,[])),('model',model())]); final.fit(X,y); out=pd.DataFrame({'id':test.player_id,'task3':np.clip(final.predict(features(test,'spending_30d')),0,None)}); save_outputs('task3',final,out,metrics); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
