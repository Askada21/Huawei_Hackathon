"""Ordered daily aggregate baseline. Same feature code for validation and test."""
import argparse,time
import numpy as np
import lightgbm as lgb
from common import *

from tree_features import features

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--day',type=int,default=26);ap.add_argument('--full',action='store_true');args=ap.parse_args()
    a=load();nu,ni=dimensions();fit,v=split(a,args.day);start=time.time()
    xx=[];yy=[]
    # Features for each training day use only strictly earlier dates.
    for day in range(21,27 if args.full else args.day):
        events=np.flatnonzero((a['time']>=cutoff(day))&(a['time']<cutoff(day+1)))
        stats=priors(a,np.flatnonzero(a['time']<cutoff(day)),nu,ni)
        x,names=features(a,events,stats);xx.append(x);yy.append(a['label'][events].ravel())
    x=np.concatenate(xx);y=np.concatenate(yy);del xx,yy
    if args.full:
        stats=priors(a,np.arange(len(a['time'])),nu,ni);ev=load('evaluation');v=np.arange(len(ev['time']));xv,_=features(ev,v,stats)
    else:xv,_=features(a,v,priors(a,fit,nu,ni))
    train=lgb.Dataset(x,label=y,feature_name=names,free_raw_data=True)
    params=dict(objective='binary',metric='auc',learning_rate=.04,num_leaves=23,min_data_in_leaf=400,
                max_bin=127,num_threads=4,verbosity=-1,seed=42,feature_fraction=.9,lambda_l2=10)
    out=RUNS/f'tree_day{args.day}{"_full" if args.full else ""}'
    out.mkdir(exist_ok=True)
    if args.full:
        best=json.loads((RUNS/f'tree_day{args.day}'/'best.json').read_text())['iteration']
        model=lgb.train(params,train,num_boost_round=best)
    else:
        valid=lgb.Dataset(xv,label=a['label'][v].ravel(),reference=train)
        model=lgb.train(params,train,num_boost_round=500,valid_sets=[valid],callbacks=[lgb.early_stopping(40),lgb.log_evaluation(50)])
        (out/'best.json').write_text(json.dumps({'iteration':model.best_iteration}))
    p=model.predict(xv).reshape(-1,10);model.save_model(str(out/'model.txt'))
    np.save(out/('evaluation_pred.npy' if args.full else 'valid_pred.npy'),p)
    np.save(out/'valid_events.npy',v)
    if not args.full:record('ordered_tree',args.day,a['label'][v],p,seconds=time.time()-start,artifact=str(out))

if __name__=='__main__':main()
