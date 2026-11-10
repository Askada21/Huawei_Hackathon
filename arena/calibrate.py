"""Calibrate using out-of-time labelled predictions, then test a later cutoff."""
import argparse,json
import numpy as np
from common import *

def make_features(path,full=False):
    from factor import Factors,predict,batch,torch
    # Import the numpy-only feature function without loading LightGBM/OpenMP
    # into this PyTorch process (separate native runtimes on macOS).
    from tree_features import features
    config=json.loads((path/'config.json').read_text());a=load();nu,ni=dimensions()
    if full:
        fit=np.arange(len(a['time']));aout=load('evaluation');v=np.arange(len(aout['time']))
    else:fit,v=split(a,config['day']);aout=a
    stats=priors(a,fit,nu,ni)
    x,names=features(aout,v,stats)
    model=Factors(nu,ni,config['dim'],stats,not config['no_history'],config.get('cosine',False))
    model.load_state_dict(torch.load(path/'model.pt',map_location='cpu',weights_only=True))
    p,z=predict(model,aout,v,config['objective'])
    cols=[x]
    def add(name,val):
        val=np.asarray(val,dtype='float32')
        if val.ndim==1:val=np.repeat(val[:,None],10,axis=1)
        cols.append(val.reshape(-1,1));names.append(name)
    add('factor_probability',p);add('factor_logit',z)
    add('factor_logit_minus_mean',z-z.mean(1,keepdims=True));add('factor_logit_minus_max',z-z.max(1,keepdims=True))
    add('factor_rank',(z[:,:,None]<z[:,None,:]).sum(-1))
    for name,val in [('event_p',p.sum(1)),('event_max',p.max(1)),('event_entropy',-(p*np.log(p.clip(1e-9))).sum(1)),('event_score_std',z.std(1))]:add(name,val)
    # Geometry of history affinities can reveal recent or repeated interest.
    wi=model.i.weight.detach().numpy();wu=model.u.weight.detach().numpy()
    wi=wi/np.maximum(np.linalg.norm(wi,axis=-1,keepdims=True),1e-8)
    for start in [0]:
        cs=[]
        for s in range(0,len(v),4096):
            e=v[s:s+4096];h=aout['history'][e];ii=aout['item'][e];n=(h>0).sum(1)
            sim=np.einsum('bcd,bhd->bch',wi[ii],wi[h])
            mean=sim.sum(-1)/n[:,None].clip(1)
            maxsim=np.where(h[:,None,:]>0,sim,-1).max(-1)
            last=sim[np.arange(len(e))[:,None],np.arange(10)[None,:],np.maximum(n-1,0)[:,None]]
            cs.append(np.stack([mean,maxsim,last],axis=-1))
        cs=np.concatenate(cs)
        for k,name in enumerate(['history_cosine_mean','history_cosine_max','history_cosine_last']):add(name,cs[:,:,k])
    return np.concatenate(cols,axis=1),names,aout,v,p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train-model');ap.add_argument('--test-model')
    ap.add_argument('--export');
    ap.add_argument('--out',default='calibrated');ap.add_argument('--full',action='store_true')
    ap.add_argument('--leaves',type=int,default=7)
    args=ap.parse_args()
    if args.leaves<2:ap.error('--leaves must be at least 2')
    if args.export:
        x,names,a,v,p=make_features(RUNS/args.export,args.full)
        np.savez(RUNS/args.export/'calibration_features.npz',x=x,names=names,y=a['label'][v].ravel(),user=a['user'][v],events=v,p=p)
        print('Exported calibration features',args.export,x.shape,flush=True);return
    import lightgbm as lgb
    out=RUNS/args.out;out.mkdir(exist_ok=False)
    train_cache=np.load(RUNS/args.train_model/'calibration_features.npz')
    xt=train_cache['x'];names=train_cache['names'].tolist();p=train_cache['p'];y=train_cache['y']
    # A first event after the earlier cutoff can occur on a LATER date.
    # Exclude those events before fitting the calibrator: they may also be in
    # the later validation sample. This assertion is essential for clean stacking.
    original=load()
    test_day=json.loads((RUNS/args.test_model/'config.json').read_text())['day']
    early=calibration_training_mask(original,train_cache['events'],test_day)
    print('Calibration training events strictly before test cutoff:',int(early.sum()),'excluded later events:',int((~early).sum()),flush=True)
    train_users=train_cache['user'][early]
    xt=xt[np.repeat(early,10)];y=y[np.repeat(early,10)];p=p[early]
    assert np.all(original['time'][train_cache['events'][early]]<cutoff(test_day))
    # Choose model complexity/rounds using disjoint user groups in the EARLIER
    # out-of-time sample, never labels from the later evaluation cutoff.
    hold=np.repeat((train_users%5)==0,10)
    params=dict(objective='binary',metric='auc',num_leaves=args.leaves,min_data_in_leaf=500,learning_rate=.03,
                lambda_l2=30,num_threads=4,verbosity=-1,seed=42)
    train=lgb.Dataset(xt[~hold],label=y[~hold],feature_name=names)
    valid=lgb.Dataset(xt[hold],label=y[hold],reference=train)
    model=lgb.train(params,train,num_boost_round=500,valid_sets=[valid],callbacks=[lgb.early_stopping(35),lgb.log_evaluation(50)])
    nrounds=model.best_iteration
    record('calibrator_inner_raw',int(json.loads((RUNS/args.train_model/'config.json').read_text())['day']),y[hold],p.ravel()[hold])
    record('calibrator_inner_tree',int(json.loads((RUNS/args.train_model/'config.json').read_text())['day']),y[hold],model.predict(xt[hold]),iteration=nrounds)
    # Full earlier OOT sample is now safe to use for fitting.
    model=lgb.train(params,lgb.Dataset(xt,label=y,feature_name=names),num_boost_round=nrounds)
    model.save_model(str(out/'model.txt'))
    test_cache=np.load(RUNS/args.test_model/'calibration_features.npz')
    xv=test_cache['x'];nn=test_cache['names'].tolist();v=test_cache['events'];pv=test_cache['p'];assert nn==names
    pred=model.predict(xv).reshape(-1,10)
    np.save(out/('evaluation_pred.npy' if args.full else 'valid_pred.npy'),pred)
    np.save(out/'valid_events.npy',v)
    (out/'config.json').write_text(json.dumps(dict(vars(args),rounds=nrounds,features=names),indent=2))
    if not args.full:
        day=json.loads((RUNS/args.test_model/'config.json').read_text())['day']
        record(args.out,day,test_cache['y'],pred,artifact=str(out))
        for w in [.25,.5,.75]:
            blend=expit(w*logit(pv.clip(1e-6,1-1e-6))+(1-w)*logit(pred.clip(1e-6,1-1e-6)))
            record(args.out+f'_raw_weight{w}',day,test_cache['y'],blend)

if __name__=='__main__':main()
