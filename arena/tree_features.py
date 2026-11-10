"""Shared numpy-only event features for tree and factor calibration."""
import numpy as np

def features(a,events,stats):
    un,uc,inn,ic,g=stats;ids=a['item'][events];u=a['user'][events];h=a['history'][events]
    ht=a['htime'][events];t=a['time'][events];n=(h>0).sum(1)
    cols=[];names=[]
    def add(name,x):
        x=np.asarray(x)
        if x.ndim==1:x=np.repeat(x[:,None],10,axis=1)
        cols.append(x.astype('float32'));names.append(name)
    def relative(name,x):
        add(name,x);add(name+'_mean_diff',x-x.mean(1,keepdims=True));add(name+'_max_diff',x-x.max(1,keepdims=True))
        add(name+'_rank',(x[:,:,None]<x[:,None,:]).sum(2))
    add('history_n',n);add('bucket',a['bucket'][events]);add('day',a['day'][events])
    for k in [1,2,5,10,20]:
        pos=np.maximum(n-k,0)
        last=ht[np.arange(len(events)),pos]
        add(f'hours_since_{k}',np.where(n>=k,(t-last)/3600,-1))
    add('history_span',np.where(n>0,(ht[np.arange(len(n)),np.maximum(n-1,0)]-ht[:,0])/3600,-1))
    matches=ids[:,:,None]==h[:,None,:]
    relative('history_count',matches.sum(2))
    candidate_last=np.where(matches,ht[:,None,:],0).max(2)
    add('candidate_recency',np.where(candidate_last>0,(t[:,None]-candidate_last)/3600,-1))
    add('user_n',un[u]);add('user_clicks',uc[u])
    for k in [5,50,200]:
        add(f'user_ctr_{k}',(uc[u]+k*g)/(un[u]+k))
        relative(f'item_ctr_{k}',(ic[ids]+k*g)/(inn[ids]+k))
    relative('item_n',inn[ids]);add('item_clicks',ic[ids])
    return np.stack(cols,axis=-1).reshape(-1,len(cols)), names

