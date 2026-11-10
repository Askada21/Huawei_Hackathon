"""Sparse shrunken positive log-lift on distinct fitting users, not raw counts."""
import argparse,json,time
import numpy as np
from scipy import sparse
from scipy.special import expit
from common import *

def build(a,fit,nu,ni,topk=200,shrink=5):
    c=interactions(a,fit,nu,ni)
    m=sparse.coo_matrix((np.ones(len(c),dtype='float32'),(c[:,0],c[:,2])),shape=(nu,ni)).tocsr()
    m.data[:]=1
    pop=np.asarray(m.sum(0)).ravel();active=int((np.asarray(m.sum(1)).ravel()>0).sum())
    mt=m.T.tocsr();indptr=[0];indices=[];values=[]
    for s in range(0,ni,500):
        co=(mt[s:s+500]@m).tocsr()
        for local in range(co.shape[0]):
            item=s+local;lo,hi=co.indptr[local:local+2];jj=co.indices[lo:hi];count=co.data[lo:hi]
            expected=pop[item]*pop[jj]/active
            lift=np.maximum(np.log(count/np.maximum(expected,1e-12)),0)*count/(count+shrink)
            keep=(jj!=item)&(lift>0);jj=jj[keep];lift=lift[keep]
            if len(jj)>topk:
                k=np.argpartition(lift,-topk)[-topk:];jj=jj[k];lift=lift[k]
            order=np.argsort(jj);indices.extend(jj[order]);values.extend(lift[order]);indptr.append(len(indices))
    return sparse.csr_matrix((np.asarray(values,dtype='float32'),np.asarray(indices,dtype='int32'),np.asarray(indptr)),shape=(ni,ni))

def score(g,a,events):
    result=[]
    for s in range(0,len(events),2048):
        e=events[s:s+2048];h=np.asarray(a['history'][e]);ii=a['item'][e]
        rows=np.broadcast_to(ii[:,:,None],(len(e),10,20)).ravel();cols=np.broadcast_to(h[:,None,:],(len(e),10,20)).ravel()
        sim=np.asarray(g[rows,cols]).reshape(len(e),10,20)
        n=(h>0).sum(1).clip(1)
        mean=sim.sum(-1)/n[:,None];maximum=sim.max(-1)
        last=sim[np.arange(len(e))[:,None],np.arange(10)[None,:],(n-1)[:,None]]
        result.append(np.stack([mean,maximum,last],axis=-1))
    return np.concatenate(result)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--day',type=int,default=26);ap.add_argument('--out',required=True);args=ap.parse_args()
    start=time.time();a=load();fit,v=split(a,args.day);nu,ni=dimensions();out=RUNS/args.out;out.mkdir(exist_ok=False)
    g=build(a,fit,nu,ni);sparse.save_npz(out/'graph.npz',g);features=score(g,a,v)
    source=RUNS/f'v5_observed_aux_d32_day{args.day}_s42';cache=np.load(source/'calibration_features.npz')
    assert np.array_equal(cache['events'],v)
    x=np.c_[cache['x'],features.reshape(-1,3)];names=cache['names'].tolist()+['shrunken_lift_mean','shrunken_lift_max','shrunken_lift_last']
    np.savez(out/'calibration_features.npz',x=x,names=names,y=a['label'][v].ravel(),user=a['user'][v],events=v,p=cache['p'])
    np.save(out/'valid_events.npy',v)
    (out/'config.json').write_text(json.dumps(dict(day=args.day,full=False,family='distinct_user_shrunken_loglift',topk=200,shrink=5,source=str(source),seconds=time.time()-start),indent=2))
    for k,name in enumerate(['mean','max','last']):record(args.out+'_'+name,args.day,a['label'][v],expit(features[:,:,k]))
    print('Complete',args.day,'seconds',time.time()-start,flush=True)
if __name__=='__main__':main()
