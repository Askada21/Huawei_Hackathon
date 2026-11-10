"""Convex same-user impression adaptation in a frozen V5 item space."""
import argparse,json,time
import numpy as np
from scipy.special import logsumexp
from history_model_v3 import construct,batch,torch
from common import *

def probabilities(z):
    return np.exp(z-logsumexp(np.c_[z,np.zeros(len(z))],axis=1)[:,None])

def adapt(x,z,y,penalty,iterations=3):
    d=x.shape[-1];delta=np.zeros(d);xf=x.reshape(-1,d)
    def objective(w):
        zz=z+x@w
        return np.sum(logsumexp(np.c_[zz,np.zeros(len(zz))],axis=1)-(zz*y).sum(1))+.5*np.sum(penalty*w*w)
    for _ in range(iterations):
        p=probabilities(z+x@delta);pf=p.ravel()
        mean=np.einsum('bc,bcd->bd',p,x)
        h=xf.T@(xf*pf[:,None])-mean.T@mean+np.diag(penalty)
        g=xf.T@(p-y).ravel()+penalty*delta
        step=np.linalg.solve(h,g);old=objective(delta)
        for t in range(8):
            proposed=delta-step*2**(-t)
            if objective(proposed)<=old:
                delta=proposed;break
        if np.linalg.norm(step)<1e-5:break
    return delta

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--day',type=int,default=26);ap.add_argument('--out',required=True)
    ap.add_argument('--source');ap.add_argument('--ridge',type=float,default=2);args=ap.parse_args()
    out=RUNS/args.out;out.mkdir(exist_ok=False);start=time.time()
    from argparse import Namespace
    source=RUNS/(args.source or f'v5_observed_aux_d32_day{args.day}_s42')
    config=Namespace(**json.loads((source/'config.json').read_text()))
    assert not config.full and config.day==args.day and config.gate_mode=='global'
    a=load();fit,v=split(a,args.day);nu,ni=dimensions()
    model=construct(config,priors(a,fit,nu,ni),nu,ni)
    model.load_state_dict(torch.load(source/'model.pt',weights_only=True));model.eval()
    torch.set_num_threads(2)
    with torch.no_grad():
        item=model.i.weight.numpy().copy();gu=float((2*model.gate.bias.sigmoid())[0])
        zfit=np.empty((len(fit),10),dtype='float64')
        for s in range(0,len(fit),2048):zfit[s:s+2048]=model.score(*batch(a,fit[s:s+2048])).numpy()
    order=np.argsort(a['user'][fit],kind='stable');users,counts=np.unique(a['user'][fit[order]],return_counts=True)
    delta=np.zeros((nu,item.shape[1]+1));pos=0;penalty=np.r_[np.full(item.shape[1],args.ridge),20.]
    for j,(u,n) in enumerate(zip(users,counts)):
        ix=order[pos:pos+n];ev=fit[ix];pos+=n
        x=np.concatenate([item[a['item'][ev]]*gu,np.ones((n,10,1))],axis=-1).astype('float64')
        delta[u]=adapt(x,zfit[ix],a['label'][ev],penalty)
        if (j+1)%10000==0:print('users',j+1,'seconds',round(time.time()-start,1),flush=True)
    with torch.no_grad():
        zz=[]
        for s in range(0,len(v),2048):zz.append(model.score(*batch(a,v[s:s+2048])).numpy())
    z=np.concatenate(zz);base=probabilities(z)
    change=np.einsum('bcd,bd->bc',item[a['item'][v]]*gu,delta[a['user'][v],:-1])+delta[a['user'][v],-1,None]
    p=probabilities(z+change)
    np.save(out/'valid_pred.npy',p);np.save(out/'valid_events.npy',v);np.save(out/'delta.npy',delta)
    (out/'config.json').write_text(json.dumps(dict(vars(args),source=str(source),iterations=3,prior='zero correction to frozen V5',seconds=time.time()-start),indent=2))
    record(args.out,args.day,a['label'][v],p,seconds=time.time()-start)
    record(args.out+'_unadapted',args.day,a['label'][v],base)
if __name__=='__main__':main()
