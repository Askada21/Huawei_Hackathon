"""Leakage-safe normalized sparse SVD baseline."""
import argparse,time,os,tempfile
import numpy as np
from scipy.sparse import coo_matrix,diags
from scipy.special import expit,logit
from sklearn.utils.extmath import randomized_svd
from sklearn.linear_model import LogisticRegression
from common import *

def embeddings(a,fit,nu,ni,rank=64,n_iter=5,degree_reg=0,negative_weight=0.,half_life_days=0.,adaptive_negatives=False):
    suffix=f'_negative{float(negative_weight)!r}' if negative_weight else ''
    if half_life_days:
        suffix += f'_halflife{float(half_life_days)!r}'
    if adaptive_negatives:
        suffix += '_adaptive'
    end_time = int(a['time'][fit[-1]])
    cache=DATA/f'spectral_{end_time}_r{rank}_iter{n_iter}_reg{degree_reg}{suffix}.npz'
    if cache.exists():
        c=np.load(cache);return c['u'],c['i']
    clicks=interactions(a,fit,nu,ni)
    def edge_weights(times):
        if not half_life_days:
            return np.ones(times.shape, dtype='float32')
        return np.exp2((times-end_time)/(86400*half_life_days)).astype('float32')

    m=coo_matrix((edge_weights(clicks[:,1]),(clicks[:,0],clicks[:,2])),shape=(nu,ni)).tocsr()
    if negative_weight:
        unclicked=np.asarray(a['label'][fit])==0
        users=np.broadcast_to(a['user'][fit,None],unclicked.shape)[unclicked]
        items=a['item'][fit][unclicked]
        weights = np.broadcast_to(edge_weights(a['time'][fit])[:, None], unclicked.shape)[unclicked]
        if adaptive_negatives and len(weights):
            un, uc, inn, ic, global_rate = priors(a, fit, nu, ni)
            if not 0 < global_rate < 1:
                raise ValueError('Adaptive negative weights require clicks and nonclicks in the fitting window')
            global_odds = global_rate / (1 - global_rate)
            user_rate = (uc + 50 * global_rate) / (un + 50)
            item_rate = (ic + 50 * global_rate) / (inn + 50)
            odds_ratio = (user_rate[users] / (1 - user_rate[users])
                          * item_rate[items] / (1 - item_rate[items]) / global_odds**2)
            relative = np.sqrt(odds_ratio).clip(.25, 4)
            weights *= (relative / np.average(relative, weights=weights)).astype('float32')
        negatives=coo_matrix((weights,(users,items)),shape=(nu,ni)).tocsr()
        m=m-negative_weight*negatives
        m.data=np.sign(m.data)*np.log1p(np.abs(m.data))
    else:
        m.data=np.log1p(m.data)
    magnitude=abs(m) if negative_weight else m
    m=diags(np.maximum(np.asarray(magnitude.sum(1)).ravel()+degree_reg,1)**-.5)@m@diags(np.maximum(np.asarray(magnitude.sum(0)).ravel()+degree_reg,1)**-.5)
    u,s,vt=randomized_svd(m,n_components=rank,n_iter=n_iter,random_state=42)
    u=(u*s**.5).astype('float32');i=(vt.T*s**.5).astype('float32')
    # Concurrent experiments may build the same deterministic cache. Readers
    # must never observe an incomplete zip archive.
    with tempfile.NamedTemporaryFile(dir=DATA,suffix='.npz',delete=False) as f:
        temporary=f.name;np.savez(f,u=u,i=i)
    os.replace(temporary,cache)
    return u,i

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--day',type=int,default=26);args=ap.parse_args()
    start=time.time();a=load();nu,ni=dimensions();fit,v=split(a,args.day)
    stats=priors(a,fit,nu,ni)
    record('smoothed_user_item_CTR',args.day,a['label'][v],prior_predict(a,v,stats),fit_events=len(fit),valid_events=len(v))
    u,i=embeddings(a,fit,nu,ni)
    np.savez(RUNS/f'svd_{args.day}.npz',u=u,i=i)
    p=np.sum(u[a['user'][v,None]]*i[a['item'][v]],axis=-1)
    record('SVD64_unadjusted',args.day,a['label'][v],expit(p),seconds=time.time()-start)
    np.save(RUNS/f'svd_{args.day}_raw.npy',p)

if __name__=='__main__':main()
