"""Confidence-weighted implicit ALS, using fitting clicks only."""
import sys,argparse,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'.techarena_deps'))
import numpy as np
from scipy.sparse import coo_matrix
from implicit.cpu.als import AlternatingLeastSquares
from common import *

def embeddings(a,fit,nu,ni,rank=32,alpha=20,iterations=30):
    clicks=interactions(a,fit,nu,ni)
    m=coo_matrix((np.ones(len(clicks),dtype='float32'),(clicks[:,0],clicks[:,2])),shape=(nu,ni)).tocsr()
    m.data=1+alpha*np.log1p(m.data)
    model=AlternatingLeastSquares(factors=rank,regularization=1,iterations=iterations,num_threads=4,random_state=42)
    model.fit(m,show_progress=False)
    return model.user_factors.copy(),model.item_factors.copy()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--day',type=int,default=26);ap.add_argument('--rank',type=int,default=32);args=ap.parse_args()
    a=load();nu,ni=dimensions();fit,v=split(a,args.day);start=time.time()
    u,i=embeddings(a,fit,nu,ni,args.rank)
    np.savez(RUNS/f'als_{args.day}_{args.rank}.npz',u=u,i=i)
    y=a['label'][v];iv=i[a['item'][v]]
    hv=i[a['history'][v]].sum(1)
    for name,uv in [('user',u[a['user'][v]]),('history',hv)]:
        for cosine in [False,True]:
            p=(uv[:,None,:]*iv).sum(-1)
            if cosine:p/=np.maximum(np.linalg.norm(uv,axis=-1)[:,None]*np.linalg.norm(iv,axis=-1),1e-8)
            record(f'ALS{args.rank}_{name}_cos{cosine}',args.day,y,expit(p),seconds=time.time()-start)

if __name__=='__main__':main()
