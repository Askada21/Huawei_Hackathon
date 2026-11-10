import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from common import *

a=load();_,v=split(a,26);z=np.load(RUNS/'svd_26.npz');u=z['u'];i=z['i'];y=a['label'][v].ravel()
for d in [4,8,12,16,24,32,64]:
    for skip in [0,1]:
        uv=u[a['user'][v],skip:d];iv=i[a['item'][v],skip:d]
        hv=i[a['history'][v],skip:d].sum(1)/(a['history'][v]>0).sum(1,keepdims=True).clip(1)
        for kind,uu in [('user',uv),('history',hv)]:
            p=(uu[:,None,:]*iv).sum(-1)
            cos=p/np.maximum(np.linalg.norm(uu,axis=-1)[:,None]*np.linalg.norm(iv,axis=-1),1e-9)
            print(d,skip,kind,'dot',roc_auc_score(y,p.ravel()),'cos',roc_auc_score(y,cos.ravel()),flush=True)
