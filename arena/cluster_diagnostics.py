import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score
from common import *

a=load();_,v=split(a,26);z=np.load(RUNS/'svd_26.npz');i=z['i'][:,1:32].copy()
i/=np.maximum(np.linalg.norm(i,axis=1,keepdims=True),1e-9)
y=a['label'][v].ravel()
for k in [8,16,32,64,128]:
    km=MiniBatchKMeans(n_clusters=k,random_state=42,n_init=3,batch_size=4096).fit(i[1:])
    c=np.r_[-1,km.labels_]
    p=(c[a['item'][v]][:,:,None]==c[a['history'][v]][:,None,:]).sum(-1)
    print('clusters',k,'history_count_auc',roc_auc_score(y,p.ravel()),flush=True)
    np.save(RUNS/f'clusters26_{k}.npy',c)
