import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, log_loss

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'arena_data'
RUNS=Path(os.environ.get('ARENA_RUNS',str(ROOT/'arena_runs')))
RUNS.mkdir(exist_ok=True)

def load(name='train'):
    return {k:np.load(DATA/f'{name}_{k}.npy',mmap_mode='r') for k in
            ['user','item','time','bucket','day','history','htime','label']}

def cutoff(day): return int(datetime(2025,1,day,tzinfo=timezone.utc).timestamp())

def split(a,day):
    fit=np.flatnonzero(a['time']<cutoff(day))
    later=np.flatnonzero(a['time']>=cutoff(day))
    _,p=np.unique(a['user'][later],return_index=True)
    valid=np.sort(later[p])
    assert not np.intersect1d(fit,valid).size
    assert np.unique(a['user'][valid]).size==len(valid)
    return fit, valid

def calibration_training_mask(a,events,later_day):
    """Earlier first-event samples may extend past the next cutoff."""
    return a['time'][events]<cutoff(later_day)

def dimensions():
    m=json.loads((DATA/'metadata.json').read_text())
    return len(m['users'])+1,len(m['items'])+1

def record(name,day,y,p,**kwargs):
    result=dict(name=name,day=day,auc=float(roc_auc_score(y.ravel(),np.asarray(p).ravel())),
                logloss=float(log_loss(y.ravel(),np.clip(np.asarray(p).ravel(),1e-7,1-1e-7))),
                recorded_utc=datetime.now(timezone.utc).isoformat(),**kwargs)
    with open(RUNS/'experiments.jsonl','a') as f: f.write(json.dumps(result)+'\n')
    print(json.dumps(result),flush=True)
    return result

def interactions(a,fit,nu,ni):
    """Deduplicate real historical clicks by (user, timestamp, item).

    Use histories attached to fitting events ONLY, plus fitting labels.
    Validation histories are inputs at prediction time, never embedding training data.
    """
    cache=DATA/f'clicks_before_{int(a["time"][fit[-1]])}.npy'
    if cache.exists(): return np.load(cache)
    h=np.asarray(a['history'][fit]); ht=np.asarray(a['htime'][fit]); mask=h>0
    u=np.broadcast_to(a['user'][fit,None],h.shape)[mask]
    triples=np.column_stack([u,ht[mask],h[mask]])
    e,c=np.where(a['label'][fit]>0)
    more=np.column_stack([a['user'][fit[e]],a['time'][fit[e]],a['item'][fit[e],c]])
    triples=np.unique(np.concatenate([triples,more]),axis=0)
    assert np.all(triples[:,1]<=(a['time'][fit].max()))
    with tempfile.NamedTemporaryFile(dir=DATA,suffix='.npy',delete=False) as f:
        temporary=f.name;np.save(f,triples)
    os.replace(temporary,cache)
    print('Distinct fitting clicks including warm-up:',len(triples),flush=True)
    return triples

def priors(a,fit,nu,ni):
    u=np.repeat(a['user'][fit],10); i=a['item'][fit].ravel(); y=a['label'][fit].ravel()
    un=np.bincount(u,minlength=nu); uc=np.bincount(u,weights=y,minlength=nu)
    inn=np.bincount(i,minlength=ni); ic=np.bincount(i,weights=y,minlength=ni)
    g=y.mean()
    return un,uc,inn,ic,g

def prior_predict(a,events,stats,k=50):
    un,uc,inn,ic,g=stats
    return expit(logit((uc[a['user'][events,None]]+k*g)/(un[a['user'][events,None]]+k))+
                 logit((ic[a['item'][events]]+k*g)/(inn[a['item'][events]]+k))-logit(g))
