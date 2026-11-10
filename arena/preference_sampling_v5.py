"""Pools of real same-user negative impressions; fitting rows only."""
import numpy as np

def observed_negative_pool(a,fit,nu):
    users=np.repeat(a['user'][fit],10);items=a['item'][fit].ravel()
    negative=a['label'][fit].ravel()==0
    users=users[negative];items=items[negative]
    order=np.argsort(users,kind='stable');items=items[order]
    counts=np.bincount(users,minlength=nu)
    starts=np.r_[0,np.cumsum(counts)[:-1]]
    return items,counts,starts

def observed_bucket_negative_pool(a,fit,nu):
    """Real negatives grouped by user AND observed UTC six-hour bucket.

    The caller must pass only fitting events. Multiplicity of impressions is
    retained; no unobserved item is fabricated as a negative.
    """
    keys=np.repeat(np.asarray(a['user'][fit])*4+np.asarray(a['bucket'][fit]),10)
    items=np.asarray(a['item'][fit]).ravel()
    negative=np.asarray(a['label'][fit]).ravel()==0
    keys=keys[negative];items=items[negative]
    order=np.argsort(keys,kind='stable')
    counts=np.bincount(keys,minlength=nu*4)
    starts=np.r_[0,np.cumsum(counts)[:-1]]
    return items[order],counts,starts
