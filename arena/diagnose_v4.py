"""Covariate and error audit; never reads evaluation targets."""
import json
import numpy as np
from common import *

def summary(a,events,fitdata,fit,day):
    nu,ni=dimensions();un,uc,inn,ic,g=priors(fitdata,fit,nu,ni)
    h=a['history'][events];ht=a['htime'][events];n=(h>0).sum(1);t=a['time'][events]
    rec=np.where(n>0,(t-ht[np.arange(len(events)),np.maximum(n-1,0)])/3600,np.nan)
    def quant(x):return np.nanquantile(x,[0,.1,.5,.9,1]).tolist()
    return dict(events=len(events),history_length=quant(n),empty_history=float((n==0).mean()),
                unseen_user=float((un[a['user'][events]]==0).mean()),unseen_item=float((inn[a['item'][events]]==0).mean()),
                user_impressions=quant(un[a['user'][events]]),item_impressions=quant(inn[a['item'][events]]),
                recency_hours=quant(rec),hours_after_cutoff=quant((t-cutoff(day))/3600),
                bucket=np.bincount(a['bucket'][events],minlength=4).tolist(),weekend=float(a['day'][events].mean()))

def main():
    a=load();out={}
    for d in [23,24,25,26]:
        f,v=split(a,d);out[f'validation_{d}']=summary(a,v,a,f,d)
    ev=load('evaluation');out['evaluation']=summary(ev,np.arange(len(ev['time'])),a,np.arange(len(a['time'])),27)
    out['training_descriptive_only']=summary(a,np.arange(len(a['time'])),a,np.arange(len(a['time'])),20)
    _,v=split(a,26);y=np.asarray(a['label'][v]);p=np.load(RUNS/'v3_cal_global_single/valid_event_strict.npy')
    clicked=y.sum(1)>0;zpos=np.where(y>0,p,-1).max(1)
    out['errors']=dict(clicked_events=int(clicked.sum()),no_click_events=int((~clicked).sum()),
      positive_score_quantiles=np.quantile(p[y>0],[.1,.5,.9]).tolist(),negative_score_quantiles=np.quantile(p[y==0],[.1,.5,.9]).tolist(),
      negative_above_positive_mean=float(((p>zpos[:,None])& (y==0))[clicked].sum(1).mean()),
      top1=float((p[clicked].argmax(1)==y[clicked].argmax(1)).mean()))
    for tag,path in [('validation','v3_cal_global_single/valid_event_strict.npy'),('evaluation','v3_final_combiner/evaluation_final_strict.npy')]:
        pp=np.load(RUNS/path);out[tag+'_prediction_quantiles']=np.quantile(pp,[0,.1,.5,.9,1]).tolist()
        out[tag+'_event_sum_quantiles']=np.quantile(pp.sum(1),[0,.1,.5,.9,1]).tolist()
    (RUNS/'v4_diagnosis.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
