"""Paired event/user bootstrap; preserves all ten dependent rows together."""
import json
import numpy as np
from common import *

a=load();_,v=split(a,26);y=np.asarray(a['label'][v])
p=np.load(RUNS/'joint_factor_d32_day26_s42'/'valid_calibrated_clean.npy')
base=np.load(RUNS/'calibrated_clean'/'valid_pred.npy')
rng=np.random.default_rng(42);aucs=[];deltas=[]
for k in range(150):
    sample=rng.integers(len(v),size=len(v))
    auc=roc_auc_score(y[sample].ravel(),p[sample].ravel())
    aucs.append(auc);deltas.append(auc-roc_auc_score(y[sample].ravel(),base[sample].ravel()))
result={'bootstrap_unit':'event = distinct validation user, all 10 rows together','replicates':150,
        'seed':42,'auc':roc_auc_score(y.ravel(),p.ravel()),'auc_95_percentile_interval':np.quantile(aucs,[.025,.975]).tolist(),
        'delta_vs_calibrated_converged':roc_auc_score(y.ravel(),p.ravel())-roc_auc_score(y.ravel(),base.ravel()),
        'delta_95_percentile_interval':np.quantile(deltas,[.025,.975]).tolist(),
        'limitation':'Sampling uncertainty only; does not account for model selection or temporal distribution shift.'}
(RUNS/'uncertainty.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
