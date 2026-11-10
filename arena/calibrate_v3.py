"""Fit a row combiner on disjoint one-day out-of-time periods.

No outer-period labels select rounds. In full evaluation mode, train and test
event indices belong to different datasets; strict timestamp separation proves
their disjointness instead of comparing those numeric index namespaces.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score,log_loss
from common import RUNS,load,cutoff,split


def sha256(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def period_mask(times,event_indices,period_day,test_day,train_window_days=1):
    """A first event after D is eligible only if it actually occurs on day D."""
    if train_window_days!=1:
        raise ValueError('Only one-day OOT windows are permitted: every period ends strictly before cutoff(D+1)')
    if period_day>=test_day:
        raise ValueError('Every fitting period must begin before the outer test cutoff')
    events=np.asarray(event_indices)
    if events.ndim!=1 or not np.issubdtype(events.dtype,np.integer):
        raise ValueError('Event indices must be a one-dimensional integer array')
    if len(np.unique(events))!=len(events):
        raise ValueError('Duplicate event indices inside an OOT cache')
    if (events<0).any() or (events>=len(times)).any():
        raise ValueError('OOT event index outside the training dataset')
    observed=np.asarray(times)[events]
    return (observed>=cutoff(period_day))&(observed<min(cutoff(period_day+1),cutoff(test_day)))


def check_unique_period(period_day,events,seen_days,seen_events,test_events=None):
    if period_day in seen_days:
        raise ValueError('Duplicate OOT date windows are not allowed')
    current=set(np.asarray(events).tolist())
    if len(current)!=len(events) or current.intersection(seen_events):
        raise ValueError('Duplicate training events across OOT periods')
    if test_events is not None and current.intersection(set(np.asarray(test_events).tolist())):
        raise ValueError('Training OOT events overlap outer validation events')
    seen_days.add(period_day);seen_events.update(current)


def feature_cache(directory):
    path=directory/'calibration_features.npz'
    with np.load(path) as cache:
        # No label field is loaded here, including in outer validation mode.
        result={key:cache[key] for key in ['x','names','events','user']}
    names=result['names'].tolist();events=result['events'];x=result['x']
    if len(set(names))!=len(names):raise ValueError('Duplicate feature names')
    if x.ndim!=2 or x.shape!=(10*len(events),len(names)):
        raise ValueError('Cache dimensions do not match ten candidate rows per event')
    if len(result['user'])!=len(events):raise ValueError('User/event dimensions differ')
    if not np.isfinite(x).all():raise ValueError('Non-finite feature values')
    return result


def cache_provenance(directory,config):
    p=directory/'calibration_features.npz';cp=directory/'config.json'
    return {'directory':str(directory.resolve()),'cache_sha256':sha256(p),'cache_bytes':p.stat().st_size,
            'config_sha256':sha256(cp),'model_config':config}


def parse_args():
    ap=argparse.ArgumentParser(description=__doc__)
    mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--train-models',nargs='+',help='Relative RUNS cache directories for distinct earlier OOT days')
    mode.add_argument('--apply-model',help='Apply a frozen LightGBM model.txt without fitting')
    ap.add_argument('--test-model',required=True,help='Relative RUNS feature cache directory to predict')
    ap.add_argument('--out',required=True,help='New relative RUNS output directory')
    ap.add_argument('--train-window-days',type=int,default=1)
    ap.add_argument('--max-rounds',type=int,default=1500,help='Chosen using earlier inner-user validation only')
    ap.add_argument('--patience',type=int,default=80)
    ap.add_argument('--fulltest',action='store_true',help='Test cache is full evaluation; outer cutoff becomes Jan27')
    args=ap.parse_args()
    if args.train_window_days!=1:ap.error('--train-window-days must be1: strict one-day windows are required')
    if args.max_rounds<1 or args.patience<1:ap.error('--max-rounds and --patience must be positive')
    return args


def main():
    args=parse_args();start=time.time();out=RUNS/args.out
    if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
    testdir=RUNS/args.test_model;testconfig=json.loads((testdir/'config.json').read_text())
    if bool(testconfig.get('full',False))!=args.fulltest:
        raise ValueError('--fulltest must exactly match the test model config full flag')
    testday=27 if args.fulltest else int(testconfig['day'])
    test=feature_cache(testdir);names=test['names'].tolist();a=load()
    if args.fulltest:
        eval_data=load('evaluation');expected=np.arange(len(eval_data['time']))
        if not np.array_equal(test['events'],expected):
            raise ValueError('Full evaluation cache events must follow the exact evaluation ordering')
        if not np.all(eval_data['time']>=cutoff(27)):
            raise ValueError('Evaluation events occur before the official Jan27 cutoff')
        if not np.array_equal(test['user'],eval_data['user']):raise ValueError('Evaluation user alignment differs')
        outer_events=None
    else:
        _,expected=split(a,testday)
        if not np.array_equal(test['events'],expected):
            raise ValueError('Test cache differs from exact first-user events after its cutoff')
        if not np.array_equal(test['user'],a['user'][expected]):raise ValueError('Outer user alignment differs')
        outer_events=expected
    details={'test_model':args.test_model,'test_cutoff_day':testday,'fulltest':args.fulltest,
             'test_feature_provenance':cache_provenance(testdir,testconfig),
             'outer_labels_used_for_fitting_or_round_selection':False,
             'evaluation_labels_loaded':False,'training_window_days':1,
             'event_index_namespaces':'training vs evaluation are distinct; strict time separation proves disjointness' if args.fulltest else 'same training-event namespace; explicit overlap check'}
    if args.apply_model:
        source=Path(args.apply_model)
        if not source.exists():source=RUNS/args.apply_model
        model=lgb.Booster(model_file=str(source))
        if model.feature_name()!=names:raise ValueError('Frozen model feature contract differs')
        details.update(mode='frozen_inference',source_model=str(source.resolve()),source_model_sha256=sha256(source))
        config_path=source.parent/'config.json'
        if config_path.exists():details['source_training_config']=json.loads(config_path.read_text())
    else:
        xx=[];yy=[];uu=[];ee=[];periods=[];seen_days=set();seen_events=set()
        for name in args.train_models:
            directory=RUNS/name;cfg=json.loads((directory/'config.json').read_text())
            if cfg.get('full',False):raise ValueError('A full-fit/evaluation cache cannot be a fitting OOT period')
            day=int(cfg['day']);cache=feature_cache(directory)
            if cache['names'].tolist()!=names:raise ValueError('OOT feature contract differs between periods')
            eligible=period_mask(a['time'],cache['events'],day,testday,args.train_window_days)
            events=cache['events'][eligible]
            if not len(events):raise ValueError(f'No eligible one-day OOT events in {name}')
            check_unique_period(day,events,seen_days,seen_events,outer_events)
            if not np.array_equal(cache['user'][eligible],a['user'][events]):raise ValueError('OOT user/event alignment differs')
            x=cache['x'].reshape(-1,10,len(names))[eligible].reshape(-1,len(names))
            # Source labels are indexed only AFTER all strict time-window checks.
            y=np.asarray(a['label'][events]).ravel()
            xx.append(x);yy.append(y);uu.append(np.asarray(a['user'][events]));ee.append(events)
            item=cache_provenance(directory,cfg)
            item.update(period_day=day,eligible_events=len(events),excluded_events=int((~eligible).sum()),
                        min_time=int(a['time'][events].min()),max_time=int(a['time'][events].max()),
                        end_exclusive=min(cutoff(day+1),cutoff(testday)))
            periods.append(item)
        xt=np.concatenate(xx);y=np.concatenate(yy);users=np.concatenate(uu);train_events=np.concatenate(ee)
        del xx,yy,uu,ee
        hold=np.repeat(users%5==0,10)
        if len(np.unique(y[hold]))!=2 or len(np.unique(y[~hold]))!=2:
            raise ValueError('Both inner partitions must contain both label classes')
        params=dict(objective='binary',metric='auc',num_leaves=7,min_data_in_leaf=500,learning_rate=.03,
                    lambda_l2=30,num_threads=2,verbosity=-1,seed=42)
        train=lgb.Dataset(xt[~hold],label=y[~hold],feature_name=names)
        valid=lgb.Dataset(xt[hold],label=y[hold],reference=train)
        chosen=lgb.train(params,train,num_boost_round=args.max_rounds,valid_sets=[valid],
                         callbacks=[lgb.early_stopping(args.patience),lgb.log_evaluation(50)])
        rounds=chosen.best_iteration;inner=chosen.predict(xt[hold],num_threads=2)
        details.update(mode='fit_earlier_OOT',training_periods=periods,training_events=len(train_events),
                       training_rows=len(y),training_event_sha256=hashlib.sha256(train_events.tobytes()).hexdigest(),
                       inner_holdout_rule='user_id_integer_lookup_index %5 ==0; same user partition across all periods',
                       inner_users_fit=int(np.unique(users[users%5!=0]).size),
                       inner_users_holdout=int(np.unique(users[users%5==0]).size),
                       inner_rows_fit=int((~hold).sum()),inner_rows_holdout=int(hold.sum()),
                       inner_auc=float(roc_auc_score(y[hold],inner)),rounds=rounds,params=params,
                       max_rounds=args.max_rounds,patience=args.patience,
                       best_iteration_at_search_boundary=bool(rounds==args.max_rounds),
                       inner_evidence_role='model-selection evidence; rounds selected on earlier held-out users, not an untouched confirmation')
        model=lgb.train(params,lgb.Dataset(xt,label=y,feature_name=names),num_boost_round=rounds)
    pred=model.predict(test['x'],num_threads=2).reshape(-1,10)
    if not np.isfinite(pred).all() or not ((pred>=0)&(pred<=1)).all():
        raise ValueError('Invalid output probabilities')
    details.update(features=names,test_events=len(test['events']),seconds=time.time()-start)
    if not args.fulltest:
        # Outer labels are read only after round selection, refit, and prediction.
        outer_y=np.asarray(a['label'][test['events']]).ravel()
        details['outer_metrics']={'auc':float(roc_auc_score(outer_y,pred.ravel())),
                                 'logloss':float(log_loss(outer_y,pred.ravel().clip(1e-7,1-1e-7))),
                                 'role':'outer model-comparison evidence; choosing single vs multi-period recipes here is still model selection'}
    out.mkdir(parents=True,exist_ok=False)
    if not args.apply_model:model.save_model(str(out/'model.txt'))
    np.save(out/('evaluation_pred.npy' if args.fulltest else 'valid_pred.npy'),pred)
    np.save(out/'events.npy',test['events'])
    if not args.fulltest:np.save(out/'valid_events.npy',test['events'])
    if not args.apply_model:np.save(out/'training_events.npy',train_events)
    (out/'config.json').write_text(json.dumps(details,indent=2));print(json.dumps(details,indent=2))
    with open(RUNS/'v3_calibration_experiments.jsonl','a') as f:
        f.write(json.dumps({'out':args.out,'test_model':args.test_model,'fulltest':args.fulltest,
                           'training_period_days':[p['period_day'] for p in details.get('training_periods',[])],
                           'outer_metrics':details.get('outer_metrics'),'rounds':details.get('rounds')})+'\n')


if __name__=='__main__':main()
