"""Clean event-click decomposition, fitted on earlier out-of-time predictions.

All training events must occur strictly before the later validation cutoff.
No candidate label from the later sample is used to fit any model or feature.
"""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score,log_loss
from common import RUNS,load,cutoff,calibration_training_mask,split

def validate_event_alignment(expected,row_events,row_count):
    expected=np.asarray(expected);row_events=np.asarray(row_events)
    if expected.ndim!=1 or row_events.ndim!=1:
        raise ValueError('Event identifiers must be one-dimensional arrays')
    if row_count!=10*len(expected):
        raise ValueError('Prediction row count differs from ten rows per feature event')
    if not np.array_equal(expected,row_events):
        raise ValueError('Prediction events differ from feature events or their ordering')

def read_cache(name):
    c=np.load(RUNS/name/'calibration_features.npz')
    return {k:c[k] for k in c.files}

def validate_training_config(config):
    if config.get('day')!=25 or config.get('full',False):
        raise ValueError('Event training features must be a January25 non-full OOT cache')
    end=config.get('selection_end_day')
    if end is not None and end>26:
        raise ValueError('Training feature checkpoint selection includes the January26 outer period')

def file_hash(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def event_features(c):
    x=c['x'].reshape(-1,10,len(c['names']));parts=[];names=[]
    # Symmetric summaries cannot learn arbitrary candidate row positions.
    for op in ['mean','min','max','std']:
        parts.append(getattr(x,op)(axis=1))
        names += [str(n)+'_'+op for n in c['names']]
    p=c['p'];q=p/p.sum(1,keepdims=True).clip(1e-8)
    parts.append(np.column_stack([-(q*np.log(q.clip(1e-8))).sum(1),np.sort(q,axis=1)[:,-1]-np.sort(q,axis=1)[:,-2]]))
    names+=['choice_entropy','choice_top_gap']
    return np.concatenate(parts,axis=1).astype('float32'),names

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--test-model',default='joint_factor_d32_day26_s42')
    ap.add_argument('--train-model',default='converged_factor_d32_day25_s42')
    ap.add_argument('--row-rounds',type=int,default=493,help='Fixed earlier-user row-combiner rounds; never chosen on outer labels')
    ap.add_argument('--row-pred',help='Clean row predictions, default saved baseline joint predictions')
    ap.add_argument('--row-events',help='Prediction event array; default valid_events.npy beside --row-pred')
    ap.add_argument('--out',default='v3_event_decomposition');args=ap.parse_args()
    if args.row_rounds<1:ap.error('--row-rounds must be positive')
    out=RUNS/args.out
    if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(exist_ok=False)
    start=time.time();a=load();tr=read_cache(args.train_model);te=read_cache(args.test_model)
    train_config=json.loads((RUNS/args.train_model/'config.json').read_text())
    validate_training_config(train_config)
    config=json.loads((RUNS/args.test_model/'config.json').read_text())
    assert config['day']==26 and not config.get('full',False),'This comparison requires the January26 validation cutoff'
    _,expected=split(a,26)
    assert np.array_equal(te['events'],expected),'Test cache must contain the exact January26 first-user events in order'
    row_path=Path(args.row_pred) if args.row_pred else RUNS/'joint_factor_d32_day26_s42'/'valid_calibrated_clean.npy'
    row_events_path=Path(args.row_events) if args.row_events else row_path.parent/'valid_events.npy'
    clean=np.load(row_path).reshape(-1,10)
    validate_event_alignment(te['events'],np.load(row_events_path),clean.size)
    early=calibration_training_mask(a,tr['events'],26)
    assert np.all(a['time'][tr['events'][early]]<cutoff(26))
    assert not np.intersect1d(tr['events'][early],te['events']).size
    x,names=event_features(tr);xt=x[early];xv,_=event_features(te)
    y=tr['y'].reshape(-1,10)[early];yt=y.sum(1);q=tr['p'][early];q/=q.sum(1,keepdims=True).clip(1e-8)
    hold=tr['user'][early]%5==0
    def score(p,data):
        return 'global_row_auc',roc_auc_score(y[hold].ravel(),(p[:,None]*q[hold]).ravel()),True
    params=dict(objective='binary',metric='None',num_leaves=7,min_data_in_leaf=300,learning_rate=.03,lambda_l2=30,num_threads=2,verbosity=-1,seed=42)
    model=lgb.train(params,lgb.Dataset(xt[~hold],label=yt[~hold],feature_name=names),num_boost_round=500,
                    valid_sets=[lgb.Dataset(xt[hold],label=yt[hold])],feval=score,
                    callbacks=[lgb.early_stopping(40),lgb.log_evaluation(100)])
    rounds=model.best_iteration
    inner=model.predict(xt[hold]);inner_summary={'raw_auc':roc_auc_score(y[hold].ravel(),tr['p'][early][hold].ravel()),
                  'decomposed_auc':roc_auc_score(y[hold].ravel(),(inner[:,None]*q[hold]).ravel()),
                  'event_auc':roc_auc_score(yt[hold],inner),'rounds':rounds}
    # Confirmation on earlier out-of-user predictions: unlike the saved clean
    # full-calibration model, this combiner has never fitted held-out users' labels.
    row_x=tr['x'].reshape(-1,10,len(tr['names']))[early]
    rowparams=dict(objective='binary',metric='auc',num_leaves=7,min_data_in_leaf=500,learning_rate=.03,
                   lambda_l2=30,num_threads=2,verbosity=-1,seed=42)
    rowmodel=lgb.train(rowparams,lgb.Dataset(row_x[~hold].reshape(-1,row_x.shape[-1]),
                       label=y[~hold].ravel(),feature_name=tr['names'].tolist()),num_boost_round=args.row_rounds)
    inner_row=rowmodel.predict(row_x[hold].reshape(-1,row_x.shape[-1])).reshape(-1,10)
    inner_decomposed=inner[:,None]*inner_row/inner_row.sum(1,keepdims=True).clip(1e-8)
    inner_blend=.5*inner_row+.5*inner_decomposed
    inner_summary.update(clean_row_combiner_auc=float(roc_auc_score(y[hold].ravel(),inner_row.ravel())),
                         clean_decomposition_auc=float(roc_auc_score(y[hold].ravel(),inner_decomposed.ravel())),
                         fixed_half_correction_auc=float(roc_auc_score(y[hold].ravel(),inner_blend.ravel())),
                         clean_row_combiner_rounds=args.row_rounds,
                         evidence_role='model-selection evidence; not an untouched confirmation set because early-stopping rounds were chosen on these users',
                         note='All combiner and event fits exclude held-out user%5==0 labels; fixed half correction specified before this earlier-period comparison.')
    np.savez(out/'inner_confirmation.npz',label=y[hold],baseline=inner_row,corrected=inner_blend,
             decomposed=inner_decomposed,events=tr['events'][early][hold])
    model=lgb.train(params,lgb.Dataset(xt,label=yt,feature_name=names),num_boost_round=rounds)
    model.save_model(str(out/'event_model.txt'))
    g=model.predict(xv);raw=te['p'];conditional=raw/raw.sum(1,keepdims=True).clip(1e-8)
    pred=g[:,None]*conditional
    label=te['y'].reshape(-1,10)
    results={}
    def add(name,p):
        results[name]={'auc':float(roc_auc_score(label.ravel(),p.ravel())),
                       'logloss':float(log_loss(label.ravel(),p.ravel().clip(1e-7,1-1e-7)))}
    add('decomposed_raw_choice',pred)
    # Frozen candidate-choice ranking supplied by the existing clean combiner.
    conditional_clean=clean/clean.sum(1,keepdims=True).clip(1e-8)
    pred_clean=g[:,None]*conditional_clean
    add('decomposed_clean_choice',pred_clean)
    add('baseline_clean_rows',clean)
    for w in [.25,.5,.75]:add(f'blend_event_weight_{w}',(1-w)*clean+w*pred_clean)
    np.save(out/'valid_pred.npy',pred_clean);np.save(out/'event_probability.npy',g);np.save(out/'valid_events.npy',te['events'])
    np.save(out/'valid_half_corrected.npy',.5*(clean+pred_clean))
    summary={'training_events':int(early.sum()),'excluded_late_events':int((~early).sum()),'test_events':len(te['events']),
             'inner_heldout_users':inner_summary,'test_results':results,'event_auc':float(roc_auc_score(label.sum(1),g)),
             'test_model':args.test_model,'test_cutoff':26,'fit_end_exclusive':cutoff(26),'seconds':time.time()-start,
             'train_model':args.train_model,'training_feature_config':train_config,
             'training_feature_config_sha256':file_hash(RUNS/args.train_model/'config.json'),
             'training_feature_cache_sha256':file_hash(RUNS/args.train_model/'calibration_features.npz'),
             'training_feature_checkpoint_selection_end_day':train_config.get('selection_end_day'),
             'training_feature_checkpoint_selection_known_strict':train_config.get('selection_end_day') is not None,
             'row_prediction_file':str(row_path.resolve()),'row_events_file':str(row_events_path.resolve()),
             'event_alignment_verified':True,
             'selection_note':'Rounds chosen using disjoint earlier-period users and global row AUC. Later results are model-selection comparisons, not independent confirmation.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    with open(RUNS/'v3_event_experiments.jsonl','a') as f:f.write(json.dumps(summary)+'\n')

if __name__=='__main__':main()
