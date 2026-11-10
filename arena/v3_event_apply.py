"""Apply fixed 50/50 event-mass correction without accessing any labels."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import lightgbm as lgb
from event_component_v3 import event_features,validate_event_alignment

def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def validate_output_path(path):
    path=Path(path)
    if path.suffix!='.npy':
        raise ValueError('Output must end in the exact .npy suffix; np.save would otherwise append a different filename')
    if path.exists():raise FileExistsError(f'Refusing to overwrite {path}')
    return path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features',required=True);ap.add_argument('--rows',required=True)
    ap.add_argument('--row-events',required=True,help='Explicit event ordering array aligned to --rows, including evaluation events')
    ap.add_argument('--event-model',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();dest=validate_output_path(args.out)
    cache=np.load(args.features)
    # Deliberately load only feature data. Evaluation label placeholders are never read.
    c={k:cache[k] for k in ['x','names','p']}
    x,names=event_features(c)
    model=lgb.Booster(model_file=args.event_model)
    assert names==model.feature_name(),'Event feature contract differs from training'
    rows=np.load(args.rows).reshape(-1,10)
    row_events=np.load(args.row_events)
    validate_event_alignment(cache['events'],row_events,rows.size)
    assert len(rows)==len(x) and np.isfinite(rows).all()
    assert (rows>=0).all() and (rows<=1).all()
    mass=rows.sum(1,keepdims=True);assert (mass>0).all()
    g=model.predict(x,num_threads=2)
    assert np.isfinite(g).all() and (g>=0).all() and (g<=1).all()
    corrected=.5*rows+.5*g[:,None]*(rows/mass)
    assert np.isfinite(corrected).all() and (corrected>=0).all() and (corrected<=1).all()
    model_summary=Path(args.event_model).parent/'summary.json'
    provenance=json.loads(model_summary.read_text())
    assert provenance['training_events']==35230
    assert provenance['fit_end_exclusive']==1737849600
    selection_end=provenance.get('training_feature_checkpoint_selection_end_day')
    if selection_end is not None and selection_end>26:
        raise ValueError('Event model training features selected checkpoints using the outer period')
    dest.parent.mkdir(parents=True,exist_ok=True);np.save(dest,corrected)
    report={'features':str(Path(args.features).resolve()),'rows':str(Path(args.rows).resolve()),
            'event_model':str(Path(args.event_model).resolve()),'output':str(dest.resolve()),
            'row_events':str(Path(args.row_events).resolve()),'event_alignment_verified':True,
            'events':len(rows),'rows_count':rows.size,'fixed_correction_weight':.5,
            'formula':'0.5 * row_probability + 0.5 * event_probability * row_probability / sum_event_row_probabilities',
            'event_model_fitting':'January25 first-user OOT events strictly before January26; 35230 events, excluding8734 later events',
            'event_model_fit_end_exclusive_utc':'2025-01-26T00:00:00Z',
            'training_feature_checkpoint_selection_end_day':selection_end,
            'training_feature_checkpoint_selection_known_strict':provenance.get('training_feature_checkpoint_selection_known_strict',False),
            'prediction_labels_read':False,'output_sha256':sha(dest),'event_model_sha256':sha(args.event_model),
            'all_probabilities_finite_and_in_range':True}
    dest.with_suffix('.provenance.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
