"""Frozen earlier-period combiner applied to new compatible factor features."""
import argparse,json
import numpy as np
import lightgbm as lgb
from common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--full',action='store_true');args=ap.parse_args()
    out=RUNS/args.model;c=np.load(out/'calibration_features.npz')
    booster=lgb.Booster(model_file=str(RUNS/'calibrated_clean/model.txt'))
    p=booster.predict(c['x'],num_threads=2).reshape(-1,10)
    np.save(out/('evaluation_clean.npy' if args.full else 'valid_clean.npy'),p)
    if not args.full:
        assert json.loads((out/'config.json').read_text())['day']==26,'Frozen combiner only valid on cutoff26 or full evaluation'
        base=np.load(RUNS/'joint_factor_d32_day26_s42/valid_calibrated_clean.npy')
        assert np.array_equal(c['events'],np.load(RUNS/'joint_factor_d32_day26_s42/valid_events.npy'))
        record(args.model+'_clean',26,c['y'],p)
        blend=(p+base)*.5;np.save(out/'valid_blend_half.npy',blend)
        record(args.model+'_blend_half',26,c['y'],blend)

if __name__=='__main__':main()
