"""Apply a frozen earlier-period calibrator to a complementary factor model."""
import argparse
import numpy as np
import lightgbm as lgb
from common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--full',action='store_true');args=ap.parse_args()
    m=RUNS/args.model;c=np.load(m/'calibration_features.npz')
    model=lgb.Booster(model_file=str(RUNS/'calibrated_clean'/'model.txt'))
    p=model.predict(c['x'],num_threads=4).reshape(-1,10)
    np.save(m/('evaluation_calibrated_clean.npy' if args.full else 'valid_calibrated_clean.npy'),p)
    if not args.full:
        record(args.model+'_frozen_calibration',26,c['y'],p)
        base=np.load(RUNS/'calibrated_clean'/'valid_pred.npy')
        assert np.array_equal(c['events'],np.load(RUNS/'calibrated_clean'/'valid_events.npy'))
        for weight in [.25,.5,.75]:
            blend=(1-weight)*base+weight*p
            record(args.model+f'_blend{weight}',26,c['y'],blend)

if __name__=='__main__':main()
