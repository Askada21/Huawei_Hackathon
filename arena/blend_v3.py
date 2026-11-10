"""Fixed weight blend with explicit event alignment and immutable output."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.special import expit,logit

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',required=True);ap.add_argument('--combined',required=True)
    ap.add_argument('--raw-events',required=True);ap.add_argument('--combined-events',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--raw-weight',type=float,default=.25)
    args=ap.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    if out.suffix!='.npy' or not 0<=args.raw_weight<=1:raise ValueError('Need .npy output and valid weight')
    re=np.load(args.raw_events);ce=np.load(args.combined_events)
    if not np.array_equal(re,ce):raise ValueError('Event order differs')
    raw=np.load(args.raw);combined=np.load(args.combined)
    if raw.shape!=combined.shape or raw.shape!=(len(re),10):raise ValueError('Prediction shape differs')
    for p in [raw,combined]:
        if not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('Invalid probability')
    p=expit(args.raw_weight*logit(raw.clip(1e-6,1-1e-6))+(1-args.raw_weight)*logit(combined.clip(1e-6,1-1e-6)))
    out.parent.mkdir(exist_ok=True,parents=True);np.save(out,p)
    meta=dict(vars(args),events=len(re),probabilities_finite_in_range=True,
              events_sha256=hashlib.sha256(re.tobytes()).hexdigest())
    out.with_suffix('.provenance.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta),flush=True)

if __name__=='__main__':main()
