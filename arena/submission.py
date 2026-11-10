"""Write a fresh submission and verify it against the original evaluation CSV."""
import argparse,csv,hashlib,json,math
from pathlib import Path
import numpy as np
from common import ROOT,DATA,RUNS

def verify(path):
    expected=set()
    with open(ROOT/'evaluation.csv',newline='') as f:
        for r in csv.DictReader(f):
            if r['rowid'] in expected:raise ValueError('Duplicate rowid in evaluation')
            expected.add(r['rowid'])
    seen=set();lo=1;hi=0;total=0.0
    with open(path,newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==['rowid','predicted_probability']
        for r in reader:
            assert set(r)=={'rowid','predicted_probability'}
            rowid=r['rowid'];p=float(r['predicted_probability'])
            assert rowid in expected and rowid not in seen
            assert math.isfinite(p) and 0<=p<=1
            seen.add(rowid);lo=min(lo,p);hi=max(hi,p);total+=p
    assert len(expected)==529390 and seen==expected
    with open(path,'rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    return dict(file=str(path.resolve()),rows=len(seen),min_probability=lo,max_probability=hi,
                mean_probability=total/len(seen),sha256=digest,header=['rowid','predicted_probability'],
                rowids_exactly_match_evaluation=True,all_values_finite_and_in_range=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--predictions',nargs='+',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--verify-only',action='store_true');args=ap.parse_args()
    path=ROOT/args.out
    if not args.verify_only:
        if path.exists():raise FileExistsError(f'Refusing to overwrite {path}')
        ps=[np.load(ROOT/p).reshape(-1) for p in args.predictions]
        assert all(len(p)==529390 and np.isfinite(p).all() and (p>=0).all() and (p<=1).all() for p in ps)
        pred=np.mean(ps,axis=0);rowids=np.load(DATA/'evaluation_rowids.npy')
        with open(path,'x',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['rowid','predicted_probability'])
            for rowid,p in zip(rowids,pred):w.writerow([rowid,f'{p:.10g}'])
    report=verify(path);report['prediction_sources']=args.predictions
    path.with_suffix('.verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
