"""Reproduce selected validation and final submission in a fresh run directory."""
import argparse,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default=None);args=ap.parse_args()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=ROOT/'arena_reproductions'/stamp
    run.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,ARENA_RUNS=str(run),OPENBLAS_NUM_THREADS='2',VECLIB_MAXIMUM_THREADS='2')
    def execute(script,*args):
        subprocess.run([sys.executable,str(ROOT/'arena'/script),*map(str,args)],cwd=ROOT,env=env,check=True)
    execute('test_contracts.py');execute('prepare.py')
    shared=['--dim','32','--pretrain','0','--svd-init','--svd-iters','25','--reg','1']
    for day in [25,26]:
        execute('factor.py','--day',day,'--epochs','12','--tag','converged_factor',*shared)
        execute('calibrate.py','--export',f'converged_factor_d32_day{day}_s42')
    execute('calibrate.py','--train-model','converged_factor_d32_day25_s42',
            '--test-model','converged_factor_d32_day26_s42','--out','calibrated_clean')
    execute('factor.py','--day','26','--epochs','12','--tag','joint_factor','--aux','.5',*shared)
    execute('calibrate.py','--export','joint_factor_d32_day26_s42')
    execute('ensemble_check.py','--model','joint_factor_d32_day26_s42')
    execute('uncertainty.py')
    execute('factor.py','--day','26','--epochs','3','--tag','final_joint','--aux','.5','--full',*shared)
    final='final_joint_d32_day26_s42_full'
    execute('calibrate.py','--export',final,'--full')
    execute('ensemble_check.py','--model',final,'--full')
    dest=args.out or f'submission_techarena_reproduced_{stamp}.csv'
    execute('submission.py','--predictions',str(run/final/'evaluation_calibrated_clean.npy'),'--out',dest)
    print('Fresh run directory:',run)

if __name__=='__main__':main()
