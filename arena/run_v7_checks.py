"""Reproduce the V7 model comparisons in a fresh output directory."""
import argparse,os,sys,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    run=ROOT/'arena_reproductions'/('v7_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    run.mkdir(parents=True,exist_ok=False)
    for n in ['v5_observed_aux_d32_day25_s42','v5_observed_aux_d32_day26_s42','v5_event_observed']:
        (run/n).symlink_to(ROOT/'arena_runs'/n,target_is_directory=True)
    env=dict(os.environ,ARENA_RUNS=str(run),OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    def execute(script,*args):
        subprocess.run([sys.executable,str(ROOT/'arena'/script),*map(str,args)],cwd=ROOT,env=env,check=True)
    execute('test_adapt_v7.py');execute('test_lift_v7.py')
    execute('adapt_v7.py','--day','26','--out','v7_adapt26')
    adapted=run/'v7_adapt26';source=run/'v5_observed_aux_d32_day26_s42'
    np.save(adapted/'valid_blend.npy',.5*np.load(adapted/'valid_pred.npy')+.5*np.load(source/'valid_pred.npy'))
    execute('compare_v3.py','--candidate',adapted/'valid_pred.npy','--events',adapted/'valid_events.npy',
            '--baseline',source/'valid_pred.npy','--out',run/'adapt_comparison.json')
    for day in [25,26]:execute('lift_v7.py','--day',day,'--out',f'v7_lift{day}')
    execute('calibrate.py','--train-model','v7_lift25','--test-model','v7_lift26','--out','v7_lift_combiner')
    cal=run/'v7_lift_combiner'
    execute('blend_v3.py','--raw',source/'valid_pred.npy','--combined',cal/'valid_pred.npy',
            '--raw-events',source/'valid_events.npy','--combined-events',cal/'valid_events.npy','--out',cal/'blend.npy')
    execute('v3_event_apply.py','--features',source/'calibration_features.npz','--rows',cal/'blend.npy',
            '--row-events',cal/'valid_events.npy','--event-model',run/'v5_event_observed/event_model.txt','--out',cal/'final.npy')
    execute('compare_v3.py','--candidate',cal/'final.npy','--events',cal/'valid_events.npy',
            '--baseline',ROOT/'arena_runs/v5_event_observed/valid_half_corrected.npy','--out',run/'comparison.json','--bootstrap','150')
    print('Review the comparison before selecting any replacement:',run)
if __name__=='__main__':main()
