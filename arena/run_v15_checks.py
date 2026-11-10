"""Test nonlinear row combination using frozen V14 artifacts; no uploads."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--submission',action='store_true',help='Generate a new CSV only if both comparisons support V15')
    args=ap.parse_args();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=(args.out or ROOT/'arena_reproductions'/f'v15_checks_{stamp}').resolve()
    run.mkdir(parents=True,exist_ok=False)
    source=ROOT/'arena_runs'
    def factor(day):return source/f'v14_coldpath_d32_day{day}_s42'
    env=dict(os.environ,ARENA_RUNS=str(run),OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    def execute(script,*parts):
        subprocess.run([sys.executable,str(ROOT/'arena'/script),*map(str,parts)],cwd=ROOT,env=env,check=True)
    # Reuse the selected frozen factor feature caches; V15 changes no factors.
    for day in [24,25,26]:
        p=factor(day)
        if not (p/'calibration_features.npz').exists():
            execute('history_model_v3.py','--export',p)
    def calibrate(day,leaves,out):
        execute('calibrate.py','--train-model',factor(day-1),'--test-model',factor(day),
                '--leaves',leaves,'--out',out)
    def blend(day,out):
        p=run/out
        execute('blend_v3.py','--raw',factor(day)/'valid_pred.npy','--combined',p/'valid_pred.npy',
                '--raw-events',factor(day)/'valid_events.npy','--combined-events',p/'valid_events.npy',
                '--out',p/'valid_blend.npy')
    calibrate(26,31,'v15_cal31_day26');blend(26,'v15_cal31_day26')
    maincal=run/'v15_cal31_day26'
    eventmodel=source/'v14_event_coldpath/event_model.txt'
    execute('v3_event_apply.py','--features',factor(26)/'calibration_features.npz',
            '--rows',maincal/'valid_blend.npy','--row-events',maincal/'valid_events.npy',
            '--event-model',eventmodel,'--out',maincal/'valid_final.npy')
    execute('compare_v3.py','--candidate',maincal/'valid_final.npy','--events',maincal/'valid_events.npy',
            '--baseline',source/'v14_event_coldpath/valid_half_corrected.npy','--bootstrap','150',
            '--out',run/'v15_final_vs_v14.json')
    for leaves in [7,31]:
        name=f'v15_screen{leaves}_day25';calibrate(25,leaves,name);blend(25,name)
    early=run/'v15_screen31_day25'
    execute('compare_v3.py','--candidate',early/'valid_blend.npy','--events',early/'valid_events.npy',
            '--baseline',run/'v15_screen7_day25/valid_blend.npy','--day','25',
            '--out',run/'v15_earlier_vs7.json')
    finalcheck=json.loads((run/'v15_final_vs_v14.json').read_text())
    earlier=json.loads((run/'v15_earlier_vs7.json').read_text())
    supported=(finalcheck['bootstrap']['paired_delta_95_percentile_interval'][0]>0
               and earlier['global_auc_delta']>0)
    if not args.submission or not supported:
        print('V15 comparison artifacts:',run,'; no new CSV generated.')
        return
    full=source/'v14_coldpath_final_d32_day26_s42_full'
    execute('calibrate_v3.py','--apply-model',maincal/'model.txt','--test-model',full,
            '--out','v15_final_combiner','--fulltest')
    final=run/'v15_final_combiner'
    execute('blend_v3.py','--raw',full/'evaluation_pred.npy','--combined',final/'evaluation_pred.npy',
            '--raw-events',full/'evaluation_events.npy','--combined-events',final/'events.npy',
            '--out',final/'evaluation_blend.npy')
    execute('v3_event_apply.py','--features',full/'calibration_features.npz','--rows',final/'evaluation_blend.npy',
            '--row-events',final/'events.npy','--event-model',eventmodel,'--out',final/'evaluation_final.npy')
    execute('submission.py','--predictions',final/'evaluation_final.npy',
            '--out',f'submission_techarena_v15_reproduced_{stamp}.csv')

if __name__=='__main__':main()
