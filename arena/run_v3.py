"""Reproduce the selected V3 recipe in a new directory, preserving old runs.

Retrains V3 factor models, earlier-period row/event combiners, and full refit.
"""
import os,sys,subprocess,argparse
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out');args=ap.parse_args()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=ROOT/'arena_reproductions'/f'v3_{stamp}';run.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,ARENA_RUNS=str(run),OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    def execute(script,*parts):
        subprocess.run([sys.executable,str(ROOT/'arena'/script),*map(str,parts)],cwd=ROOT,env=env,check=True)
    # The previous delivered predictions are used for comparison only, not fitting.
    baseline=ROOT/'arena_runs/joint_factor_d32_day26_s42/valid_calibrated_clean.npy'
    if not baseline.exists():raise FileNotFoundError(f'Restore baseline comparison artifact: {baseline}')
    for test in ['test_contracts.py','test_history_model_v3.py','test_calibrate_v3.py','test_v3_event_contracts.py']:
        execute(test)
    execute('prepare.py')
    shared=['--mode','baseline','--gates','--gate-mode','global']
    for day in [25,26]:
        more=['--selection-end-day','26'] if day==25 else []
        execute('history_model_v3.py','--day',day,'--tag','v3_global_gate','--epochs','10',*shared,*more)
        execute('history_model_v3.py','--export',run/f'v3_global_gate_d32_day{day}_s42')
    execute('calibrate.py','--train-model','v3_global_gate_d32_day25_s42',
            '--test-model','v3_global_gate_d32_day26_s42','--out','v3_cal_global_single')
    v=run/'v3_global_gate_d32_day26_s42';cal=run/'v3_cal_global_single'
    execute('blend_v3.py','--raw',v/'valid_pred.npy','--combined',cal/'valid_pred.npy',
            '--raw-events',v/'valid_events.npy','--combined-events',cal/'valid_events.npy','--out',cal/'valid_blend.npy')
    execute('event_component_v3.py','--train-model','v3_global_gate_d32_day25_s42',
            '--test-model',v.name,'--row-rounds','500','--row-pred',cal/'valid_blend.npy',
            '--row-events',v/'valid_events.npy','--out','v3_event_global_strict')
    execute('v3_event_apply.py','--features',v/'calibration_features.npz','--rows',cal/'valid_blend.npy',
            '--row-events',v/'valid_events.npy','--event-model',run/'v3_event_global_strict/event_model.txt',
            '--out',cal/'valid_event_corrected.npy')
    execute('compare_v3.py','--candidate',cal/'valid_event_corrected.npy','--events',v/'valid_events.npy',
            '--out',run/'comparison.json','--bootstrap','150')
    execute('history_model_v3.py','--day','26','--tag','v3_final_global','--epochs','2','--full',*shared)
    full=run/'v3_final_global_d32_day26_s42_full'
    with np.load(full/'calibration_features.npz') as c:np.save(full/'evaluation_events.npy',c['events'])
    execute('calibrate_v3.py','--apply-model',cal/'model.txt','--test-model',full.name,'--out','v3_final_combiner','--fulltest')
    final=run/'v3_final_combiner'
    execute('blend_v3.py','--raw',full/'evaluation_pred.npy','--combined',final/'evaluation_pred.npy',
            '--raw-events',full/'evaluation_events.npy','--combined-events',final/'events.npy','--out',final/'evaluation_blend.npy')
    execute('v3_event_apply.py','--features',full/'calibration_features.npz','--rows',final/'evaluation_blend.npy',
            '--row-events',final/'events.npy','--event-model',run/'v3_event_global_strict/event_model.txt',
            '--out',final/'evaluation_final.npy')
    execute('submission.py','--predictions',final/'evaluation_final.npy',
            '--out',args.out or f'submission_techarena_v3_reproduced_{stamp}.csv')
    print('Reproduction artifacts:',run)

if __name__=='__main__':main()
