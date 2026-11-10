"""Reproduce selected time-conditioned preference training in fresh artifacts."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out');ap.add_argument('--broad',action='store_true')
    args=ap.parse_args()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=ROOT/'arena_reproductions'/f'v8_{stamp}';run.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,ARENA_RUNS=str(run),OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    def execute(script,*parts):
        subprocess.run([sys.executable,str(ROOT/'arena'/script),*map(str,parts)],cwd=ROOT,env=env,check=True)
    for test in ['test_contracts.py','test_objectives_v5.py','test_preference_sampling_v5.py','test_context_v8.py']:
        execute(test)
    execute('prepare.py')
    common=['--mode','baseline','--gates','--gate-mode','global','--supervised-loss','pairwise','--aux-controls','observed','--context-rank','4','--context-mode','bucket']
    for day in [25,26]:
        cap=['--selection-end-day','26'] if day==25 else []
        execute('history_model_v3.py','--day',day,'--tag','v8_bucket','--epochs','12',*common,*cap)
        execute('history_model_v3.py','--export',run/f'v8_bucket_d32_day{day}_s42')
    if args.broad:
        execute('history_model_v3.py','--day','24','--selection-end-day','25','--tag','v8_bucket','--epochs','15',*common)
        p=run/'v8_bucket_d32_day24_s42'
        execute('compare_v3.py','--candidate',p/'valid_pred.npy','--events',p/'valid_events.npy','--day','24',
                '--baseline',ROOT/'arena_runs/v5_observed_aux_d32_day24_s42/valid_pred.npy','--out',run/'broad_comparison.json')
    execute('calibrate.py','--train-model','v8_bucket_d32_day25_s42',
            '--test-model','v8_bucket_d32_day26_s42','--out','v8_cal_bucket')
    v=run/'v8_bucket_d32_day26_s42';cal=run/'v8_cal_bucket'
    execute('blend_v3.py','--raw',v/'valid_pred.npy','--combined',cal/'valid_pred.npy','--raw-events',v/'valid_events.npy',
            '--combined-events',cal/'valid_events.npy','--out',cal/'valid_blend.npy')
    rounds=json.loads((cal/'config.json').read_text())['rounds']
    execute('event_component_v3.py','--train-model','v8_bucket_d32_day25_s42','--test-model',v.name,
            '--row-rounds',rounds,'--row-pred',cal/'valid_blend.npy','--row-events',v/'valid_events.npy','--out','v8_event_bucket')
    event=run/'v8_event_bucket'
    execute('compare_v3.py','--candidate',event/'valid_half_corrected.npy','--events',event/'valid_events.npy',
            '--baseline',ROOT/'arena_runs/v5_event_observed/valid_half_corrected.npy','--out',run/'comparison.json','--bootstrap','150')
    execute('history_model_v3.py','--day','26','--tag','v8_bucket_final','--epochs','2','--full',*common)
    full=run/'v8_bucket_final_d32_day26_s42_full'
    with np.load(full/'calibration_features.npz') as c:np.save(full/'evaluation_events.npy',c['events'])
    execute('calibrate_v3.py','--apply-model',cal/'model.txt','--test-model',full.name,'--out','v8_final_combiner','--fulltest')
    final=run/'v8_final_combiner'
    execute('blend_v3.py','--raw',full/'evaluation_pred.npy','--combined',final/'evaluation_pred.npy',
            '--raw-events',full/'evaluation_events.npy','--combined-events',final/'events.npy','--out',final/'evaluation_blend.npy')
    execute('v3_event_apply.py','--features',full/'calibration_features.npz','--rows',final/'evaluation_blend.npy',
            '--row-events',final/'events.npy','--event-model',event/'event_model.txt','--out',final/'evaluation_final.npy')
    execute('submission.py','--predictions',final/'evaluation_final.npy',
            '--out',args.out or f'submission_techarena_v8_reproduced_{stamp}.csv')
    print('Fresh V8 artifacts:',run)

if __name__=='__main__':main()
