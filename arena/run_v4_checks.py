"""Reproduce V4 experiments in a fresh directory; never overwrite submissions."""
import os,sys,subprocess,argparse
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    run=ROOT/'arena_reproductions'/('v4_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    run.mkdir(parents=True,exist_ok=False)
    for name in ['v3_global_gate_d32_day25_s42','v3_global_gate_d32_day26_s42','v3_event_global_strict']:
        (run/name).symlink_to(ROOT/'arena_runs'/name,target_is_directory=True)
    env=dict(os.environ,ARENA_RUNS=str(run),OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    def execute(script,*args):
        subprocess.run([sys.executable,str(ROOT/'arena'/script),*map(str,args)],cwd=ROOT,env=env,check=True)
    for day in [23,24]:
        execute('history_model_v3.py','--day',day,'--selection-end-day',day+1,'--mode','baseline',
                '--gates','--gate-mode','global','--tag','v4_window','--epochs','15')
        execute('history_model_v3.py','--export',run/f'v4_window_d32_day{day}_s42')
    execute('history_model_v3.py','--day','26','--mode','baseline','--gates','--gate-mode','global',
            '--init-normalization','soft','--tag','v4_soft','--epochs','10')
    execute('calibrate_v3.py','--train-models','v4_window_d32_day23_s42',
            '--test-model','v4_window_d32_day24_s42','--out','v4_cal_weekday_backtest','--max-rounds','1000')
    execute('calibrate_v3.py','--train-models','v4_window_d32_day24_s42','v3_global_gate_d32_day25_s42',
            '--test-model','v3_global_gate_d32_day26_s42','--out','v4_cal_mixed','--max-rounds','1000')
    cal=run/'v4_cal_mixed';raw=run/'v3_global_gate_d32_day26_s42'
    execute('blend_v3.py','--raw',raw/'valid_pred.npy','--combined',cal/'valid_pred.npy',
            '--raw-events',raw/'valid_events.npy','--combined-events',cal/'valid_events.npy','--out',cal/'valid_blend.npy')
    execute('v3_event_apply.py','--features',raw/'calibration_features.npz','--rows',cal/'valid_blend.npy',
            '--row-events',cal/'valid_events.npy','--event-model',run/'v3_event_global_strict/event_model.txt','--out',cal/'valid_final.npy')
    execute('compare_v3.py','--candidate',cal/'valid_final.npy','--events',cal/'valid_events.npy',
            '--baseline',ROOT/'arena_runs/v3_cal_global_single/valid_event_strict.npy',
            '--out',run/'comparison.json','--bootstrap','150')
    print('Read the comparison before making any submission:',run)
if __name__=='__main__':main()
