"""Reproduce the V16 profile-confidence interpolation ablation in fresh artifacts; no submission upload."""
import argparse
import json
import os
import subprocess
import sys
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--full', action='store_true', help='Refit and generate CSV only if the checks support V16')
    parser.add_argument('--broad', action='store_true', help='Also confirm the January 24 first-user frame')
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = args.out or ROOT/'arena_reproductions'/f'v16_checks_{stamp}'
    run = run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, ARENA_RUNS=str(run), OPENBLAS_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1')

    def execute(script, *parts):
        subprocess.run([sys.executable, str(ROOT/'arena'/script), *map(str, parts)],
                       cwd=ROOT, env=env, check=True)

    execute('test_context_v8.py')
    execute('test_time_history_v10.py')
    execute('test_cold_history_v14.py')
    execute('test_profile_shrinkage_v16.py')
    execute('prepare.py')
    common = ['--mode', 'baseline', '--gates', '--gate-mode', 'global',
              '--supervised-loss', 'pairwise', '--aux-controls', 'observed',
              '--context-rank', '4', '--context-mode', 'bucket', '--aux', '2', '--user-dropout', '.1', '--profile-shrinkage', '20']
    for day in [26, 25]:
        cap = ['--selection-end-day', '26'] if day == 25 else []
        execute('history_model_v3.py', '--day', day, '--tag', 'v16_confidence',
                '--epochs', '12', *common, *cap)
        execute('history_model_v3.py', '--export', run/f'v16_confidence_d32_day{day}_s42')
        factor = run/f'v16_confidence_d32_day{day}_s42'
        execute('compare_v3.py', '--candidate', factor/'valid_pred.npy',
                '--events', factor/'valid_events.npy', '--day', day,
                '--baseline', ROOT/f'arena_runs/v14_coldpath_d32_day{day}_s42/valid_pred.npy',
                '--out', run/f'v16_raw{day}_vs_v14.json')
        result=json.loads((run/f'v16_raw{day}_vs_v14.json').read_text())
        if day==26 and result['global_auc_delta']<=0:
            print('Main raw comparison declined; retain V14. Artifacts:',run)
            return
    early = 'v16_confidence_d32_day25_s42'
    later = 'v16_confidence_d32_day26_s42'
    if args.broad:
        execute('history_model_v3.py','--day','24','--selection-end-day','25',
                '--tag','v16_confidence','--epochs','15',*common)
        p=run/'v16_confidence_d32_day24_s42'
        execute('compare_v3.py','--candidate',p/'valid_pred.npy','--events',p/'valid_events.npy',
                '--day','24','--baseline',ROOT/'arena_runs/v14_coldpath_d32_day24_s42/valid_pred.npy',
                '--out',run/'v16_raw24_vs_v14.json')
    execute('calibrate.py', '--train-model', early, '--test-model', later, '--out', 'v16_cal_confidence')
    cal = run/'v16_cal_confidence'
    factor = run/later
    execute('blend_v3.py', '--raw', factor/'valid_pred.npy', '--combined', cal/'valid_pred.npy',
            '--raw-events', factor/'valid_events.npy', '--combined-events', cal/'valid_events.npy',
            '--out', cal/'valid_blend.npy')
    rounds = json.loads((cal/'config.json').read_text())['rounds']
    execute('event_component_v3.py', '--train-model', early, '--test-model', later,
            '--row-rounds', rounds, '--row-pred', cal/'valid_blend.npy',
            '--row-events', factor/'valid_events.npy', '--out', 'v16_event_confidence')
    event = run/'v16_event_confidence'
    execute('compare_v3.py', '--candidate', event/'valid_half_corrected.npy',
            '--events', event/'valid_events.npy',
            '--baseline', ROOT/'arena_runs/v14_event_coldpath/valid_half_corrected.npy',
            '--bootstrap', '150', '--out', run/'v16_vs_v14.json')
    print('V16 comparison artifacts:', run)
    comparison=json.loads((run/'v16_vs_v14.json').read_text())
    earlier=json.loads((run/'v16_raw25_vs_v14.json').read_text())
    supported=(comparison['bootstrap']['paired_delta_95_percentile_interval'][0]>0
               and earlier['global_auc_delta']>0)
    if args.broad:
        supported=supported and json.loads((run/'v16_raw24_vs_v14.json').read_text())['global_auc_delta']>0
    if not args.full or not supported:
        print('No full refit requested or insufficient evidence. V14 preserved.')
        return
    epochs=json.loads((factor/'best.json').read_text())['epoch']
    execute('history_model_v3.py','--day','26','--tag','v16_confidence_final','--epochs',epochs,'--full',*common)
    full=run/'v16_confidence_final_d32_day26_s42_full'
    with np.load(full/'calibration_features.npz') as c:np.save(full/'evaluation_events.npy',c['events'])
    execute('calibrate_v3.py','--apply-model',cal/'model.txt','--test-model',full.name,
            '--out','v16_final_combiner','--fulltest')
    final=run/'v16_final_combiner'
    execute('blend_v3.py','--raw',full/'evaluation_pred.npy','--combined',final/'evaluation_pred.npy',
            '--raw-events',full/'evaluation_events.npy','--combined-events',final/'events.npy',
            '--out',final/'evaluation_blend.npy')
    execute('v3_event_apply.py','--features',full/'calibration_features.npz','--rows',final/'evaluation_blend.npy',
            '--row-events',final/'events.npy','--event-model',event/'event_model.txt','--out',final/'evaluation_final.npy')
    execute('submission.py','--predictions',final/'evaluation_final.npy',
            '--out',f'submission_techarena_v16_reproduced_{stamp}.csv')


if __name__ == '__main__':
    main()
