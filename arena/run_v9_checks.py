"""Reproduce the small V9 gate ablation in fresh artifacts; no submission upload."""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run = args.out or ROOT/'arena_reproductions'/f'v9_checks_{stamp}'
    run = run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, ARENA_RUNS=str(run), OPENBLAS_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1')

    def execute(script, *parts):
        subprocess.run([sys.executable, str(ROOT/'arena'/script), *map(str, parts)],
                       cwd=ROOT, env=env, check=True)

    execute('test_context_v8.py')
    execute('prepare.py')
    common = ['--mode', 'baseline', '--gates', '--gate-mode', 'global',
              '--supervised-loss', 'pairwise', '--aux-controls', 'observed',
              '--context-rank', '4', '--context-mode', 'bucket', '--temporal-gates']
    for day in [25, 26]:
        cap = ['--selection-end-day', '26'] if day == 25 else []
        execute('history_model_v3.py', '--day', day, '--tag', 'v9_controlled',
                '--epochs', '12', *common, *cap)
        execute('history_model_v3.py', '--export', run/f'v9_controlled_d32_day{day}_s42')
        factor = run/f'v9_controlled_d32_day{day}_s42'
        execute('compare_v3.py', '--candidate', factor/'valid_pred.npy',
                '--events', factor/'valid_events.npy', '--day', day,
                '--baseline', ROOT/f'arena_runs/v8_bucket_d32_day{day}_s42/valid_pred.npy',
                '--out', run/f'v9_raw{day}_vs_v8.json')
    early = 'v9_controlled_d32_day25_s42'
    later = 'v9_controlled_d32_day26_s42'
    execute('calibrate.py', '--train-model', early, '--test-model', later, '--out', 'v9_cal_controlled')
    cal = run/'v9_cal_controlled'
    factor = run/later
    execute('blend_v3.py', '--raw', factor/'valid_pred.npy', '--combined', cal/'valid_pred.npy',
            '--raw-events', factor/'valid_events.npy', '--combined-events', cal/'valid_events.npy',
            '--out', cal/'valid_blend.npy')
    rounds = json.loads((cal/'config.json').read_text())['rounds']
    execute('event_component_v3.py', '--train-model', early, '--test-model', later,
            '--row-rounds', rounds, '--row-pred', cal/'valid_blend.npy',
            '--row-events', factor/'valid_events.npy', '--out', 'v9_event_controlled')
    event = run/'v9_event_controlled'
    execute('compare_v3.py', '--candidate', event/'valid_half_corrected.npy',
            '--events', event/'valid_events.npy',
            '--baseline', ROOT/'arena_runs/v8_event_bucket/valid_half_corrected.npy',
            '--bootstrap', '150', '--out', run/'v9_vs_v8.json')
    print('V9 comparison artifacts:', run)
    print('Inspect the temporal results and comparison before any full refit.')


if __name__ == '__main__':
    main()
