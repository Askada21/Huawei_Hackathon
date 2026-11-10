"""Add the lighter signed graph to V31 at a fixed one-sixth logit weight."""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

from common import ROOT, RUNS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--reuse-checks', action='store_true')
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    out = RUNS / args.out
    member = RUNS/'v30_checks'
    incumbent = RUNS/'v31_verified'

    def execute(script, *options):
        subprocess.run([sys.executable, str(ROOT/'arena'/script), *map(str, options)],
                       cwd=ROOT, check=True, env=dict(os.environ, OPENBLAS_NUM_THREADS='1',
                                                     VECLIB_MAXIMUM_THREADS='1'))

    def blend(raw, events, base, base_events, dest):
        execute('blend_v3.py', '--raw', raw, '--raw-events', events, '--combined', base,
                '--combined-events', base_events, '--raw-weight', 1/6, '--out', dest)

    def compare(raw, events, base, base_events, name, day, bootstrap=0):
        dest = out/f'{name}.npy'
        blend(raw, events, base, base_events, dest)
        execute('compare_v3.py', '--candidate', dest, '--events', events, '--baseline', base,
                '--day', day, '--bootstrap', bootstrap, '--out', out/f'{name}.json')

    if not args.reuse_checks:
        out.mkdir(exist_ok=False)
        for day in (24, 25, 26):
            a = member/f'v30_signed025_d32_day{day}_s42'
            b = incumbent/f'v31_decay_d32_day{day}_s42'
            compare(a/'valid_pred.npy', a/'valid_events.npy', b/'ensemble_valid.npy',
                    b/'valid_events.npy', f'raw{day}', day)
        a = member/'v30_event_confidence'
        b = incumbent/'v31_event_confidence'
        compare(a/'valid_half_corrected.npy', a/'valid_events.npy', b/'ensemble_valid.npy',
                b/'valid_events.npy', 'final', 26, 150)

    comparison = json.loads((out/'final.json').read_text())
    supported = all(json.loads((out/f'raw{day}.json').read_text())['global_auc_delta'] > 0
                    for day in (24, 25, 26))
    if not args.full or not supported or comparison['bootstrap']['paired_delta_95_percentile_interval'][0] <= 0:
        return

    epochs = json.loads((member/'v30_signed025_d32_day26_s42/best.json').read_text())['epoch']
    tag = f'{out.name}_member'
    execute('history_model_v3.py', '--day', 26, '--mode', 'baseline', '--gates',
            '--gate-mode', 'global', '--supervised-loss', 'pairwise', '--aux-controls',
            'observed', '--context-rank', 4, '--context-mode', 'bucket', '--aux', 2,
            '--user-dropout', .1, '--profile-shrinkage', 40, '--spectral-negative-weight',
            .025, '--tag', tag, '--epochs', epochs, '--full')
    full = RUNS/f'{tag}_d32_day26_s42_full'
    with np.load(full/'calibration_features.npz') as cache:
        np.save(full/'evaluation_events.npy', cache['events'])
    final = out/'evaluation'
    execute('calibrate_v3.py', '--apply-model', member/'v30_cal_confidence/model.txt',
            '--test-model', full, '--fulltest', '--out', final)
    execute('blend_v3.py', '--raw', full/'evaluation_pred.npy', '--raw-events',
            full/'evaluation_events.npy', '--combined', final/'evaluation_pred.npy',
            '--combined-events', final/'events.npy', '--out', final/'row_blend.npy')
    execute('v3_event_apply.py', '--features', full/'calibration_features.npz',
            '--rows', final/'row_blend.npy', '--row-events', final/'events.npy',
            '--event-model', member/'v30_event_confidence/event_model.txt',
            '--out', final/'member.npy')
    reference = incumbent/'v31_final_combiner'
    blend(final/'member.npy', final/'events.npy', reference/'ensemble_evaluation.npy',
          reference/'events.npy', final/'blend.npy')
    execute('submission.py', '--predictions', final/'blend.npy',
            '--out', f'submission_techarena_{out.name}.csv')


if __name__ == '__main__':
    main()
