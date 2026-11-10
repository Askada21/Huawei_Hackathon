"""Compare a fixed seven-member ensemble with V32."""
import argparse
import json
import subprocess
import sys

from common import ROOT, RUNS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='v37_join')
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    out = RUNS / args.out
    out.mkdir(exist_ok=False)
    base = RUNS/'v31_verified'
    a = RUNS/'v30_checks'
    b = RUNS/'v32_checks'

    def call(script, *options):
        subprocess.run([sys.executable, str(ROOT/'arena'/script), *map(str, options)],
                       cwd=ROOT, check=True)

    def blend(raw, raw_events, combined, combined_events, weight, dest):
        call('blend_v3.py', '--raw', raw, '--raw-events', raw_events, '--combined',
             combined, '--combined-events', combined_events, '--raw-weight', weight,
             '--out', dest)

    def join(first, second, reference, name):
        pair = out/f'{name}_pair.npy'
        result = out/f'{name}.npy'
        blend(*first, *second, .5, pair)
        blend(pair, first[1], *reference, 2/7, result)
        return result

    for day in (24, 25, 26):
        pa = a/f'v30_signed025_d32_day{day}_s42'
        pb = b/f'v32_adaptive_d32_day{day}_s42'
        pc = base/f'v31_decay_d32_day{day}_s42'
        name = f'raw{day}'
        p = join((pa/'valid_pred.npy', pa/'valid_events.npy'),
                 (pb/'valid_pred.npy', pb/'valid_events.npy'),
                 (pc/'ensemble_valid.npy', pc/'valid_events.npy'), name)
        call('compare_v3.py', '--candidate', p, '--events', pc/'valid_events.npy',
             '--baseline', pb/'ensemble_valid.npy', '--day', day, '--out', out/f'{name}.json')

    pa = a/'v30_event_confidence'
    pb = b/'v32_event_confidence'
    pc = base/'v31_event_confidence'
    p = join((pa/'valid_half_corrected.npy', pa/'valid_events.npy'),
             (pb/'valid_half_corrected.npy', pb/'valid_events.npy'),
             (pc/'ensemble_valid.npy', pc/'valid_events.npy'), 'final')
    call('compare_v3.py', '--candidate', p, '--events', pc/'valid_events.npy',
         '--baseline', pb/'ensemble_valid.npy', '--bootstrap', 150, '--out', out/'final.json')
    r = json.loads((out/'final.json').read_text())
    supported = all(json.loads((out/f'raw{day}.json').read_text())['global_auc_delta'] > 0
                    for day in (24, 25, 26))
    if not args.full or not supported or r['bootstrap']['paired_delta_95_percentile_interval'][0] <= 0:
        return
    pa = RUNS/'v36_blend/evaluation'
    pb = b/'v32_final_combiner'
    pc = base/'v31_final_combiner'
    p = join((pa/'member.npy', pa/'events.npy'), (pb/'evaluation_final.npy', pb/'events.npy'),
             (pc/'ensemble_evaluation.npy', pc/'events.npy'), 'evaluation')
    call('submission.py', '--predictions', p, '--out', f'submission_techarena_{out.name}.csv')


if __name__ == '__main__':
    main()
