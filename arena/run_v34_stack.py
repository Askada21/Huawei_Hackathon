"""Stack the five V31 members using earlier first-user predictions."""
import argparse
import json
import subprocess
import sys

import numpy as np
from scipy.special import logit

from common import ROOT, RUNS, dimensions, load, priors, split
from tree_features import features


MEMBERS = [
    ('v19_checks', 'v19_shrink40', 42),
    ('v22_checks', 'v22_seed43', 43),
    ('v24_checks', 'v24_seed44', 44),
    ('v25_checks', 'v25_signed', 42),
    ('v31_verified', 'v31_decay', 42),
]


def build_cache(directory, day, full=False):
    train = load()
    if full:
        fit = np.arange(len(train['time']))
        data = load('evaluation')
        events = np.arange(len(data['time']))
    else:
        data = train
        fit, events = split(train, day)
    x, names = features(data, events, priors(train, fit, *dimensions()))
    columns = [x]
    for run, tag, seed in MEMBERS:
        if full:
            stem = f'{tag}_final_d32_day26_s{seed}_full'
        else:
            stem = f'{tag}_d32_day{day}_s{seed}'
        member = RUNS / run / stem
        config = json.loads((member/'config.json').read_text())
        if config['day'] != day or config['full'] != full:
            raise ValueError(f'Wrong member period: {member}')
        if day < 26 and config.get('selection_end_day') != day + 1:
            raise ValueError(f'Member checkpoint crosses calibration cutoff: {member}')
        prefix = 'evaluation' if full else 'valid'
        if not np.array_equal(np.load(member/f'{prefix}_events.npy'), events):
            raise ValueError(f'Event alignment differs: {member}')
        p = np.load(member/f'{prefix}_pred.npy')
        if p.shape != (len(events), 10) or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise ValueError(f'Invalid probabilities: {member}')
        z = logit(p.clip(1e-6, 1-1e-6))
        member_columns = [z, z-z.mean(1, keepdims=True), p.sum(1, keepdims=True), p.max(1, keepdims=True)]
        for name, values in zip(('logit', 'centered', 'mass', 'maximum'), member_columns):
            columns.append(np.broadcast_to(values, p.shape).reshape(-1, 1))
            names.append(f'{tag}_{name}')
    directory.mkdir(exist_ok=False)
    np.savez(directory/'calibration_features.npz', x=np.column_stack(columns).astype('float32'),
             names=names, events=events, user=data['user'][events])
    (directory/'config.json').write_text(json.dumps(dict(day=day, full=full, members=MEMBERS), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='v34_stack')
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    run = RUNS / args.out
    run.mkdir(exist_ok=False)

    def execute(script, *options):
        subprocess.run([sys.executable, str(ROOT/'arena'/script), *map(str, options)],
                       cwd=ROOT, check=True)

    def blend(raw, raw_events, base, base_events, dest):
        execute('blend_v3.py', '--raw', raw, '--raw-events', raw_events, '--combined', base,
                '--combined-events', base_events, '--raw-weight', .5, '--out', dest)

    for day in (24, 25, 26):
        build_cache(run/f'day{day}', day)
    for day in (25, 26):
        cal = run/f'cal{day}'
        sources = [run/f'day{d}' for d in range(24, day)]
        execute('calibrate_v3.py', '--train-models', *sources, '--test-model', run/f'day{day}',
                '--max-rounds', 500, '--patience', 35, '--out', cal)
        reference = RUNS/'v31_verified'
        if day == 25:
            reference /= 'v31_decay_d32_day25_s42'
        else:
            reference /= 'v31_event_confidence'
        dest = cal/'blend.npy'
        blend(cal/'valid_pred.npy', cal/'valid_events.npy', reference/'ensemble_valid.npy',
              reference/'valid_events.npy', dest)
        execute('compare_v3.py', '--candidate', dest, '--events', cal/'valid_events.npy',
                '--baseline', reference/'ensemble_valid.npy', '--day', day,
                '--bootstrap', 150 if day == 26 else 0, '--out', run/f'comparison{day}.json')

    main = json.loads((run/'comparison26.json').read_text())
    earlier = json.loads((run/'comparison25.json').read_text())
    if not args.full or earlier['global_auc_delta'] <= 0 or main['bootstrap']['paired_delta_95_percentile_interval'][0] <= 0:
        return
    build_cache(run/'evaluation', 26, full=True)
    final = run/'final'
    execute('calibrate_v3.py', '--apply-model', run/'cal26/model.txt',
            '--test-model', run/'evaluation', '--fulltest', '--out', final)
    reference = RUNS/'v31_verified/v31_final_combiner'
    blend(final/'evaluation_pred.npy', final/'events.npy', reference/'ensemble_evaluation.npy',
          reference/'events.npy', final/'blend.npy')
    execute('submission.py', '--predictions', final/'blend.npy',
            '--out', f'submission_techarena_{run.name}.csv')


if __name__ == '__main__':
    main()
