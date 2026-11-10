"""Fit a conditional slate ranker on the earlier V34 feature caches."""
import argparse
import json
import subprocess
import sys

import lightgbm as lgb
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax

from calibrate_v3 import feature_cache, period_mask
from common import ROOT, RUNS, load, split
from run_v34_stack import build_cache


def fit_ranker(source, outer_day):
    data = load()
    features, labels, users = [], [], []
    names = None
    for day in range(24, outer_day):
        cache = feature_cache(source/f'day{day}')
        if not np.array_equal(cache['events'], split(data, day)[1]):
            raise ValueError('Fitting cache is not the expected first-user frame')
        eligible = period_mask(data['time'], cache['events'], day, outer_day)
        events = cache['events'][eligible]
        y = data['label'][events]
        clicked = y.sum(1) == 1
        if names is not None and names != cache['names'].tolist():
            raise ValueError('Inconsistent feature schema')
        names = cache['names'].tolist()
        features.append(cache['x'].reshape(-1, 10, len(names))[eligible][clicked])
        labels.append(y[clicked])
        users.append(data['user'][events[clicked]])
    x = np.concatenate(features)
    y = np.concatenate(labels)
    hold = np.concatenate(users) % 5 == 0

    def dataset(mask):
        return lgb.Dataset(x[mask].reshape(-1, len(names)), label=y[mask].ravel(),
                           group=np.full(mask.sum(), 10), feature_name=names)

    params = dict(objective='lambdarank', metric='ndcg', eval_at=[10],
                  lambdarank_truncation_level=10, num_leaves=15, min_data_in_leaf=300,
                  learning_rate=.03, lambda_l2=30, num_threads=2, verbosity=-1, seed=42)
    trial = lgb.train(params, dataset(~hold), num_boost_round=400, valid_sets=[dataset(hold)],
                      callbacks=[lgb.early_stopping(35), lgb.log_evaluation(50)])
    scores = trial.predict(x[hold].reshape(-1, len(names))).reshape(-1, 10)
    clicked = y[hold].argmax(axis=1)
    temperature = minimize_scalar(
        lambda t: -log_softmax(scores / t, axis=1)[np.arange(len(scores)), clicked].mean(),
        bounds=(.1, 10), method='bounded').x
    model = lgb.train(params, dataset(np.ones(len(x), dtype=bool)), num_boost_round=trial.best_iteration)
    return model, float(temperature), len(x)


def reallocate(model, temperature, cache, reference, event_path):
    events = np.load(event_path)
    if not np.array_equal(events, cache['events']):
        raise ValueError('Ranker and incumbent event alignment differs')
    if model.feature_name() != cache['names'].tolist():
        raise ValueError('Ranker feature schema differs')
    p = np.load(reference)
    if p.shape != (len(events), 10) or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Invalid incumbent probabilities')
    scores = model.predict(cache['x']).reshape(-1, 10)
    conditional = np.exp(log_softmax(scores / temperature, axis=1))
    return (.5 * p + .5 * p.sum(1, keepdims=True).clip(0, 1) * conditional).clip(0, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='v35_rank')
    parser.add_argument('--source', default='v34_stack')
    parser.add_argument('--full', action='store_true')
    args = parser.parse_args()
    out, source = RUNS/args.out, RUNS/args.source
    out.mkdir(exist_ok=False)
    ref = RUNS/'v31_verified'

    def execute(script, *options):
        subprocess.run([sys.executable, str(ROOT/'arena'/script), *map(str, options)],
                       cwd=ROOT, check=True)

    for day in (25, 26):
        model, temperature, count = fit_ranker(source, day)
        model.save_model(str(out/f'model{day}.txt'))
        (out/f'config{day}.json').write_text(json.dumps(dict(
            cutoff=day, temperature=temperature, training_clicked_events=count,
            rounds=model.current_iteration(), candidate_weight=.5), indent=2))
        base = ref/('v31_event_confidence' if day == 26 else 'v31_decay_d32_day25_s42')
        reference = base/'ensemble_valid.npy'
        events = base/'valid_events.npy'
        dest = out/f'prediction{day}.npy'
        np.save(dest, reallocate(model, temperature, feature_cache(source/f'day{day}'), reference, events))
        execute('compare_v3.py', '--candidate', dest, '--events', events, '--baseline', reference,
                '--day', day, '--bootstrap', 150 if day == 26 else 0, '--out', out/f'comparison{day}.json')

    later = json.loads((out/'comparison26.json').read_text())
    earlier = json.loads((out/'comparison25.json').read_text())
    if not args.full or earlier['global_auc_delta'] <= 0 or later['bootstrap']['paired_delta_95_percentile_interval'][0] <= 0:
        return
    cache_path = out/'evaluation'
    build_cache(cache_path, 26, full=True)
    final = ref/'v31_final_combiner'
    dest = out/'evaluation_pred.npy'
    np.save(dest, reallocate(model, temperature, feature_cache(cache_path),
                              final/'ensemble_evaluation.npy', final/'events.npy'))
    execute('submission.py', '--predictions', dest, '--out', f'submission_techarena_{out.name}.csv')


if __name__ == '__main__':
    main()
