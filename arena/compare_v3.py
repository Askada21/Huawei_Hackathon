"""Read-only comparison on identical first-event temporal validation rows.

No fitting is performed. Frequency/cold-user diagnostics use fitting rows only.
Bootstrap resamples whole events, which are distinct users in this validation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

from common import ROOT, load, split, cutoff

DEFAULT_BASELINE = ROOT / 'arena_runs/joint_factor_d32_day26_s42/valid_calibrated_clean.npy'


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def assert_valid_artifact(path):
    """Explicitly reject known invalid stacking artifacts."""
    path = Path(path)
    if 'calibrated_converged' in path.parts or path.name in {
        'valid_calibrated.npy', 'evaluation_calibrated.npy'
    }:
        raise ValueError(f'Known invalid stacking artifact: {path}')


def checked_predictions(path, events):
    path = resolve(path)
    assert_valid_artifact(path)
    predictions = np.load(path, allow_pickle=False)
    if predictions.size != len(events) * 10:
        raise ValueError(f'{path} has {predictions.size} values for {len(events)} events')
    predictions = predictions.reshape(len(events), 10)
    if not (np.isfinite(predictions).all() and
            (predictions >= 0).all() and (predictions <= 1).all()):
        raise ValueError(f'{path} contains invalid probabilities')
    event_path = path.parent / 'valid_events.npy'
    if event_path.exists() and not np.array_equal(np.load(event_path), events):
        raise ValueError(f'Prediction event order differs: {event_path}')
    return predictions


def auc_or_none(y, p):
    y = np.asarray(y).ravel()
    return float(roc_auc_score(y, np.asarray(p).ravel())) if len(np.unique(y)) == 2 else None


def metrics(y, p):
    clicked = y.sum(axis=1) > 0
    result = {
        'events': len(y), 'rows': int(y.size), 'clicked_rows': int(y.sum()),
        'global_row_auc': auc_or_none(y, p),
        'binary_row_logloss': float(log_loss(y.ravel(), p.ravel(), labels=[0, 1])),
        'event_click_auc_using_probability_sum': auc_or_none(clicked, p.sum(axis=1)),
        'clicked_event_row_auc': auc_or_none(y[clicked], p[clicked]),
        'clicked_event_top1_accuracy': float(np.mean(p[clicked].argmax(axis=1) == y[clicked].argmax(axis=1))) if clicked.any() else None,
        'event_prediction_sum_min': float(p.sum(axis=1).min()),
        'event_prediction_sum_max': float(p.sum(axis=1).max()),
    }
    return result


def paired_row_summary(y, candidate, baseline, mask):
    y = y[mask]
    c = candidate[mask]
    b = baseline[mask]
    ca, ba = auc_or_none(y, c), auc_or_none(y, b)
    return {'rows': int(y.size), 'clicked_rows': int(y.sum()),
            'candidate_auc': ca, 'baseline_auc': ba,
            'auc_delta': None if ca is None or ba is None else ca - ba}


def compare(candidate_path, events_path, baseline_path, day, bootstrap=0, seed=42):
    a = load()
    fitting, expected = split(a, day)
    events = np.load(resolve(events_path), allow_pickle=False)
    if events.dtype.kind not in 'iu' or events.ndim != 1:
        raise ValueError('Events must be a one-dimensional integer index array')
    if not np.array_equal(events, expected):
        raise ValueError('Events must exactly match first events per user after this cutoff')
    assert np.unique(a['user'][events]).size == len(events)
    assert np.all(a['time'][fitting] < cutoff(day))
    y = np.asarray(a['label'][events])
    candidate = checked_predictions(candidate_path, events)
    baseline = checked_predictions(baseline_path, events)
    cmetrics, bmetrics = metrics(y, candidate), metrics(y, baseline)
    n = (a['history'][events] > 0).sum(axis=1)
    known_users = np.unique(a['user'][fitting])
    cold = ~np.isin(a['user'][events], known_users)
    row_frequency = np.bincount(a['item'][fitting].ravel(), minlength=int(a['item'].max()) + 1)[a['item'][events]]
    groups = {}
    for name, event_mask in {
        'history_0_5': n <= 5, 'history_6_15': (n > 5) & (n <= 15),
        'history_16_20': n > 15, 'unseen_fitting_user': cold, 'seen_fitting_user': ~cold
    }.items():
        group = paired_row_summary(y, candidate, baseline, np.repeat(event_mask[:, None], 10, axis=1))
        group['events'] = int(event_mask.sum())
        groups[name] = group
    for name, row_mask in {
        'item_fit_impressions_0_20': row_frequency <= 20,
        'item_fit_impressions_21_100': (row_frequency > 20) & (row_frequency <= 100),
        'item_fit_impressions_above_100': row_frequency > 100
    }.items():
        groups[name] = paired_row_summary(y, candidate, baseline, row_mask)
    result = {
        'cutoff_day': day, 'candidate_path': str(resolve(candidate_path)),
        'baseline_path': str(resolve(baseline_path)), 'events_path': str(resolve(events_path)),
        'fitting_events': len(fitting), 'candidate': cmetrics, 'baseline': bmetrics,
        'global_auc_delta': cmetrics['global_row_auc'] - bmetrics['global_row_auc'],
        'subgroups': groups,
        'notes': [
            'Metrics are local validation diagnostics, not official leaderboard scores.',
            'All comparisons use identical ordered events, one event per validation user.',
            'Subgroup AUCs are not causal comparisons and need not average to global AUC.',
            'Probability-sum event AUC is a ranking diagnostic; sums may exceed one after calibration.',
        ],
    }
    for key, path in [('candidate_sha256', candidate_path), ('baseline_sha256', baseline_path), ('events_sha256', events_path)]:
        with open(resolve(path), 'rb') as stream:
            result[key] = hashlib.file_digest(stream, 'sha256').hexdigest()
    if bootstrap:
        rng = np.random.default_rng(seed)
        c_aucs, b_aucs = [], []
        for _ in range(bootstrap):
            sample = rng.integers(len(events), size=len(events))
            c_aucs.append(roc_auc_score(y[sample].ravel(), candidate[sample].ravel()))
            b_aucs.append(roc_auc_score(y[sample].ravel(), baseline[sample].ravel()))
        result['bootstrap'] = {
            'replicates': bootstrap, 'seed': seed, 'unit': 'whole event = distinct validation user',
            'candidate_auc_95_percentile_interval': np.quantile(c_aucs, [.025, .975]).tolist(),
            'baseline_auc_95_percentile_interval': np.quantile(b_aucs, [.025, .975]).tolist(),
            'paired_delta_95_percentile_interval': np.quantile(np.asarray(c_aucs) - np.asarray(b_aucs), [.025, .975]).tolist(),
            'limitation': 'Sampling uncertainty only, not model-selection bias or temporal shift.'
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--events', required=True)
    parser.add_argument('--baseline', default=str(DEFAULT_BASELINE))
    parser.add_argument('--day', type=int, default=26)
    parser.add_argument('--out', required=True)
    parser.add_argument('--bootstrap', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.day != 26 and args.baseline == str(DEFAULT_BASELINE):
        parser.error('An explicit same-cutoff baseline is required for days other than 26')
    if args.bootstrap < 0:
        parser.error('--bootstrap must be non-negative; use 150 for a final comparison')
    out = resolve(args.out)
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    result = compare(args.candidate, args.events, args.baseline, args.day, args.bootstrap, args.seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
