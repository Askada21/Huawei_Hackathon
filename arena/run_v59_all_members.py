"""Train a global pairwise residual on temporally out-of-sample V37 predictions."""
import numpy as np
import lightgbm as lgb
from scipy.special import expit, logit

from common import RUNS, cutoff, load, record, split


def main():
    data = load()
    frames = {}
    for day in (24, 25, 26):
        _, events = split(data, day)
        with np.load(RUNS/'v34_stack'/f'day{day}'/'calibration_features.npz') as cache:
            if not np.array_equal(cache['events'], events):
                raise ValueError('Validation alignment differs')
            names = cache['names']
            values = cache['x']
            members = values[:, np.char.endswith(names, '_logit')]
            if members.shape[1] != 5:
                raise ValueError('Expected five member logits')
            extra = []
            for run, tag in [('v30_checks', 'v30_signed025'), ('v32_checks', 'v32_adaptive')]:
                folder = RUNS/run/f'{tag}_d32_day{day}_s42'
                if not np.array_equal(events, np.load(folder/'valid_events.npy')):
                    raise ValueError('Member validation alignment differs')
                p = np.load(folder/'valid_pred.npy')
                extra.append(logit(p.clip(1e-6, 1-1e-6)).ravel())
            members = np.column_stack([members, *extra])
            spread = members.std(axis=1).reshape(-1, 10)
            x = np.column_stack([values[:, names != 'day'], *extra, spread.ravel(),
                                 (spread-spread.mean(axis=1, keepdims=True)).ravel()])
        p = np.load(RUNS/'v37_join'/f'raw{day}.npy')
        z = logit(p.clip(1e-6, 1-1e-6)).ravel()
        frames[day] = events, x, z
    for day in (25, 26):
        xs, zs, ys = [], [], []
        for earlier in range(24, day):
            events, x, z = frames[earlier]
            mask = data['time'][events] < cutoff(day)
            rows = np.repeat(mask, 10)
            xs.append(x[rows])
            zs.append(z[rows])
            ys.append(data['label'][events[mask]].ravel())
        y = np.concatenate(ys)
        z = np.concatenate(zs)
        positives = np.flatnonzero(y)
        negatives = np.flatnonzero(y == 0)
        rng = np.random.default_rng(51)
        left = np.repeat(positives, 8)
        right = rng.choice(negatives, len(left))

        # Add all nine within-event comparisons for each clicked row.
        local_left = np.repeat(positives, 10)
        local_right = (local_left // 10)*10 + np.tile(np.arange(10), len(positives))
        other = local_left != local_right
        left = np.concatenate([left, local_left[other]])
        right = np.concatenate([right, local_right[other]])

        # Emphasize pairs close in the original ranking while retaining broad coverage.
        importance = .2 + np.exp(-np.abs(z[left] - z[right]))

        def pair_loss(prediction, dataset):
            score = prediction + z
            error = expit(score[right] - score[left])
            curvature = importance * error * (1-error)
            error *= importance
            gradient = (np.bincount(right, error, minlength=len(y))
                        - np.bincount(left, error, minlength=len(y)))
            hessian = (np.bincount(left, curvature, minlength=len(y))
                       + np.bincount(right, curvature, minlength=len(y)))
            return gradient, hessian

        train = lgb.Dataset(np.concatenate(xs), label=y)
        model = lgb.train(dict(objective=pair_loss, learning_rate=.04, num_leaves=12,
                               min_data_in_leaf=500, verbosity=-1, num_threads=4,
                               deterministic=True, force_col_wise=True), train, num_boost_round=80)
        events, x, raw = frames[day]
        residual = model.predict(x, raw_score=True)
        references = {'raw': raw}
        if day == 25:
            calibrated = RUNS/'v34_stack/cal25'
            if not np.array_equal(events, np.load(calibrated/'valid_events.npy')):
                raise ValueError('Calibrated event alignment differs')
            p = np.load(calibrated/'blend.npy')
            references['v34_blend'] = logit(p.clip(1e-6, 1-1e-6)).ravel()
        if day == 26:
            p = np.load(RUNS/'v37_join/final.npy')
            references['final'] = logit(p.clip(1e-6, 1-1e-6)).ravel()
        for name, z in references.items():
            for weight in (0., .25, .5):
                p = expit(z + weight*residual).reshape(-1, 10)
                late = data['time'][events] >= cutoff(day+1)
                if late.any():
                    record('v59_all_members_late', day, data['label'][events[late]], p[late],
                           reference=name, weight=weight, events=int(late.sum()))
                record('v59_all_members', day, data['label'][events], p,
                       reference=name, weight=weight, fit_rows=sum(map(len, ys)))


if __name__ == '__main__':
    main()
