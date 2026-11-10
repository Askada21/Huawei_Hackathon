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
            x = cache['x'][:, cache['names'] != 'day']
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

        def pair_loss(prediction, dataset):
            score = prediction + z
            error = expit(score[right] - score[left])
            curvature = error * (1-error)
            gradient = (np.bincount(right, error, minlength=len(y))
                        - np.bincount(left, error, minlength=len(y)))
            hessian = (np.bincount(left, curvature, minlength=len(y))
                       + np.bincount(right, curvature, minlength=len(y)))
            return gradient, hessian

        train = lgb.Dataset(np.concatenate(xs), label=y)
        model = lgb.train(dict(objective=pair_loss, learning_rate=.04, num_leaves=12,
                               min_data_in_leaf=500, verbosity=-1, num_threads=4,
                               deterministic=True, force_col_wise=True), train, num_boost_round=80)
        binary = lgb.train(dict(objective='binary', learning_rate=.04, num_leaves=12,
                                min_data_in_leaf=500, verbosity=-1, num_threads=4,
                                deterministic=True, force_col_wise=True),
                           lgb.Dataset(np.concatenate(xs), label=y, init_score=z),
                           num_boost_round=80)
        events, x, raw = frames[day]
        residual = .5 * (model.predict(x, raw_score=True) + binary.predict(x, raw_score=True))
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
                    record('v54_blended_residual_late', day, data['label'][events[late]], p[late],
                           reference=name, weight=weight, events=int(late.sum()))
                record('v54_blended_residual', day, data['label'][events], p,
                       reference=name, weight=weight, fit_rows=sum(map(len, ys)))


if __name__ == '__main__':
    main()
