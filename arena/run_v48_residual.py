"""Train a boosted residual on temporally out-of-sample V37 predictions."""
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
        train = lgb.Dataset(np.concatenate(xs), label=np.concatenate(ys),
                            init_score=np.concatenate(zs))
        model = lgb.train(dict(objective='binary', learning_rate=.04, num_leaves=12,
                               min_data_in_leaf=500, verbosity=-1, num_threads=4,
                               deterministic=True, force_col_wise=True), train, num_boost_round=80)
        events, x, raw = frames[day]
        residual = model.predict(x, raw_score=True)
        references = {'raw': raw}
        if day == 26:
            p = np.load(RUNS/'v37_join/final.npy')
            references['final'] = logit(p.clip(1e-6, 1-1e-6)).ravel()
        for name, z in references.items():
            for weight in (0., .25, .5):
                p = expit(z + weight*residual).reshape(-1, 10)
                record('v48_residual', day, data['label'][events], p,
                       reference=name, weight=weight, fit_rows=sum(map(len, ys)))


if __name__ == '__main__':
    main()
