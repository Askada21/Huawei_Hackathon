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
    def fit_residual(day, before, offset):
        events, x, _ = frames[day]
        mask = data['time'][events] < cutoff(before)
        rows = np.repeat(mask, 10)
        train = lgb.Dataset(x[rows], label=data['label'][events[mask]].ravel(),
                            init_score=offset[rows])
        return lgb.train(dict(objective='binary', learning_rate=.04, num_leaves=12,
                              min_data_in_leaf=500, verbosity=-1, num_threads=4,
                              deterministic=True, force_col_wise=True), train, num_boost_round=80)

    first = fit_residual(24, 25, frames[24][2])
    _, x25, z25 = frames[25]
    second = fit_residual(25, 26, z25 + .5*first.predict(x25, raw_score=True))
    events, x, raw = frames[26]
    proxy = raw + .5*first.predict(x, raw_score=True)
    p = np.load(RUNS/'v37_join/final.npy')
    final = logit(p.clip(1e-6, 1-1e-6)).ravel()
    residual = second.predict(x, raw_score=True)
    from sklearn.metrics import roc_auc_score

    size = (data['history'][events] > 0).sum(axis=1)
    labels = data['label'][events]
    baseline = final.reshape(-1, 10)
    candidate = (final + .25*residual).reshape(-1, 10)
    for lo, hi in ((0, 4), (4, 10), (10, 21)):
        mask = (size >= lo) & (size < hi)
        y = labels[mask].ravel()
        before = roc_auc_score(y, baseline[mask].ravel())
        after = roc_auc_score(y, candidate[mask].ravel())
        print(lo, hi, int(mask.sum()), before, after, after-before)


if __name__ == '__main__':
    main()
