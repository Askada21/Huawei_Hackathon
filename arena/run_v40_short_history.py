"""Test a short-history specialist with earlier user-item impressions."""
import numpy as np
from lightgbm import LGBMClassifier
from scipy.sparse import csr_matrix
from scipy.special import expit, logit

from common import RUNS, cutoff, dimensions, load, priors, record, split
from tree_features import features


def main():
    data = load()
    nu, ni = dimensions()
    history_size = (data['history'] > 0).sum(axis=1)
    for day in (24, 25, 26):
        _, valid = split(data, day)
        past = np.flatnonzero(data['time'] < cutoff(day-2))
        fit = np.flatnonzero((data['time'] >= cutoff(day-2)) &
                             (data['time'] < cutoff(day)) & (history_size < 10))
        users = np.repeat(data['user'][past], 10)
        items = data['item'][past].ravel()
        labels = data['label'][past].ravel()
        clicked = labels > 0
        shows = csr_matrix((np.ones(len(items), dtype=np.int32), (users, items)), shape=(nu, ni))
        clicks = csr_matrix((labels[clicked].astype('int32'), (users[clicked], items[clicked])), shape=(nu, ni))
        stats = priors(data, past, nu, ni)

        def design(events):
            base, _ = features(data, events, stats)
            users = np.repeat(data['user'][events], 10)
            items = data['item'][events].ravel()
            return np.column_stack([base, np.asarray(shows[users, items]).ravel(),
                                    np.asarray(clicks[users, items]).ravel()]).astype('float32')

        model = LGBMClassifier(n_estimators=180, num_leaves=15, learning_rate=.05,
                               min_child_samples=100, n_jobs=4, verbosity=-1)
        model.fit(design(fit), data['label'][fit].ravel())
        frame = RUNS/'v31_verified'/f'v31_decay_d32_day{day}_s42'/'valid_events.npy'
        if not np.array_equal(np.load(frame), valid):
            raise ValueError('Validation event alignment differs')
        mask = history_size[valid] < 10
        score = model.predict_proba(design(valid[mask]))[:, 1].reshape(-1, 10)
        directory = RUNS/'v37_join'
        base = np.load(directory/f'raw{day}.npy')
        for weight in (0., .1, .25):
            pred = base.copy()
            pred[mask] = expit((1-weight)*logit(base[mask].clip(1e-6, 1-1e-6))
                               + weight*logit(score.clip(1e-6, 1-1e-6)))
            record('v40_short_history', day, data['label'][valid], pred,
                   weight=weight, specialist_fit_events=len(fit),
                   specialist_validation_events=int(mask.sum()))


if __name__ == '__main__':
    main()
