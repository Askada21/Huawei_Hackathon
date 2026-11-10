"""Screen directional item affinity from fitting-only impressions."""
import numpy as np
from scipy.special import expit, logit

from common import RUNS, dimensions, load, record, split


def main():
    data = load()
    _, ni = dimensions()
    history = data['history']
    latest = history[np.arange(len(history)), data['htime'].argmax(axis=1)]
    for day in (24, 25, 26):
        fit, valid = split(data, day)
        # Keep only observed pairs rather than allocating a dense item matrix.
        keys = (latest[fit, None].astype('int64') * ni + data['item'][fit]).ravel()
        labels = data['label'][fit].ravel()
        pairs, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        clicks = np.bincount(inverse, weights=labels)
        items = data['item'][fit].ravel()
        prior = (np.bincount(items, weights=labels, minlength=ni) + 5) / (
            np.bincount(items, minlength=ni) + 100)
        candidate = data['item'][valid]
        query = latest[valid, None].astype('int64') * ni + candidate
        index = np.minimum(np.searchsorted(pairs, query), len(pairs)-1)
        p = prior[candidate]
        lift = np.where((pairs[index] == query) & (latest[valid, None] > 0),
                        logit((clicks[index] + 20*p)/(counts[index]+20))-logit(p), 0.)
        frame = RUNS/'v31_verified'/f'v31_decay_d32_day{day}_s42'/'valid_events.npy'
        if not np.array_equal(valid, np.load(frame)):
            raise ValueError('Validation frame differs')
        base = np.load(RUNS/'v37_join'/f'raw{day}.npy')
        for weight in (0., .25, .5):
            scores = expit(logit(base.clip(1e-6, 1-1e-6)) + weight*lift)
            record('v41_item_neighbors', day, data['label'][valid], scores,
                   weight=weight, coverage=float(np.mean(lift != 0)))


if __name__ == '__main__':
    main()
