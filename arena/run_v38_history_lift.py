"""Screen smoothed recent-history cluster effects on fixed V37 frames."""
import json
from time import monotonic

import numpy as np
from scipy.special import expit, logit
from sklearn.cluster import MiniBatchKMeans

from common import DATA, RUNS, dimensions, load, record, split


def context(data, events, clusters):
    history = data['history'][events]
    latest = history[np.arange(len(events)), data['htime'][events].argmax(axis=1)]
    return clusters[latest]


def main():
    train = load()
    _, items = dimensions()
    for day in (24, 25, 26):
        started = monotonic()
        fit, valid = split(train, day)
        embedding_file = DATA / f'spectral_{train["time"][fit[-1]]}_r64_iter25_reg0.npz'
        with np.load(embedding_file) as model:
            embedding = model['i'][1:, 1:32].copy()
        embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-9)
        clusters = np.r_[32, MiniBatchKMeans(n_clusters=32, batch_size=4096,
                                            n_init=3, random_state=42).fit_predict(embedding)]
        counts = np.zeros((33, items))
        clicks = np.zeros_like(counts)
        for start in range(0, len(fit), 20000):
            events = fit[start:start+20000]
            keys = context(train, events, clusters)[:, None] * items + train['item'][events]
            counts += np.bincount(keys.ravel(), minlength=counts.size).reshape(counts.shape)
            clicks += np.bincount(keys.ravel(), weights=train['label'][events].ravel(),
                                  minlength=counts.size).reshape(counts.shape)
        prior = (clicks.sum(axis=0) + 5) / (counts.sum(axis=0) + 100)
        rate = (clicks + 100 * prior) / (counts + 100)
        correction = logit(rate) - logit(prior)
        signal = correction[context(train, valid, clusters)[:, None], train['item'][valid]]
        raw = RUNS/'v37_join'/f'raw{day}.npy'
        frame = RUNS/'v31_verified'/f'v31_decay_d32_day{day}_s42'/'valid_events.npy'
        if not np.array_equal(np.load(frame), valid):
            raise ValueError('V37 validation frame differs')
        base = np.load(raw)
        labels = train['label'][valid]
        for weight in (0., .25, .5):
            pred = expit(logit(base.clip(1e-6, 1-1e-6)) + weight * signal)
            record('v38_history_lift', day, labels, pred, weight=weight,
                   fit_events=len(fit), valid_events=len(valid), runtime_seconds=monotonic()-started,
                   embedding=str(embedding_file), baseline=str(raw))


if __name__ == '__main__':
    main()
