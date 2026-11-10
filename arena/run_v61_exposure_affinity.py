"""Estimate candidate affinity from co-exposure with historical clicks."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit, logit

from common import RUNS, dimensions, load, record, split


def main():
    data = load()
    _, ni = dimensions()
    for day in (24, 25, 26):
        fit, valid = split(data, day)
        events, positions = np.where(data['label'][fit] == 1)
        slate = data['item'][fit[events]]
        anchor = np.repeat(slate[np.arange(len(events)), positions], 10)
        others = slate.ravel()
        graph = csr_matrix((np.ones(len(anchor), dtype='float32'), (anchor, others)),
                           shape=(ni, ni))
        graph.setdiag(0)
        graph.eliminate_zeros()
        row_mass = np.asarray(graph.sum(axis=1)).ravel()
        col_mass = np.asarray(graph.sum(axis=0)).ravel()
        prior = (col_mass + 1) / (col_mass.sum() + ni)
        frame = RUNS/'v31_verified'/f'v31_decay_d32_day{day}_s42'/'valid_events.npy'
        if not np.array_equal(valid, np.load(frame)):
            raise ValueError('Validation frame differs')
        history = data['history'][valid]
        candidate = data['item'][valid]
        affinity = np.zeros(candidate.shape, dtype='float32')
        for j in range(history.shape[1]):
            h = history[:, j, None]
            counts = np.asarray(graph[np.broadcast_to(h, candidate.shape).ravel(),
                                      candidate.ravel()]).reshape(candidate.shape)
            expected = row_mass[h] * prior[candidate]
            affinity += np.where(h > 0, np.log((counts + 5)/(expected + 5)), 0)
        affinity /= np.maximum((history > 0).sum(axis=1, keepdims=True), 1)
        base = np.load(RUNS/'v37_join'/f'raw{day}.npy')
        for weight in (0., .25, .5):
            p = expit(logit(base.clip(1e-6, 1-1e-6)) + weight*affinity)
            record('v61_exposure_affinity', day, data['label'][valid], p, weight=weight)


if __name__ == '__main__':
    main()
