"""Estimate candidate affinity from co-exposure with historical clicks."""
import numpy as np
from scipy.sparse import csr_matrix, diags
from sklearn.utils.extmath import randomized_svd
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
        normalized = diags(1/np.sqrt(row_mass+20)) @ graph @ diags(1/np.sqrt(col_mass+20))
        left, singular, right = randomized_svd(normalized, n_components=33, n_iter=4, random_state=42)
        source = left[:, 1:] * np.sqrt(singular[1:])
        target = right[1:].T * np.sqrt(singular[1:])
        for embedding in (source, target):
            embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-8)
        source[0] = 0
        frame = RUNS/'v31_verified'/f'v31_decay_d32_day{day}_s42'/'valid_events.npy'
        if not np.array_equal(valid, np.load(frame)):
            raise ValueError('Validation frame differs')
        history = data['history'][valid]
        candidate = data['item'][valid]
        profile = source[history].sum(axis=1)
        profile /= np.maximum(np.linalg.norm(profile, axis=1, keepdims=True), 1e-8)
        affinity = np.empty(candidate.shape, dtype='float32')
        for j in range(candidate.shape[1]):
            affinity[:, j] = np.einsum('ij,ij->i', profile, target[candidate[:, j]])
        base = np.load(RUNS/'v37_join'/f'raw{day}.npy')
        for weight in (0., .25, .5):
            p = expit(logit(base.clip(1e-6, 1-1e-6)) + weight*affinity)
            record('v62_spectral_exposure', day, data['label'][valid], p, weight=weight)


if __name__ == '__main__':
    main()
