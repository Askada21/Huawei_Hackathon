"""Screen factors learned from the user-item impression matrix."""
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.special import expit, logit
from sklearn.utils.extmath import randomized_svd

from common import RUNS, dimensions, load, record, split


def main():
    data = load()
    nu, ni = dimensions()
    for day in (24, 25, 26):
        fit, valid = split(data, day)
        ids = data['item'][fit].ravel()
        matrix = csr_matrix((np.ones(len(ids), dtype='float32'),
                             (np.repeat(data['user'][fit], 10), ids)), shape=(nu, ni))
        user_mass = np.asarray(matrix.sum(axis=1)).ravel()
        item_mass = np.asarray(matrix.sum(axis=0)).ravel()
        matrix = diags(1/np.sqrt(user_mass+20)) @ matrix @ diags(1/np.sqrt(item_mass+20))
        u, s, vt = randomized_svd(matrix, n_components=33, n_iter=4, random_state=68)
        left, right = u[:, 1:]*np.sqrt(s[1:]), vt[1:].T*np.sqrt(s[1:])
        for factor in (left, right):
            factor /= np.maximum(np.linalg.norm(factor, axis=1, keepdims=True), 1e-8)
        frame = RUNS/'v31_verified'/f'v31_decay_d32_day{day}_s42'/'valid_events.npy'
        if not np.array_equal(valid, np.load(frame)):
            raise ValueError('Validation alignment differs')
        signal = np.einsum('ed,ecd->ec', left[data['user'][valid]], right[data['item'][valid]])
        base = logit(np.load(RUNS/'v37_join'/f'raw{day}.npy').clip(1e-6, 1-1e-6))
        for weight in (-.25, 0., .25):
            record('v68_impression_factors', day, data['label'][valid],
                   expit(base + weight*signal), weight=weight)


if __name__ == '__main__':
    main()
