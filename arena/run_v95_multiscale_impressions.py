"""Screen factors learned from the user-item impression matrix."""
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, svds
from scipy.special import expit, logit
from sklearn.utils.extmath import randomized_svd

from common import RUNS, dimensions, load, record, split, cutoff


def main():
    data = load()
    nu, ni = dimensions()
    for day in (24, 25, 26):
        target = data if day else load('evaluation')
        if day:
            fit, valid = split(data, day)
        else:
            fit = np.arange(len(data['time']))
            valid = np.arange(len(target['time']))
        ids = data['item'][fit].ravel()
        matrix = csr_matrix((np.ones(len(ids), dtype='float32'),
                             (np.repeat(data['user'][fit], 10), ids)), shape=(nu, ni))
        user_mass = np.asarray(matrix.sum(axis=1)).ravel()
        item_mass = np.asarray(matrix.sum(axis=0)).ravel()
        matrix = diags(1/np.sqrt(user_mass+20)) @ matrix @ diags(1/np.sqrt(item_mass+20))
        user_null = user_mass / np.sqrt(user_mass + 20)
        item_null = item_mass / np.sqrt(item_mass + 20) / item_mass.sum()
        residual = LinearOperator(matrix.shape, dtype=matrix.dtype,
                                  matvec=lambda x: (matrix @ x).ravel() - user_null * (item_null @ x.ravel()),
                                  rmatvec=lambda x: (matrix.T @ x).ravel() - item_null * (user_null @ x.ravel()))
        u, singular, vt = svds(residual, k=32, random_state=68)
        order = np.argsort(singular)[::-1]
        user_ids = target['user'][valid]
        item_ids = target['item'][valid]
        signals = []
        for rank in (8, 16, 32):
            ix = order[:rank]
            left = u[:, ix] * np.sqrt(singular[ix])
            right = vt[ix].T * np.sqrt(singular[ix])
            for factor in (left, right):
                factor /= np.maximum(np.linalg.norm(factor, axis=1, keepdims=True), 1e-8)
            signals.append(np.einsum('ed,ecd->ec', left[user_ids], right[item_ids]))
        signal = .2*signals[0] + .6*signals[1] + .2*signals[2]
        name = 'evaluation' if day is None else 'final' if day == 26 else f'raw{day}'
        p = np.load(RUNS/'v37_join'/f'{name}.npy')
        if day:
            frame = RUNS/'v31_verified'/f'v31_decay_d32_day{day}_s42'/'valid_events.npy'
            if not np.array_equal(valid, np.load(frame)):
                raise ValueError('Validation alignment differs')
        late = data['time'][valid] >= cutoff(day+1)
        for weight in (.25, .5, 1., 2.):
            pred = expit(logit(p.clip(1e-6, 1-1e-6)) + weight*signal)
            record('v95_weight', day, data['label'][valid], pred, weight=weight)
            if late.any():
                record('v95_late_weight', day, data['label'][valid][late], pred[late], weight=weight)


if __name__ == '__main__':
    main()
