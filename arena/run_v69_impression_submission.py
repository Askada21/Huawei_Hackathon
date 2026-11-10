"""Screen factors learned from the user-item impression matrix."""
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.special import expit, logit
from sklearn.utils.extmath import randomized_svd

from common import RUNS, dimensions, load, record, split, cutoff


def main():
    data = load()
    nu, ni = dimensions()
    for day in (24, 25, 26, None):
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
        u, s, vt = randomized_svd(matrix, n_components=33, n_iter=4, random_state=68)
        left, right = u[:, 1:]*np.sqrt(s[1:]), vt[1:].T*np.sqrt(s[1:])
        for factor in (left, right):
            factor /= np.maximum(np.linalg.norm(factor, axis=1, keepdims=True), 1e-8)
        signal = np.einsum('ed,ecd->ec', left[target['user'][valid]], right[target['item'][valid]])
        name = 'evaluation' if day is None else 'final' if day == 26 else f'raw{day}'
        p = np.load(RUNS/'v37_join'/f'{name}.npy')
        if day:
            frame = RUNS/'v31_verified'/f'v31_decay_d32_day{day}_s42'/'valid_events.npy'
            if not np.array_equal(valid, np.load(frame)):
                raise ValueError('Validation alignment differs')
        pred = expit(logit(p.clip(1e-6, 1-1e-6)) + .25*signal)
        if day is None:
            with (RUNS/'v69_evaluation.npy').open('xb') as out:
                np.save(out, pred)
            continue
        baseline = record('v69_impressions_baseline', day, data['label'][valid], p)
        result = record('v69_impressions', day, data['label'][valid], pred)
        late = data['time'][valid] >= cutoff(day+1)
        if late.any():
            record('v69_late_baseline', day, data['label'][valid][late], p[late])
            record('v69_late_impressions', day, data['label'][valid][late], pred[late])
        if result['auc'] <= baseline['auc']:
            return


if __name__ == '__main__':
    main()
