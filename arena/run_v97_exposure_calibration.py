"""Screen factors learned from the user-item impression matrix."""
import numpy as np
import lightgbm as lgb
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, svds
from scipy.special import expit, logit
from sklearn.utils.extmath import randomized_svd

from common import RUNS, dimensions, load, record, split, cutoff


def main():
    data = load()
    nu, ni = dimensions()
    training = []
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
        u, s, vt = svds(residual, k=16, random_state=68)
        left, right = u*np.sqrt(s), vt.T*np.sqrt(s)
        for factor in (left, right):
            factor /= np.maximum(np.linalg.norm(factor, axis=1, keepdims=True), 1e-8)
        signal = np.einsum('ed,ecd->ec', left[target['user'][valid]], right[target['item'][valid]])
        name = 'evaluation' if day is None else 'final' if day == 26 else f'raw{day}'
        p = np.load(RUNS/'v37_join'/f'{name}.npy')
        if day:
            frame = RUNS/'v31_verified'/f'v31_decay_d32_day{day}_s42'/'valid_events.npy'
            if not np.array_equal(valid, np.load(frame)):
                raise ValueError('Validation alignment differs')
        base = logit(p.clip(1e-6, 1-1e-6)) + .5*signal
        x = np.column_stack([base.ravel(), signal.ravel(),
                             (signal-signal.mean(axis=1, keepdims=True)).ravel(),
                             np.repeat((data['history'][valid]>0).sum(axis=1), 10),
                             np.repeat(np.log1p(user_mass[data['user'][valid]]), 10)])
        y = data['label'][valid].ravel()
        if training:
            earlier = [(features[mask], labels[mask], offset[mask])
                       for times, features, labels, offset in training
                       for mask in [times < cutoff(day)]]
            features, labels, offset = map(np.concatenate, zip(*earlier))
            model = lgb.train(dict(objective='binary', num_leaves=7, learning_rate=.04,
                                   min_data_in_leaf=500, verbosity=-1, num_threads=4),
                              lgb.Dataset(features, label=labels, init_score=offset),
                              num_boost_round=60)
            correction = model.predict(x, raw_score=True)
            for weight in (0., .25, .5):
                record('v97_exposure_calibration', day, y,
                       expit(base.ravel()+weight*correction), weight=weight)
        training.append((np.repeat(data['time'][valid], 10), x, y, base.ravel()))


if __name__ == '__main__':
    main()
