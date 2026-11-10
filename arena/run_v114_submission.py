"""Screen factors learned from the user-item impression matrix."""
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.special import expit, logit
from scipy.sparse.linalg import LinearOperator, svds

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
        candidates = right[target['item'][valid]]
        peers = candidates.sum(axis=1, keepdims=True) - candidates
        agreement = np.einsum('ecd,ecd->ec', candidates, peers) / 9
        history_ids = target['history'][valid]
        history = np.einsum('eh,ehd->ed', history_ids > 0, right[history_ids])
        history /= np.maximum(np.linalg.norm(history, axis=1, keepdims=True), 1e-8)
        affinity = np.einsum('ed,ecd->ec', history, candidates)
        clicked = fit[data['label'][fit].any(axis=1)]
        shown = data['item'][clicked]
        profile = np.zeros_like(left)
        for start in range(0, len(clicked), 4096):
            batch = slice(start, start+4096)
            np.add.at(profile, data['user'][clicked[batch]], right[shown[batch]].mean(axis=1))
        profile /= np.maximum(np.linalg.norm(profile, axis=1, keepdims=True), 1e-8)
        contrast = np.einsum('ed,ecd->ec', profile[target['user'][valid]], candidates) - signal
        pred = expit(logit(p.clip(1e-6, 1-1e-6)) + .5*signal + .125*(agreement + affinity) + .25*contrast - .25*signal.mean(axis=1, keepdims=True))
        if day is None:
            with (RUNS/'v114_evaluation.npy').open('xb') as out:
                np.save(out, pred)
            continue
        baseline = record('v114_impressions_baseline', day, data['label'][valid], p)
        result = record('v114_impressions', day, data['label'][valid], pred)
        late = data['time'][valid] >= cutoff(day+1)
        if late.any():
            record('v114_late_baseline', day, data['label'][valid][late], p[late])
            record('v114_late_impressions', day, data['label'][valid][late], pred[late])
        if result['auc'] <= baseline['auc']:
            return


if __name__ == '__main__':
    main()
