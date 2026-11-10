"""Screen factors learned from the user-item impression matrix."""
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.special import expit, logit
from scipy.sparse.linalg import LinearOperator, svds

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
        base = logit(p.clip(1e-6, 1-1e-6)) + .5*signal + .125*(agreement + affinity)
        late = target['time'][valid] >= cutoff(day+1)
        clicks = np.bincount(ids, weights=data['label'][fit].ravel(), minlength=ni)
        ctr = logit((clicks+5)/(item_mass+100))[target['item'][valid]]
        relative_ctr = ctr - ctr.mean(axis=1, keepdims=True)
        popularity = np.log1p(item_mass[target["item"][valid]])
        popularity -= popularity.mean(axis=1, keepdims=True)
        period = (data['time'][fit] // 21600) % 4
        keys = (ni*period[:, None]+data['item'][fit]).ravel()
        count = np.bincount(keys, minlength=4*ni).reshape(4, ni)
        hits = np.bincount(keys, weights=data['label'][fit].ravel(), minlength=4*ni).reshape(4, ni)
        prior = (clicks+5)/(item_mass+100)
        lift = logit((hits+100*prior)/(count+100))-logit(prior)
        period = (target['time'][valid] // 21600) % 4
        hourly = lift[period[:, None], target['item'][valid]]
        hourly -= hourly.mean(axis=1, keepdims=True)
        counts = item_mass[target['item'][valid]]
        reliability = counts/(counts+100)
        adjusted = (reliability-.5)*relative_ctr
        adjusted -= adjusted.mean(axis=1, keepdims=True)
        similarity = np.einsum('ecd,ekd->eck', candidates, candidates)
        diagonal = np.arange(10)
        similarity[:, diagonal, diagonal] = -np.inf
        nearest = np.sort(similarity, axis=2)[:, :, -3:].mean(axis=2)
        difference = nearest - agreement
        difference -= difference.mean(axis=1, keepdims=True)
        from sklearn.cluster import MiniBatchKMeans
        groups = MiniBatchKMeans(n_clusters=32, random_state=42, n_init=3).fit_predict(right)
        slate = groups[data['item'][fit]]
        graph = np.zeros((32, 32))
        for j in range(10):
            peers = np.delete(slate, j, axis=1)
            keys = 32*np.repeat(slate[:, j], 9) + peers.ravel()
            graph += np.bincount(keys, minlength=1024).reshape(32, 32)
        mass = graph.sum(axis=1)
        expected = np.outer(mass, mass)/mass.sum()
        table = np.log((graph+100)/(expected+100))
        items = groups[target['item'][valid]]
        local = (table[items[:, :, None], items[:, None, :]].sum(axis=2)
                 - table[items, items])/9
        local = local.mean(axis=1, keepdims=True)
        for weight in (-.25, 0., .25):
            pred = expit(base + .25*contrast + -.25*signal.mean(axis=1, keepdims=True) - .375*relative_ctr - .25*ctr.mean(axis=1, keepdims=True) - .08*popularity + (-.25*hourly + .25*adjusted + .25*difference + weight*local))
            record('v172_contrast_weight', day, data['label'][valid], pred, weight=weight)
            if late.any():
                record('v172_late', day, data['label'][valid][late], pred[late], weight=weight)


if __name__ == '__main__':
    main()
