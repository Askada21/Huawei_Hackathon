"""Fine-tune spectral user and item factors under an eleven-choice likelihood."""
import numpy as np
from scipy.special import expit, logit, softmax

from common import DATA, RUNS, load, record, split


def main():
    data = load()
    for day in (24, 25, 26):
        fit, valid = split(data, day)
        with np.load(DATA/f'spectral_{data["time"][fit[-1]]}_r64_iter25_reg0.npz') as model:
            user, item = [model[k][:, 1:33].astype('float32') for k in ('u', 'i')]
        for factors in (user, item):
            factors /= np.maximum(np.linalg.norm(factors, axis=1, keepdims=True), 1e-8)
        item[0] = 0

        def profile(rows):
            history = data['history'][rows]
            age = np.maximum(data['time'][rows, None] - data['htime'][rows], 0)
            weight = np.exp2(-age / 172800).astype('float32') * (history > 0)
            context = np.einsum('eh,ehd->ed', weight, item[history])
            context /= np.maximum(np.linalg.norm(context, axis=1, keepdims=True), 1e-8)
            return context

        bias = np.zeros(len(item), dtype='float32')
        rng = np.random.default_rng(42)
        for epoch in range(3):
            order = rng.permutation(fit)
            rate = .03 / np.sqrt(epoch+1)
            for start in range(0, len(order), 2048):
                rows = order[start:start+2048]
                u, i = data['user'][rows], data['item'][rows]
                a, b = user[u] + profile(rows), item[i]
                logits = np.einsum('ed,ecd->ec', a, b) + bias[i]
                error = data['label'][rows] - softmax(
                    np.pad(logits, ((0, 0), (0, 1)), constant_values=2), axis=1)[:, :10]
                np.add.at(user, u, rate*(np.einsum('ec,ecd->ed', error, b)-.005*user[u]))
                np.add.at(item, i, rate*(error[..., None]*a[:, None]-.005*b))
                np.add.at(bias, i, rate*(error-.005*bias[i]))
        frame = RUNS/'v31_verified'/f'v31_decay_d32_day{day}_s42'/'valid_events.npy'
        if not np.array_equal(valid, np.load(frame)):
            raise ValueError('Validation alignment differs')
        ids = data['item'][valid]
        logits = np.einsum('ed,ecd->ec', user[data['user'][valid]] + profile(valid), item[ids]) + bias[ids]
        p = softmax(np.pad(logits, ((0, 0), (0, 1)), constant_values=2), axis=1)[:, :10]
        base = np.load(RUNS/'v37_join'/f'raw{day}.npy')
        for weight in (0., .1, .25):
            pred = expit((1-weight)*logit(base.clip(1e-6, 1-1e-6))
                         + weight*logit(p.clip(1e-6, 1-1e-6)))
            record('v67_slate_history', day, data['label'][valid], pred, weight=weight)


if __name__ == '__main__':
    main()
