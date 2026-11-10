"""Test a regularized bilinear history transition model."""
import numpy as np
from scipy.special import expit, logit
from scipy.special import softmax

from common import DATA, RUNS, load, record, split


def main():
    data = load()
    for day in (24, 25, 26):
        fit, valid = split(data, day)
        path = DATA/f'spectral_{data["time"][fit[-1]]}_r64_iter25_reg0.npz'
        with np.load(path) as factors:
            embedding = factors['i'][:, 1:17].astype('float32')
        embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-8)
        embedding[0] = 0

        def design(events):
            history = data['history'][events]
            age = data['time'][events, None] - data['htime'][events]
            weights = np.exp2(-np.maximum(age, 0) / 86400).astype('float32')
            weights *= history > 0
            weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
            previous = np.einsum('eh,ehd->ed', weights, embedding[history])
            candidate = embedding[data['item'][events]]
            cross = previous[:, None, :, None] * candidate[:, :, None, :]
            return cross.reshape(-1, 256)

        coefficients = np.zeros(256, dtype='float32')
        bias = np.float32(0)
        rng = np.random.default_rng(42)
        for epoch in range(5):
            order = rng.permutation(fit)
            for start in range(0, len(order), 2048):
                events = order[start:start+2048]
                x = design(events)
                scores = (x @ coefficients).reshape(-1, 10) + bias
                probs = softmax(np.pad(scores, ((0, 0), (0, 1))), axis=1)[:, :10]
                error = probs - data['label'][events]
                step = 3 / np.sqrt(epoch+1)
                coefficients -= step * (x.T @ error.ravel()/len(events) + .0001*coefficients)
                bias -= .1 * error.sum(axis=1).mean()

        frame = RUNS/'v31_verified'/f'v31_decay_d32_day{day}_s42'/'valid_events.npy'
        if not np.array_equal(valid, np.load(frame)):
            raise ValueError('Validation event alignment differs')
        transition = np.concatenate([
            design(valid[start:start+4096]) @ coefficients
            for start in range(0, len(valid), 4096)
        ]).reshape(-1, 10)
        base = np.load(RUNS/'v37_join'/f'raw{day}.npy')
        for weight in (0., .1, .25):
            prediction = expit(logit(base.clip(1e-6, 1-1e-6)) + weight*transition)
            record('v65_slate_choice', day, data['label'][valid], prediction,
                   weight=weight, fit_events=len(fit), embedding=str(path))


if __name__ == '__main__':
    main()
