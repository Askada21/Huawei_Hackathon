"""Test a regularized bilinear history transition model."""
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import SGDClassifier

from common import DATA, RUNS, load, record, split


def main():
    data = load()
    history = data['history']
    latest = history[np.arange(len(history)), data['htime'].argmax(axis=1)]
    for day in (24, 25, 26):
        fit, valid = split(data, day)
        path = DATA/f'spectral_{data["time"][fit[-1]]}_r64_iter25_reg0.npz'
        with np.load(path) as factors:
            embedding = factors['i'][:, 1:17].astype('float32')
        embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-8)
        embedding[0] = 0

        def design(events):
            previous = embedding[latest[events]]
            candidate = embedding[data['item'][events]]
            cross = previous[:, None, :, None] * candidate[:, :, None, :]
            return cross.reshape(-1, 256)

        model = SGDClassifier(loss='log_loss', alpha=.00001, average=True, random_state=42)
        rng = np.random.default_rng(42)
        for _ in range(2):
            order = rng.permutation(fit)
            for start in range(0, len(order), 4096):
                events = order[start:start+4096]
                model.partial_fit(design(events), data['label'][events].ravel(), classes=[0, 1])

        frame = RUNS/'v31_verified'/f'v31_decay_d32_day{day}_s42'/'valid_events.npy'
        if not np.array_equal(valid, np.load(frame)):
            raise ValueError('Validation event alignment differs')
        transition = np.concatenate([
            model.decision_function(design(valid[start:start+4096]))
            for start in range(0, len(valid), 4096)
        ]).reshape(-1, 10) - model.intercept_[0]
        base = np.load(RUNS/'v37_join'/f'raw{day}.npy')
        for weight in (0., .1, .25):
            prediction = expit(logit(base.clip(1e-6, 1-1e-6)) + weight*transition)
            record('v42_dense_transition', day, data['label'][valid], prediction,
                   weight=weight, fit_events=len(fit), embedding=str(path))


if __name__ == '__main__':
    main()
