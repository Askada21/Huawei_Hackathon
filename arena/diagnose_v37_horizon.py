"""Measure V37 discrimination at different delays after each temporal cutoff."""
import numpy as np
from sklearn.metrics import roc_auc_score

from common import RUNS, cutoff, load, split


def main():
    data = load()
    for day in (24, 25, 26):
        _, events = split(data, day)
        frame = RUNS/'v31_verified'/f'v31_decay_d32_day{day}_s42'/'valid_events.npy'
        if not np.array_equal(np.load(frame), events):
            raise ValueError('Validation event alignment differs')
        age = (data['time'][events] - cutoff(day)) / 3600
        scores = np.load(RUNS/'v37_join'/f'raw{day}.npy')
        for start, stop in ((0, 6), (6, 24), (24, 48), (48, np.inf)):
            mask = (age >= start) & (age < stop)
            labels = data['label'][events[mask]].ravel()
            if np.unique(labels).size == 2:
                print(day, start, stop, mask.sum(),
                      f'{roc_auc_score(labels, scores[mask].ravel()):.6f}')
    evaluation = load('evaluation')
    delay = (evaluation['time'] - data['time'].max()) / 3600
    print('Evaluation delay quantiles (hours):', np.quantile(delay, [0, .25, .5, .75, 1]))


if __name__ == '__main__':
    main()
