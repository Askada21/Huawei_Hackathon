"""Screen item CTR by time of day on first events after temporal cutoffs."""
import numpy as np
from scipy.special import expit, logit

from common import RUNS, cutoff, dimensions, load, record, split


def main():
    data = load()
    _, size = dimensions()
    for day in (24, 25, 26):
        fit, events = split(data, day)
        frame = RUNS/'v31_verified'/f'v31_decay_d32_day{day}_s42'/'valid_events.npy'
        if not np.array_equal(events, np.load(frame)):
            raise ValueError('Validation alignment differs')
        def time_bin(rows):
            return (data['time'][rows] // 21600) % 4

        ids = data['item'][fit].ravel()
        keys = (time_bin(fit)[:, None] * size + data['item'][fit]).ravel()
        labels = data['label'][fit].ravel()
        count = np.bincount(keys, minlength=4*size).reshape(4, size)
        clicks = np.bincount(keys, weights=labels, minlength=4*size).reshape(4, size)
        prior = (clicks.sum(axis=0) + 5) / (np.bincount(ids, minlength=size) + 100)
        drift = logit((clicks + 100*prior)/(count + 100)) - logit(prior)
        base = np.load(RUNS/'v37_join'/f'raw{day}.npy')
        late = data['time'][events] >= cutoff(day+1)
        for weight in (0., .25, .5):
            p = expit(logit(base.clip(1e-6, 1-1e-6)) + weight*drift[time_bin(events)[:, None], data['item'][events]])
            record('v47_hour_item', day, data['label'][events], p, weight=weight, cohort='all')
            if late.any():
                record('v47_hour_item', day, data['label'][events[late]], p[late],
                       weight=weight, cohort='24h_plus', events=int(late.sum()))


if __name__ == '__main__':
    main()
