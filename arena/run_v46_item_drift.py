"""Screen recent item CTR drift on first events after temporal cutoffs."""
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
        recent = data['time'][fit] >= cutoff(day-2)

        def counts(rows):
            ids = data['item'][rows].ravel()
            labels = data['label'][rows].ravel()
            return np.bincount(ids, minlength=size), np.bincount(ids, weights=labels, minlength=size)

        older_n, older_y = counts(fit[~recent])
        newer_n, newer_y = counts(fit[recent])
        prior = (older_y + 5)/(older_n + 100)
        drift = logit((newer_y + 100*prior)/(newer_n + 100)) - logit(prior)
        base = np.load(RUNS/'v37_join'/f'raw{day}.npy')
        late = data['time'][events] >= cutoff(day+1)
        for weight in (0., .25, .5):
            p = expit(logit(base.clip(1e-6, 1-1e-6)) + weight*drift[data['item'][events]])
            record('v46_item_drift', day, data['label'][events], p, weight=weight, cohort='all')
            if late.any():
                record('v46_item_drift', day, data['label'][events[late]], p[late],
                       weight=weight, cohort='24h_plus', events=int(late.sum()))


if __name__ == '__main__':
    main()
