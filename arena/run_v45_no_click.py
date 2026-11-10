"""Screen event click probability learned on earlier V37 predictions."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from common import RUNS, cutoff, load, record, split


def main():
    data = load()
    frames = {}
    for day in (24, 25, 26):
        _, events = split(data, day)
        frame = RUNS/'v31_verified'/f'v31_decay_d32_day{day}_s42'/'valid_events.npy'
        if not np.array_equal(events, np.load(frame)):
            raise ValueError('Validation event alignment differs')
        p = np.load(RUNS/'v37_join'/f'raw{day}.npy')
        ordered = np.sort(p, axis=1)
        size = (data['history'][events] > 0).sum(axis=1)
        age = data['time'][events] - data['htime'][events].max(axis=1)
        x = np.column_stack([ordered, p.sum(axis=1), size, age])
        frames[day] = events, p, x
    for day in (25, 26):
        xs, ys = [], []
        for earlier in range(24, day):
            events, _, x = frames[earlier]
            mask = data['time'][events] < cutoff(day)
            xs.append(x[mask])
            ys.append(data['label'][events[mask]].any(axis=1))
        model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                               min_samples_leaf=150, random_state=42)
        model.fit(np.concatenate(xs), np.concatenate(ys))
        events, raw, x = frames[day]
        mass = model.predict_proba(x)[:, 1, None]
        references = {'raw': raw}
        if day == 26:
            references['final'] = np.load(RUNS/'v37_join/final.npy')
        for name, p in references.items():
            updated = mass * p / np.maximum(p.sum(axis=1, keepdims=True), 1e-8)
            for weight in (0., .25, .5):
                prediction = (1-weight)*p + weight*updated
                record('v45_no_click', day, data['label'][events], prediction,
                       reference=name, weight=weight, fit_events=sum(map(len, ys)))


if __name__ == '__main__':
    main()
