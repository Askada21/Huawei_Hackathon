"""Measure history-dependent logit corrections learned on earlier events."""
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from common import RUNS, calibration_training_mask, load, record, split


def features(probabilities, history_size):
    logits = logit(probabilities.clip(1e-6, 1-1e-6))
    centered = logits - logits.mean(axis=1, keepdims=True)
    columns = [logits, centered]
    groups = np.digitize(history_size, [1, 4, 10])
    for group in range(4):
        mask = (groups == group)[:, None]
        columns.extend([np.broadcast_to(mask, logits.shape), mask * logits, mask * centered])
    return np.column_stack([column.ravel() for column in columns])


def main():
    data = load()
    directory = RUNS / 'v37_join'
    frames = {}
    for day in (24, 25, 26):
        _, events = split(data, day)
        frame = RUNS/'v31_verified'/f'v31_decay_d32_day{day}_s42'/'valid_events.npy'
        if not np.array_equal(np.load(frame), events):
            raise ValueError(f'Validation alignment differs for day {day}')
        size = (data['history'][events] > 0).sum(axis=1)
        probabilities = np.load(directory / f'raw{day}.npy')
        frames[day] = events, size, probabilities

    for day in (25, 26):
        inputs, targets = [], []
        for earlier in range(24, day):
            events, size, probabilities = frames[earlier]
            mask = calibration_training_mask(data, events, day)
            inputs.append(features(probabilities[mask], size[mask]))
            targets.append(data['label'][events[mask]].ravel())
        x, y = np.concatenate(inputs), np.concatenate(targets)
        pooled = LogisticRegression(max_iter=300).fit(x[:, :2], y)
        conditional = LogisticRegression(max_iter=300).fit(x, y)
        events, size, raw = frames[day]
        x_valid = features(raw, size)
        correction = (conditional.decision_function(x_valid)
                      - pooled.decision_function(x_valid[:, :2])).reshape(raw.shape)
        references = {'raw': raw}
        if day == 26:
            references['final'] = np.load(directory/'final.npy')
        for label, base in references.items():
            for weight in (0., .25, .5):
                prediction = expit(logit(base.clip(1e-6, 1-1e-6)) + weight * correction)
                record('v39_history_calibration', day, data['label'][events], prediction,
                       reference=label, weight=weight, fit_rows=len(y), valid_events=len(events))


if __name__ == '__main__':
    main()
