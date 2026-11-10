"""Test whether slate context improves event click propensity over V194."""
import time

import numpy as np
from lightgbm import LGBMClassifier
from scipy.special import expit, logit

from common import cutoff, record
from run_v194_submission import main


def features(data, rows, left, right):
    result = np.empty((len(rows), 17), dtype=np.float32)
    for start in range(0, len(rows), 4096):
        batch = rows[start:start + 4096]
        history_ids = data['history'][batch]
        present = history_ids > 0
        count = present.sum(axis=1)
        timestamp = data['time'][batch]
        last = data['htime'][batch].max(axis=1)
        gap = np.where(count > 0, timestamp - last, -1)
        recent = (present & (data['htime'][batch] >= timestamp[:, None] - 86400)).sum(axis=1)
        history = np.einsum('eh,ehd->ed', present, right[history_ids])
        norm = np.linalg.norm(history, axis=1)
        history /= np.maximum(norm[:, None], 1e-8)
        candidates = right[data['item'][batch]]
        centroid = candidates.mean(axis=1)
        affinity = np.einsum('ed,ecd->ec', history, candidates)
        user = np.einsum('ed,ecd->ec', left[data['user'][batch]], candidates)
        hour = (timestamp % 86400) * (2 * np.pi / 86400)
        result[start:start + len(batch)] = np.column_stack([
            count, gap, recent, np.sin(hour), np.cos(hour), data['day'][batch],
            norm / np.maximum(count, 1),
            np.linalg.norm(centroid, axis=1),
            affinity.mean(axis=1), affinity.max(axis=1), affinity.std(axis=1),
            user.mean(axis=1), user.max(axis=1), user.std(axis=1),
            (affinity * user).mean(axis=1),
            (user > 0).sum(axis=1), (affinity > 0).sum(axis=1),
        ])
    return result


def screen(day, data, fit, target, valid, left, right, baseline):
    start = time.monotonic()
    train_x = features(data, fit, left, right)
    valid_x = features(target, valid, left, right)
    hit = data['label'][fit].any(axis=1)

    def predict(n_features):
        model = LGBMClassifier(n_estimators=150, learning_rate=.04, num_leaves=15,
                               min_child_samples=500, reg_lambda=10,
                               n_jobs=4, random_state=42, verbosity=-1)
        model.fit(train_x[:, :n_features], hit)
        return model.predict(valid_x[:, :n_features], raw_score=True)

    history_only = predict(7)
    slate = predict(train_x.shape[1])
    v194 = logit(baseline.astype(np.float64))
    labels = target['label'][valid]
    for name, correction in (
        ('slate_increment', slate - history_only),
        ('full_event', slate - logit(hit.mean())),
    ):
        for weight in (0., .25, .5, 1.):
            record('v202_slate_propensity', day, labels,
                   expit(v194 + weight * correction[:, None]),
                   variant=name, weight=weight, fitting_events=len(fit),
                   fitting_end_utc=int(data['time'][fit].max()), validation_events=len(valid),
                   validation_start_utc=cutoff(day), elapsed_s=round(time.monotonic() - start, 2))


if __name__ == '__main__':
    main(days=(24, 25, 26), screen=screen)
