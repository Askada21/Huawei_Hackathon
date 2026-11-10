"""Optimize the conditional likelihood of the clicked item within each slate."""
import time

import numpy as np
from lightgbm import LGBMClassifier
from scipy.special import expit, logit, softmax

from common import cutoff, record
from run_v194_submission import main
from run_v203_slate_choice import features


def grouped_objective(labels, scores):
    probabilities = softmax(scores.reshape(-1, 10), axis=1).ravel()
    gradient = probabilities - labels
    # Twice the diagonal is a diagonal upper bound on the group Hessian.
    hessian = 2 * probabilities * (1 - probabilities)
    return gradient, np.maximum(hessian, 1e-8)


def screen(day, data, fit, target, valid, left, right, baseline):
    start = time.monotonic()
    clicked = fit[data['label'][fit].any(axis=1)]
    labels = data['label'][clicked]
    if not np.all(labels.sum(axis=1) == 1):
        raise ValueError('Conditional choice requires one click per training event')
    x = features(data, clicked, right)
    model = LGBMClassifier(objective=grouped_objective, n_estimators=200, learning_rate=.05,
                           num_leaves=31, min_child_samples=250, reg_lambda=10,
                           n_jobs=4, random_state=42, verbosity=-1)
    model.fit(x.reshape(-1, x.shape[-1]), labels.ravel())
    del x
    x = features(target, valid, right)
    scores = model.predict(x.reshape(-1, x.shape[-1]), raw_score=True).reshape(-1, 10)
    scores -= scores.mean(axis=1, keepdims=True)
    base = logit(baseline.astype(np.float64))
    for weight in (.125, .25, .5):
        record('v204_grouped_choice', day, target['label'][valid], expit(base + weight * scores),
               weight=weight, fitting_events=len(clicked), fitting_end_utc=int(data['time'][fit].max()),
               validation_events=len(valid), validation_start_utc=cutoff(day),
               elapsed_s=round(time.monotonic() - start, 2))


if __name__ == '__main__':
    main(days=(26, 25), screen=screen)
