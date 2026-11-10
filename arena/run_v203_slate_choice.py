"""Test a conditional click model using nonlinear history and slate features."""
import time

import numpy as np
from lightgbm import LGBMClassifier
from scipy.special import expit, logit

from common import cutoff, record
from run_v194_submission import main


def features(data, rows, right):
    result = np.empty((len(rows), 10, 31), dtype=np.float32)
    for start in range(0, len(rows), 2048):
        batch = rows[start:start + 2048]
        ids = data['item'][batch]
        items = right[ids]
        hist_ids = data['history'][batch]
        present = hist_ids > 0
        age = (data['time'][batch, None] - data['htime'][batch]) / 86400
        history = right[hist_ids]
        similarities = np.einsum('ecd,ehd->ech', items, history)
        valid_sim = np.where(present[:, None, :], similarities, 0)
        count = present.sum(axis=1, keepdims=True)
        average = valid_sim.sum(axis=2) / np.maximum(count, 1)
        recency = np.exp(-np.maximum(age, 0)) * present
        recent_average = np.einsum('ech,eh->ec', valid_sim, recency)
        recent_average /= np.maximum(recency.sum(axis=1, keepdims=True), 1e-8)
        last = np.take_along_axis(valid_sim, data['htime'][batch].argmax(axis=1)[:, None, None],
                                  axis=2).squeeze(axis=2)
        max_similarity = np.where(present[:, None, :], similarities, -1).max(axis=2)
        matches = (ids[:, :, None] == hist_ids[:, None, :]).sum(axis=2)
        peer = np.einsum('ecd,ekd->eck', items, items)
        diagonal = np.arange(10)
        peer[:, diagonal, diagonal] = 0
        peer_mean = peer.sum(axis=2) / 9
        peer[:, diagonal, diagonal] = -np.inf
        peer_max = peer.max(axis=2)
        relative = [v - v.mean(axis=1, keepdims=True)
                    for v in (average, recent_average, last, max_similarity, peer_mean)]
        broadcast = lambda x: np.broadcast_to(x, ids.shape)
        result[start:start + len(batch)] = np.concatenate([
            items,
            np.stack([average, recent_average, last, max_similarity, matches,
                      peer_mean, peer_max, broadcast(count),
                      broadcast((data['time'][batch] % 86400 // 21600)[:, None]),
                      broadcast(data['day'][batch, None]), *relative], axis=2),
        ], axis=2)
    return result


def screen(day, data, fit, target, valid, left, right, baseline):
    start = time.monotonic()
    clicked = fit[data['label'][fit].any(axis=1)]
    x = features(data, clicked, right)
    model = LGBMClassifier(n_estimators=250, learning_rate=.05, num_leaves=31,
                           min_child_samples=250, reg_lambda=10, n_jobs=4,
                           random_state=42, verbosity=-1)
    model.fit(x.reshape(-1, x.shape[-1]), data['label'][clicked].ravel())
    del x
    x = features(target, valid, right)
    logits = model.predict(x.reshape(-1, x.shape[-1]), raw_score=True).reshape(-1, 10)
    logits -= logits.mean(axis=1, keepdims=True)
    base = logit(baseline.astype(np.float64))
    for weight in (0., .125, .25, .5):
        record('v203_slate_choice', day, target['label'][valid], expit(base + weight * logits),
               weight=weight, fitting_events=len(clicked), fitting_end_utc=int(data['time'][fit].max()),
               validation_events=len(valid), validation_start_utc=cutoff(day),
               elapsed_s=round(time.monotonic() - start, 2))
        adjusted = baseline.astype(np.float64) * np.exp(weight * logits)
        adjusted *= baseline.sum(axis=1, keepdims=True) / adjusted.sum(axis=1, keepdims=True)
        record('v203_slate_choice_fixed_mass', day, target['label'][valid], adjusted,
               weight=weight, fitting_events=len(clicked), fitting_end_utc=int(data['time'][fit].max()),
               validation_events=len(valid), validation_start_utc=cutoff(day),
               elapsed_s=round(time.monotonic() - start, 2))


if __name__ == '__main__':
    main(days=(24, 25, 26), screen=screen)
