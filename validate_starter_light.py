
import csv
import math
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import roc_auc_score

SOURCE = "train.csv"
CUTOFF = int(datetime(2025, 1, 26, tzinfo=timezone.utc).timestamp())
K = 50

item_stats = defaultdict(lambda: [0, 0])
user_stats = defaultdict(lambda: [0, 0])

total_rows = 0
total_clicks = 0

print("Pass 1: building statistics...", flush=True)

with open(SOURCE, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if int(row["timestamp"]) >= CUTOFF:
            continue

        clicked = int(row["clicked"])

        item_stats[row["item_id"]][0] += 1
        item_stats[row["item_id"]][1] += clicked

        user_stats[row["user_id"]][0] += 1
        user_stats[row["user_id"]][1] += clicked

        total_rows += 1
        total_clicks += clicked

global_ctr = total_clicks / total_rows


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


global_logit = logit(global_ctr)

predictions = []
labels = []

print("Pass 2: validating January 26...", flush=True)

with open(SOURCE, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if int(row["timestamp"]) < CUTOFF:
            continue

        item_n, item_c = item_stats.get(row["item_id"], (0, 0))
        user_n, user_c = user_stats.get(row["user_id"], (0, 0))

        item_ctr = (item_c + K * global_ctr) / (item_n + K)
        user_ctr = (user_c + K * global_ctr) / (user_n + K)

        score = logit(item_ctr) + logit(user_ctr) - global_logit

        predictions.append(score)
        labels.append(int(row["clicked"]))

auc = roc_auc_score(
    np.asarray(labels, dtype=np.uint8),
    np.asarray(predictions, dtype=np.float32),
)

print("\n==============================")
print(f"Starter CTR-only ROC AUC: {auc:.6f}")
print("==============================")
print(f"Validation rows: {len(labels):,}")
