
import csv
import json
import hashlib
from collections import defaultdict
from datetime import datetime, timezone
from itertools import groupby

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

SOURCE = "train.csv"

STATS_CUTOFF = int(
    datetime(2025, 1, 25, tzinfo=timezone.utc).timestamp()
)
VALID_CUTOFF = int(
    datetime(2025, 1, 26, tzinfo=timezone.utc).timestamp()
)

FEATURES = [
    "history_length",
    "candidate_in_history",
    "candidate_click_count",
    "time_since_last_click_hours",
    "time_since_candidate_click_hours",
    "time_bucket",
    "day_type",
    "item_impressions",
    "item_ctr",
    "user_impressions",
    "user_ctr",
    "item_ctr_rank",
    "item_ctr_vs_event_mean",
    "item_ctr_vs_event_max",
    "item_impressions_rank",
    "history_match_rank",
]

item_stats = defaultdict(lambda: [0, 0])
user_stats = defaultdict(lambda: [0, 0])

total_rows = 0
total_clicks = 0

print("Pass 1: historical statistics...", flush=True)

with open(SOURCE, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if int(row["timestamp"]) >= STATS_CUTOFF:
            continue

        clicked = int(row["clicked"])

        item_stats[row["item_id"]][0] += 1
        item_stats[row["item_id"]][1] += clicked

        user_stats[row["user_id"]][0] += 1
        user_stats[row["user_id"]][1] += clicked

        total_rows += 1
        total_clicks += clicked

global_ctr = total_clicks / total_rows
print(f"Historical CTR: {global_ctr:.4%}", flush=True)


def keep_event(event_id, fraction):
    digest = hashlib.blake2b(
        event_id.encode("utf-8"), digest_size=8
    ).digest()
    value = int.from_bytes(digest, "big") / (2 ** 64)
    return value < fraction


def base_features(row):
    history = json.loads(row["history_item_ids"])
    times = json.loads(row["history_timestamps"])

    item = row["item_id"]
    user = row["user_id"]
    timestamp = int(row["timestamp"])

    item_n, item_c = item_stats.get(item, (0, 0))
    user_n, user_c = user_stats.get(user, (0, 0))

    candidate_times = [
        t for h, t in zip(history, times) if h == item
    ]

    item_ctr = (
        item_c + 50 * global_ctr
    ) / (item_n + 50)

    user_ctr = (
        user_c + 50 * global_ctr
    ) / (user_n + 50)

    features = [
        len(history),
        int(item in history),
        history.count(item),
        (timestamp - times[-1]) / 3600 if times else -1,
        (timestamp - candidate_times[-1]) / 3600
        if candidate_times else -1,
        int(row["time_bucket"]),
        int(row["day_type"]),
        item_n,
        item_ctr,
        user_n,
        user_ctr,
    ]

    return features


def add_event_features(event_rows):
    base = [base_features(row) for row in event_rows]

    ctrs = [x[8] for x in base]
    impressions = [x[7] for x in base]
    matches = [x[2] for x in base]

    mean_ctr = sum(ctrs) / len(ctrs)
    max_ctr = max(ctrs)

    result = []

    for x in base:
        ctr_rank = 1 + sum(v > x[8] for v in ctrs)
        impression_rank = 1 + sum(
            v > x[7] for v in impressions
        )
        match_rank = 1 + sum(
            v > x[2] for v in matches
        )

        extra = [
            ctr_rank,
            x[8] - mean_ctr,
            x[8] - max_ctr,
            impression_rank,
            match_rank,
        ]

        result.append(x + extra)

    return result


X_train, y_train = [], []
X_valid, y_valid = [], []

print("Pass 2: event-level features...", flush=True)

with open(SOURCE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    # Requires rows of each event to be consecutive.
    for event_id, group in groupby(
        reader, key=lambda row: row["event_id"]
    ):
        rows = list(group)

        if len(rows) != 10:
            raise ValueError(
                f"Event {event_id} has {len(rows)} consecutive "
                "rows instead of 10. Check CSV event ordering."
            )

        timestamp = int(rows[0]["timestamp"])

        if STATS_CUTOFF <= timestamp < VALID_CUTOFF:
            if not keep_event(event_id, 0.60):
                continue
            target_X, target_y = X_train, y_train

        elif timestamp >= VALID_CUTOFF:
            if not keep_event(event_id, 0.50):
                continue
            target_X, target_y = X_valid, y_valid

        else:
            continue

        features = add_event_features(rows)

        for row, x in zip(rows, features):
            target_X.append(x)
            target_y.append(int(row["clicked"]))

X_train = np.asarray(X_train, dtype=np.float32)
X_valid = np.asarray(X_valid, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.uint8)
y_valid = np.asarray(y_valid, dtype=np.uint8)

print(f"Training rows: {len(y_train):,}")
print(f"Validation rows: {len(y_valid):,}")

print("Training LightGBM v4...", flush=True)

model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=700,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=100,
    random_state=42,
    n_jobs=4,
    verbosity=-1,
)

model.fit(
    X_train,
    y_train,
    eval_X=X_valid,
    eval_y=y_valid,
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(40),
        lgb.log_evaluation(50),
    ],
)

predictions = model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, predictions)

print("\n==============================")
print(f"Validation ROC AUC v4: {auc:.6f}")
print("==============================")

print("\nFeature importance:")

for feature, importance in sorted(
    zip(FEATURES, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True,
):
    print(f"{feature}: {importance}")

model.booster_.save_model("lightgbm_v4.txt")
print("\nSaved: lightgbm_v4.txt")
