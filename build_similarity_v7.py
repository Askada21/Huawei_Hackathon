
import csv
import json
import hashlib
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from itertools import groupby

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE = "train.csv"
SIMILARITY_DB = "item_similarity_v7.sqlite"
MODEL_OUTPUT = "lightgbm_v7.txt"

STATS_CUTOFF = int(
    datetime(2025, 1, 23, tzinfo=timezone.utc).timestamp()
)

VALID_CUTOFF = int(
    datetime(2025, 1, 26, tzinfo=timezone.utc).timestamp()
)

TRAIN_CAPACITY = 3_000_000
VALID_CAPACITY = 600_000

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
    "cf_max_similarity",
    "cf_mean_similarity",
    "cf_recent_similarity",
]


# ============================================================
# PASS 1: HISTORICAL STATISTICS
# ============================================================

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

if total_rows == 0:
    raise ValueError("No historical rows found.")

global_ctr = total_clicks / total_rows

print(f"Historical CTR: {global_ctr:.4%}", flush=True)


# ============================================================
# LOAD NORMALIZED COLLABORATIVE FILTERING
# ============================================================

print("Loading normalized item neighbors...", flush=True)

conn = sqlite3.connect(SIMILARITY_DB)

neighbors = defaultdict(dict)

for item_a, item_b, similarity in conn.execute(
    "SELECT item_a, item_b, similarity FROM item_neighbors"
):
    neighbors[item_a][item_b] = float(similarity)

conn.close()

print(
    f"Loaded neighbors for {len(neighbors):,} items",
    flush=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def keep_event(event_id, fraction):
    digest = hashlib.blake2b(
        event_id.encode("utf-8"),
        digest_size=8,
    ).digest()

    value = int.from_bytes(digest, "big") / (2**64)
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
        t for h, t in zip(history, times)
        if h == item
    ]

    item_ctr = (
        item_c + 50 * global_ctr
    ) / (item_n + 50)

    user_ctr = (
        user_c + 50 * global_ctr
    ) / (user_n + 50)

    return [
        len(history),
        int(item in history),
        history.count(item),
        (timestamp - times[-1]) / 3600 if times else -1,
        (
            (timestamp - candidate_times[-1]) / 3600
            if candidate_times else -1
        ),
        int(row["time_bucket"]),
        int(row["day_type"]),
        item_n,
        item_ctr,
        user_n,
        user_ctr,
    ]


def collaborative_features(row):
    history = json.loads(row["history_item_ids"])
    candidate = row["item_id"]

    candidate_neighbors = neighbors.get(candidate, {})

    scores = [
        candidate_neighbors.get(item, 0.0)
        for item in history
    ]

    if not scores:
        return [0.0, 0.0, 0.0]

    return [
        max(scores),
        sum(scores) / len(scores),
        sum(scores[-5:]) / min(5, len(scores)),
    ]


def add_event_features(event_rows):
    base = [
        base_features(row)
        for row in event_rows
    ]

    ctrs = [x[8] for x in base]
    impressions = [x[7] for x in base]
    matches = [x[2] for x in base]

    mean_ctr = sum(ctrs) / len(ctrs)
    max_ctr = max(ctrs)

    result = []

    for row, x in zip(event_rows, base):
        ctr_rank = 1 + sum(
            value > x[8] for value in ctrs
        )

        impression_rank = 1 + sum(
            value > x[7] for value in impressions
        )

        match_rank = 1 + sum(
            value > x[2] for value in matches
        )

        extra = [
            ctr_rank,
            x[8] - mean_ctr,
            x[8] - max_ctr,
            impression_rank,
            match_rank,
        ]

        cf = collaborative_features(row)
        result.append(x + extra + cf)

    return result


# ============================================================
# PASS 2: CREATE DISK-BACKED FEATURE ARRAYS
# ============================================================

print("Pass 2: preparing disk-backed features...", flush=True)

n_features = len(FEATURES)

X_train = np.memmap(
    "v7_train_X.dat",
    dtype="float32",
    mode="w+",
    shape=(TRAIN_CAPACITY, n_features),
)

y_train = np.memmap(
    "v7_train_y.dat",
    dtype="uint8",
    mode="w+",
    shape=(TRAIN_CAPACITY,),
)

X_valid = np.memmap(
    "v7_valid_X.dat",
    dtype="float32",
    mode="w+",
    shape=(VALID_CAPACITY, n_features),
)

y_valid = np.memmap(
    "v7_valid_y.dat",
    dtype="uint8",
    mode="w+",
    shape=(VALID_CAPACITY,),
)

train_count = 0
valid_count = 0

with open(SOURCE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for event_id, group in groupby(
        reader,
        key=lambda row: row["event_id"],
    ):
        rows = list(group)

        if len(rows) != 10:
            raise ValueError(
                f"Unexpected event size for {event_id}: "
                f"{len(rows)}"
            )

        timestamp = int(rows[0]["timestamp"])

        if STATS_CUTOFF <= timestamp < VALID_CUTOFF:
            if not keep_event(event_id, 0.60):
                continue

            features = add_event_features(rows)
            n = len(rows)

            if train_count + n > TRAIN_CAPACITY:
                raise MemoryError("Increase TRAIN_CAPACITY")

            X_train[train_count:train_count + n] = features
            y_train[train_count:train_count + n] = [
                int(row["clicked"]) for row in rows
            ]

            train_count += n

        elif timestamp >= VALID_CUTOFF:
            if not keep_event(event_id, 0.50):
                continue

            features = add_event_features(rows)
            n = len(rows)

            if valid_count + n > VALID_CAPACITY:
                raise MemoryError("Increase VALID_CAPACITY")

            X_valid[valid_count:valid_count + n] = features
            y_valid[valid_count:valid_count + n] = [
                int(row["clicked"]) for row in rows
            ]

            valid_count += n


X_train.flush()
y_train.flush()
X_valid.flush()
y_valid.flush()

X_train = X_train[:train_count]
y_train = y_train[:train_count]
X_valid = X_valid[:valid_count]
y_valid = y_valid[:valid_count]

print(f"Training rows: {train_count:,}", flush=True)
print(f"Validation rows: {valid_count:,}", flush=True)

if train_count == 0 or valid_count == 0:
    raise ValueError("Training or validation dataset is empty.")


# ============================================================
# TRAIN LIGHTGBM V7
# ============================================================

print("Training LightGBM v7...", flush=True)

model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=150,
    max_bin=127,
    random_state=42,
    n_jobs=4,
    verbosity=-1,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="auc",
    callbacks=[
        lgb.early_stopping(40),
        lgb.log_evaluation(50),
    ],
)


# ============================================================
# EVALUATION
# ============================================================

predictions = model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, predictions)

print("\n==============================")
print(f"Validation ROC AUC v7: {auc:.6f}")
print("==============================")

print("\nFeature importance:")

for feature, importance in sorted(
    zip(FEATURES, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True,
):
    print(f"{feature}: {importance}")


# ============================================================
# SAVE MODEL
# ============================================================

model.booster_.save_model(MODEL_OUTPUT)
print(f"\nSaved: {MODEL_OUTPUT}")
