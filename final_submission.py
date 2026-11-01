import csv
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from itertools import groupby

import numpy as np
import lightgbm as lgb


TRAIN_FILE = "train.csv"
EVALUATION_FILE = "evaluation.csv"
MODEL_FILE = "lightgbm_v5.txt"
SIMILARITY_DB = "item_similarity.sqlite"
OUTPUT_FILE = "submission.csv"

STATS_CUTOFF = int(
    datetime(2025, 1, 25, tzinfo=timezone.utc).timestamp()
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
    "cf_max_similarity",
    "cf_mean_similarity",
    "cf_recent_similarity",
]


# ============================================================
# 1. LOAD MODEL
# ============================================================

print("Loading LightGBM v5...", flush=True)

model = lgb.Booster(model_file=MODEL_FILE)

if model.num_feature() != len(FEATURES):
    raise ValueError(
        f"Model expects {model.num_feature()} features, "
        f"but script provides {len(FEATURES)}."
    )

print("Model loaded successfully.", flush=True)


# ============================================================
# 2. REBUILD HISTORICAL STATISTICS
# ============================================================

print("Building historical statistics...", flush=True)

item_stats = defaultdict(lambda: [0, 0])
user_stats = defaultdict(lambda: [0, 0])

total_rows = 0
total_clicks = 0

with open(TRAIN_FILE, newline="", encoding="utf-8") as f:
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
# 3. LOAD COLLABORATIVE FILTERING
# ============================================================

print("Loading item similarity database...", flush=True)

neighbors = defaultdict(dict)

conn = sqlite3.connect(SIMILARITY_DB)

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
# 4. FEATURE ENGINEERING
# ============================================================

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
    base = [base_features(row) for row in event_rows]

    ctrs = [x[8] for x in base]
    impressions = [x[7] for x in base]
    matches = [x[2] for x in base]

    mean_ctr = sum(ctrs) / len(ctrs)
    max_ctr = max(ctrs)

    result = []

    for i, x in enumerate(base):
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

        cf = collaborative_features(event_rows[i])

        result.append(x + extra + cf)

    return result


# ============================================================
# 5. GENERATE PREDICTIONS
# ============================================================

print("Generating evaluation predictions...", flush=True)

total_predictions = 0
batch_features = []
batch_rowids = []

BATCH_SIZE = 10_000


def write_batch(writer):
    global total_predictions

    if not batch_features:
        return

    X = np.asarray(batch_features, dtype=np.float32)

    probabilities = model.predict(X)

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Non-finite predictions detected.")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Predictions outside [0, 1].")

    for rowid, probability in zip(
        batch_rowids, probabilities
    ):
        writer.writerow([
            rowid,
            f"{float(probability):.10f}",
        ])

    total_predictions += len(probabilities)

    batch_features.clear()
    batch_rowids.clear()

    if total_predictions % 100_000 == 0:
        print(
            f"Predicted {total_predictions:,} rows",
            flush=True,
        )


with open(
    EVALUATION_FILE,
    newline="",
    encoding="utf-8",
) as evaluation_file, open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8",
) as output_file:

    reader = csv.DictReader(evaluation_file)
    writer = csv.writer(output_file)

    writer.writerow([
        "rowid",
        "predicted_probability",
    ])

    for event_id, group in groupby(
        reader,
        key=lambda row: row["event_id"],
    ):
        rows = list(group)

        if len(rows) != 10:
            raise ValueError(
                f"Event {event_id} has {len(rows)} rows, "
                "expected 10."
            )

        features = add_event_features(rows)

        for row, feature_vector in zip(rows, features):
            batch_rowids.append(row["rowid"])
            batch_features.append(feature_vector)

        if len(batch_features) >= BATCH_SIZE:
            write_batch(writer)

    write_batch(writer)


# ============================================================
# 6. FINAL CHECK
# ============================================================

expected_rows = 529_390

print("\n==============================")
print("SUBMISSION COMPLETE")
print("==============================")
print(f"File: {OUTPUT_FILE}")
print(f"Predictions: {total_predictions:,}")

if total_predictions != expected_rows:
    raise ValueError(
        f"Expected {expected_rows:,} predictions, "
        f"got {total_predictions:,}."
    )

print("All checks passed.")
print("Ready for Huawei TechArena upload.")
