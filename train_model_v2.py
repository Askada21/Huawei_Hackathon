
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

TRAIN_FILE = "train.csv"
CUTOFF = int(datetime(2025, 1, 26, tzinfo=timezone.utc).timestamp())

# Keep memory usage manageable on an 8 GB Mac.
MAX_TRAIN = 500_000
MAX_VALID = 100_000

FEATURES = [
    "history_length",
    "candidate_in_history",
    "candidate_click_count",
    "time_since_last_click_hours",
    "time_bucket",
    "day_type",
    "item_impressions",
    "item_clicks",
    "item_ctr",
]

# Pass 1: collect item statistics from all pre-cutoff rows.
item_impressions = defaultdict(int)
item_clicks = defaultdict(int)
total_rows = 0
total_clicks = 0

print("Pass 1: calculating item statistics...", flush=True)

with open(TRAIN_FILE, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if int(row["timestamp"]) >= CUTOFF:
            continue

        item = row["item_id"]
        clicked = int(row["clicked"])

        item_impressions[item] += 1
        item_clicks[item] += clicked
        total_rows += 1
        total_clicks += clicked

global_ctr = total_clicks / total_rows
print(f"Global CTR: {global_ctr:.4%}", flush=True)


def extract_features(row):
    history = json.loads(row["history_item_ids"])
    history_times = json.loads(row["history_timestamps"])
    item = row["item_id"]
    timestamp = int(row["timestamp"])

    impressions = item_impressions[item]
    clicks = item_clicks[item]

    # Smooth rare items toward global CTR.
    smoothed_ctr = (
        (clicks + 50 * global_ctr) / (impressions + 50)
    )

    return [
        len(history),
        int(item in history),
        history.count(item),
        (timestamp - history_times[-1]) / 3600
        if history_times else -1,
        int(row["time_bucket"]),
        int(row["day_type"]),
        impressions,
        clicks,
        smoothed_ctr,
    ]


X_train, y_train = [], []
X_valid, y_valid = [], []

print("Pass 2: preparing features...", flush=True)

with open(TRAIN_FILE, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        is_train = int(row["timestamp"]) < CUTOFF

        if is_train and len(X_train) < MAX_TRAIN:
            X_train.append(extract_features(row))
            y_train.append(int(row["clicked"]))

        elif not is_train and len(X_valid) < MAX_VALID:
            X_valid.append(extract_features(row))
            y_valid.append(int(row["clicked"]))

        if len(X_train) >= MAX_TRAIN and len(X_valid) >= MAX_VALID:
            break

X_train = np.asarray(X_train, dtype=np.float32)
X_valid = np.asarray(X_valid, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.uint8)
y_valid = np.asarray(y_valid, dtype=np.uint8)

print("Training LightGBM v2...", flush=True)

model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
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
        lgb.early_stopping(30),
        lgb.log_evaluation(50),
    ],
)

predictions = model.predict_proba(X_valid)[:, 1]
auc = roc_auc_score(y_valid, predictions)

print(f"\nValidation ROC AUC v2: {auc:.6f}")

for feature, importance in sorted(
    zip(FEATURES, model.feature_importances_),
    key=lambda x: x[1],
    reverse=True,
):
    print(f"{feature}: {importance}")

model.booster_.save_model("lightgbm_v2.txt")
print("Saved: lightgbm_v2.txt")
