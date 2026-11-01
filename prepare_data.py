
import csv
import json
from datetime import datetime, timezone

TRAIN_FILE = "train.csv"
OUTPUT_TRAIN = "train_features.csv"
OUTPUT_VALID = "valid_features.csv"

CUTOFF = int(
    datetime(2025, 1, 26, tzinfo=timezone.utc).timestamp()
)

FEATURE_COLUMNS = [
    "history_length",
    "candidate_in_history",
    "candidate_click_count",
    "time_since_last_click_hours",
    "time_bucket",
    "day_type",
    "clicked",
]

MAX_TRAIN_ROWS = 500_000
MAX_VALID_ROWS = 100_000

train_count = 0
valid_count = 0

with open(TRAIN_FILE, newline="", encoding="utf-8") as source, \
     open(OUTPUT_TRAIN, "w", newline="", encoding="utf-8") as train_out, \
     open(OUTPUT_VALID, "w", newline="", encoding="utf-8") as valid_out:

    reader = csv.DictReader(source)
    train_writer = csv.DictWriter(train_out, fieldnames=FEATURE_COLUMNS)
    valid_writer = csv.DictWriter(valid_out, fieldnames=FEATURE_COLUMNS)

    train_writer.writeheader()
    valid_writer.writeheader()

    for row in reader:
        timestamp = int(row["timestamp"])
        history = json.loads(row["history_item_ids"])
        history_times = json.loads(row["history_timestamps"])
        candidate = row["item_id"]

        features = {
            "history_length": len(history),
            "candidate_in_history": int(candidate in history),
            "candidate_click_count": history.count(candidate),
            "time_since_last_click_hours": (
                (timestamp - history_times[-1]) / 3600
                if history_times else -1
            ),
            "time_bucket": int(row["time_bucket"]),
            "day_type": int(row["day_type"]),
            "clicked": int(row["clicked"]),
        }

        if timestamp < CUTOFF and train_count < MAX_TRAIN_ROWS:
            train_writer.writerow(features)
            train_count += 1
        elif timestamp >= CUTOFF and valid_count < MAX_VALID_ROWS:
            valid_writer.writerow(features)
            valid_count += 1

        if train_count >= MAX_TRAIN_ROWS and valid_count >= MAX_VALID_ROWS:
            break

print(f"Training rows: {train_count:,}")
print(f"Validation rows: {valid_count:,}")
print("Feature preparation completed!")
