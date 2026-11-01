
import csv
from collections import defaultdict
from datetime import datetime, timezone

DATA_FILE = "train.csv"

daily_rows = defaultdict(int)
daily_clicks = defaultdict(int)
daily_events = defaultdict(set)

with open(DATA_FILE, "r", encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)

    for i, row in enumerate(reader, start=1):
        timestamp = int(row["timestamp"])
        date = datetime.fromtimestamp(
            timestamp, tz=timezone.utc
        ).strftime("%Y-%m-%d")

        daily_rows[date] += 1
        daily_clicks[date] += int(row["clicked"])
        daily_events[date].add(row["event_id"])

        if i % 1_000_000 == 0:
            print(f"Processed {i:,} rows...", flush=True)

print("\n=== TRAINING DATA SUMMARY ===")

for date in sorted(daily_rows):
    rows = daily_rows[date]
    clicks = daily_clicks[date]
    events = len(daily_events[date])
    ctr = clicks / rows

    print(
        f"{date}: "
        f"{rows:,} rows | "
        f"{events:,} events | "
        f"{clicks:,} clicks | "
        f"CTR = {ctr:.4%}"
    )

print("\nFinished!")
