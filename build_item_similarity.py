
import csv
import json
import sqlite3
from datetime import datetime, timezone

SOURCE = "train.csv"
DATABASE = "item_similarity.sqlite"

CUTOFF = int(
    datetime(2025, 1, 25, tzinfo=timezone.utc).timestamp()
)

# Only compare nearby clicks within each history.
WINDOW = 5
TOP_K = 30
BATCH_SIZE = 5000

conn = sqlite3.connect(DATABASE)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA synchronous=NORMAL")

conn.execute("""
CREATE TABLE IF NOT EXISTS pair_counts (
    item_a TEXT NOT NULL,
    item_b TEXT NOT NULL,
    count INTEGER NOT NULL,
    PRIMARY KEY (item_a, item_b)
)
""")

conn.execute("""
CREATE TABLE IF NOT EXISTS item_neighbors (
    item_a TEXT NOT NULL,
    item_b TEXT NOT NULL,
    similarity REAL NOT NULL,
    PRIMARY KEY (item_a, item_b)
)
""")

# Rebuild from scratch on every run.
conn.execute("DELETE FROM pair_counts")
conn.execute("DELETE FROM item_neighbors")
conn.commit()

pending = {}
events_processed = 0


def flush_pairs():
    if not pending:
        return

    conn.executemany("""
        INSERT INTO pair_counts (item_a, item_b, count)
        VALUES (?, ?, ?)
        ON CONFLICT(item_a, item_b)
        DO UPDATE SET count = count + excluded.count
    """, [
        (a, b, count)
        for (a, b), count in pending.items()
    ])

    conn.commit()
    pending.clear()


print("Building item co-occurrence statistics...", flush=True)

with open(SOURCE, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    last_event = None

    for row in reader:
        timestamp = int(row["timestamp"])

        if timestamp >= CUTOFF:
            continue

        event_id = row["event_id"]

        # One history per event.
        if event_id == last_event:
            continue

        last_event = event_id

        history = json.loads(row["history_item_ids"])

        for i, item_a in enumerate(history):
            start = max(0, i - WINDOW)

            for item_b in history[start:i]:
                if item_a == item_b:
                    continue

                # Undirected pair.
                a, b = sorted((item_a, item_b))
                key = (a, b)

                pending[key] = pending.get(key, 0) + 1

        events_processed += 1

        if events_processed % BATCH_SIZE == 0:
            flush_pairs()

        if events_processed % 50000 == 0:
            print(
                f"Processed {events_processed:,} events",
                flush=True
            )

flush_pairs()

print("Selecting strongest neighbors...", flush=True)

# Index makes neighbor lookup faster.
conn.execute("""
CREATE INDEX IF NOT EXISTS idx_pair_b
ON pair_counts(item_b)
""")

# For each item, select the strongest observed neighbors.
# This avoids creating a dense 50,000 x 50,000 matrix.
items = conn.execute("""
    SELECT item_a FROM pair_counts
    UNION
    SELECT item_b FROM pair_counts
""").fetchall()

for index, (item,) in enumerate(items, 1):
    neighbors = conn.execute("""
        SELECT item_b, count
        FROM pair_counts
        WHERE item_a = ?

        UNION ALL

        SELECT item_a, count
        FROM pair_counts
        WHERE item_b = ?

        ORDER BY count DESC
        LIMIT ?
    """, (item, item, TOP_K)).fetchall()

    conn.executemany("""
        INSERT OR REPLACE INTO item_neighbors
        (item_a, item_b, similarity)
        VALUES (?, ?, ?)
    """, [
        (item, neighbor, float(count))
        for neighbor, count in neighbors
    ])

    if index % 5000 == 0:
        conn.commit()
        print(
            f"Neighbors completed: {index:,}",
            flush=True
        )

conn.commit()

pair_count = conn.execute(
    "SELECT COUNT(*) FROM pair_counts"
).fetchone()[0]

neighbor_count = conn.execute(
    "SELECT COUNT(*) FROM item_neighbors"
).fetchone()[0]

print("\n==============================")
print(f"Events processed: {events_processed:,}")
print(f"Unique pairs: {pair_count:,}")
print(f"Stored neighbors: {neighbor_count:,}")
print("==============================")
print(f"Saved database: {DATABASE}")

conn.close()
