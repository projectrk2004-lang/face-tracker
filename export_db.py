
import csv
import sqlite3
from pathlib import Path

# Database path
db_path = Path("db/faces.db")

# Create output folder if it does not exist
output_dir = Path("sample_output")
output_dir.mkdir(parents=True, exist_ok=True)

if not db_path.exists():
    raise FileNotFoundError(f"Database not found: {db_path}")

conn = sqlite3.connect(db_path)

try:
    tables = [
        ("faces", "face_id, first_seen"),
        ("events", "event_id, face_id, event_type, timestamp, image_path"),
    ]

    for table, cols in tables:
        rows = conn.execute(
            f"SELECT {cols} FROM {table}"
        ).fetchall()

        # Export table data to CSV
        with open(
            output_dir / f"{table}.csv",
            "w",
            newline="",
            encoding="utf-8"
        ) as f:
            writer = csv.writer(f)
            writer.writerow([c.strip() for c in cols.split(",")])
            writer.writerows(rows)

        print(table, len(rows), "rows exported")

    # Count unique registered faces
    unique = conn.execute(
        "SELECT COUNT(*) FROM faces"
    ).fetchone()[0]

    print("Unique visitors:", unique)

finally:
    conn.close()
