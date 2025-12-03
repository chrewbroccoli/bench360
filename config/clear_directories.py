#!/usr/bin/env python3
import csv
import pathlib

RUN_DIR = pathlib.Path("run_report")
READ_DIR = pathlib.Path("readings")

for csv_path in RUN_DIR.glob("*.csv"):
    # Read first row and find num_queries
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        first_row = next(reader, None)

    if not first_row:
        continue

    num_q = first_row.get("num_queries")
    if num_q is None:
        continue

    try:
        if int(num_q) != 10:
            continue
    except ValueError:
        continue

    # num_queries == 10 → delete
    stem = csv_path.stem
    hash_part = stem.rsplit("_", 1)[-1]

    print("Deleting run_report file:", csv_path)
    csv_path.unlink()

    # Delete matching readings files
    for read_path in READ_DIR.glob(f"*{hash_part}*"):
        print("Deleting readings file:", read_path)
        read_path.unlink()

print("Cleanup finished.")
