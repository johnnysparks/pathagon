#!/usr/bin/env python3
"""Select a color- and partition-balanced deep-label calibration slice."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-cell", type=int, default=64)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.base.read_text().splitlines() if line.strip()]
    buckets: dict[tuple[str, str], list[dict]] = {
        (partition, turn): []
        for partition in ("train", "heldout")
        for turn in ("L", "D")
    }
    for row in rows:
        turn = row["state"].split(".")[4]
        buckets[(row["partition"], turn)].append(row)
    selected = [
        row
        for row in rows
        if row in sum((buckets[key][: args.per_cell] for key in sorted(buckets)), [])
    ]
    expected = args.per_cell * 4
    if len(selected) != expected:
        counts = {f"{partition}-{turn}": len(buckets[(partition, turn)]) for partition, turn in buckets}
        raise SystemExit(f"calibration slice has {len(selected)} rows, expected {expected}; buckets={counts}")
    if len({row["id"] for row in selected}) != expected:
        raise SystemExit("calibration slice contains duplicate IDs")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in selected))
    print(json.dumps({
        "roots": len(selected),
        "turns": Counter(row["state"].split(".")[4] for row in selected),
        "partitions": Counter(row["partition"] for row in selected),
        "output": str(args.output),
    }, sort_keys=True, default=dict))


if __name__ == "__main__":
    main()
