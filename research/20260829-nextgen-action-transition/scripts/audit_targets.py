#!/usr/bin/env python3
"""Validate the next-generation target shards before model selection."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True, help="glob for JSONL target shards")
    parser.add_argument("--roots", required=True)
    parser.add_argument("--excluded-roots", required=True)
    parser.add_argument("--expected", type=int, default=6000)
    parser.add_argument("--teacher-depth", type=int, default=7)
    parser.add_argument("--teacher-nodes", type=int, default=1_000_000)
    args = parser.parse_args()

    roots = [json.loads(line) for line in Path(args.roots).read_text().splitlines() if line.strip()]
    excluded = {
        row["source_game_id"]
        for line in Path(args.excluded_roots).read_text().splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    paths = sorted(glob.glob(args.targets))
    rows = [
        json.loads(line)
        for path in paths
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    ids = [row["id"] for row in rows]
    sources = [row["sourceGameId"] for row in rows]
    roots_by_id = {row["id"]: row for row in roots}
    if len(rows) != args.expected:
        raise SystemExit(f"expected {args.expected} targets, found {len(rows)}")
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate target IDs")
    if len(set(sources)) != len(sources):
        raise SystemExit("a source game has more than one target")
    if set(sources) & excluded:
        raise SystemExit("target source overlaps excluded roots")
    if set(ids) != set(roots_by_id):
        raise SystemExit("target IDs do not match the frozen roots")
    if any(
        row["teacher"]["maxNodes"] != args.teacher_nodes
        or row["teacher"]["depth"] != args.teacher_depth
        for row in rows
    ):
        raise SystemExit("unexpected teacher configuration")

    turns = Counter(roots_by_id[row_id]["state"].split(".")[4] for row_id in ids)
    partitions = Counter(row["partition"] for row in rows)
    depths = Counter(row["completedDepth"] for row in rows)
    exhausted = sum(row["exhausted"] for row in rows)
    print(json.dumps({
        "targets": len(rows),
        "sourceGames": len(set(sources)),
        "turns": turns,
        "partitions": partitions,
        "completedDepth": depths,
        "exhausted": exhausted,
        "teacher": rows[0]["teacher"],
        "shards": paths,
    }, sort_keys=True, default=dict))


if __name__ == "__main__":
    main()
