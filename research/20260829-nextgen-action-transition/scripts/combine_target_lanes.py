#!/usr/bin/env python3
"""Combine source-disjoint target lanes into one deterministic corpus."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def read(pattern: str) -> list[dict]:
    return [
        json.loads(line)
        for path in sorted(glob.glob(pattern))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", action="append", required=True, help="target shard glob; repeat per lane")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected", type=int, default=10_000)
    parser.add_argument("--teacher-depth", type=int, default=7)
    parser.add_argument("--teacher-nodes", type=int, default=1_000_000)
    parser.add_argument(
        "--allow-mixed-teacher",
        action="store_true",
        help="allow a deliberate blend of teacher configurations and report their counts",
    )
    args = parser.parse_args()

    rows = [row for pattern in args.lane for row in read(pattern)]
    by_id = {row["id"]: row for row in rows}
    sources = [row["sourceGameId"] for row in rows]
    if len(rows) != args.expected:
        raise SystemExit(f"expected {args.expected} targets, found {len(rows)}")
    if len(by_id) != len(rows) or len(set(sources)) != len(sources):
        raise SystemExit("duplicate target IDs or source games")
    if not args.allow_mixed_teacher and any(
        row["teacher"]["depth"] != args.teacher_depth
        or row["teacher"]["maxNodes"] != args.teacher_nodes
        for row in rows
    ):
        raise SystemExit("unexpected teacher configuration")
    rows.sort(key=lambda row: row["id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "targets": len(rows),
                "sourceGames": len(set(sources)),
                "turns": Counter(row["state"].split(".")[4] for row in rows),
                "partitions": Counter(row["partition"] for row in rows),
                "teacherConfigurations": Counter(
                    f"depth{row['teacher']['depth']}/nodes{row['teacher']['maxNodes']}"
                    for row in rows
                ),
                "output": str(args.output),
            },
            sort_keys=True,
            default=dict,
        )
    )


if __name__ == "__main__":
    main()
