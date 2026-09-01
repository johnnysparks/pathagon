#!/usr/bin/env python3
"""Audit Rust-emitted transition-policy rows before learner training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def identity(action: dict) -> tuple:
    value = action["action"]
    return value.get("kind"), value.get("from"), value.get("to")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--expected", type=int, default=0)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.targets.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.expected and len(rows) != args.expected:
        raise SystemExit(f"expected {args.expected} rows, found {len(rows)}")
    if not rows:
        raise SystemExit("target set is empty")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate root IDs")
    source_partitions: dict[str, set[str]] = {}
    for row in rows:
        required = {
            "schemaVersion", "id", "sourceGameId", "sourcePly", "phase",
            "partition", "state", "teacher", "teacherAction", "teacherScore",
            "teacherNodes", "completedDepth", "exhausted", "actions",
        }
        missing = required - row.keys()
        if missing:
            raise SystemExit(f"{row.get('id', '<unknown>')} missing {sorted(missing)}")
        teacher = row["teacher"]
        if (
            row["schemaVersion"] != 1
            or teacher["depth"] != 5
            or teacher["maxNodes"] != 500_000
            or teacher["beamWidth"] != 256
        ):
            raise SystemExit(f"unexpected schema or teacher configuration at {row['id']}")
        if row["partition"] not in {"train", "heldout"}:
            raise SystemExit(f"invalid partition at {row['id']}")
        source_partitions.setdefault(row["sourceGameId"], set()).add(row["partition"])
        actions = row["actions"]
        if not actions or len({identity(action) for action in actions}) != len(actions):
            raise SystemExit(f"empty or duplicate legal action list at {row['id']}")
        matches = [action for action in actions if identity(action) == identity({"action": row["teacherAction"]})]
        if len(matches) != 1:
            raise SystemExit(f"teacher action is not exactly one legal row at {row['id']}")
        if any(
            not isinstance(action.get("features"), list)
            or len(action["features"]) != 6
            or not isinstance(action.get("safe"), bool)
            or not isinstance(action.get("immediateWin"), bool)
            for action in actions
        ):
            raise SystemExit(f"malformed action features at {row['id']}")
    overlap = [source for source, partitions in source_partitions.items() if len(partitions) > 1]
    if overlap:
        raise SystemExit(f"source games cross train/heldout boundary: {overlap[:3]}")
    partitions = Counter(row["partition"] for row in rows)
    phases = Counter(row["phase"] for row in rows)
    if not {"opening", "placement", "movement"}.issubset(phases):
        raise SystemExit(f"target set lacks phase coverage: {dict(phases)}")
    turns = Counter(row["state"].split(".")[4] for row in rows)
    if set(turns) != {"L", "D"}:
        raise SystemExit(f"target set must contain both turns, found {dict(turns)}")
    print(json.dumps({
        "rows": len(rows),
        "sourceGames": len(source_partitions),
        "partitions": partitions,
        "phases": phases,
        "turns": turns,
        "exhausted": sum(bool(row["exhausted"]) for row in rows),
        "teacherNodes": sum(int(row["teacherNodes"]) for row in rows),
    }, sort_keys=True, default=dict))


if __name__ == "__main__":
    main()
