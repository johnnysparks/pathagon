#!/usr/bin/env python3
"""Assemble the balanced depth-8 labels from retained rows and a top-up shard."""

from __future__ import annotations

import argparse
import glob
import json
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
    parser.add_argument("--roots", required=True, type=Path)
    parser.add_argument("--original", required=True)
    parser.add_argument("--topup", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-turn", type=int, default=128)
    args = parser.parse_args()

    roots = read(str(args.roots))
    original = read(args.original)
    topup = read(args.topup)
    root_ids = {row["id"] for row in roots}
    topup_by_id = {row["id"]: row for row in topup}
    original_by_id = {row["id"]: row for row in original}
    if len(roots) != args.per_turn * 2:
        raise SystemExit("balanced root set has the wrong size")
    if len(topup_by_id) != len(topup) or len(original_by_id) != len(original):
        raise SystemExit("duplicate target IDs")
    if not set(topup_by_id) <= root_ids:
        raise SystemExit("top-up target is not in the balanced root set")
    rows = []
    for root in roots:
        row = topup_by_id.get(root["id"], original_by_id.get(root["id"]))
        if row is None:
            raise SystemExit(f"missing target for {root['id']}")
        if row["teacher"]["depth"] != 8 or row["teacher"]["maxNodes"] != 2_000_000:
            raise SystemExit("unexpected teacher configuration")
        rows.append(row)
    turns = {color: sum(root["state"].split(".")[4] == color for root in roots) for color in ("L", "D")}
    if turns != {"L": args.per_turn, "D": args.per_turn}:
        raise SystemExit(f"unexpected turn balance: {turns}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"targets": len(rows), "turns": turns, "topup": len(topup), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
