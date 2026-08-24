#!/usr/bin/env python3
"""Merge schema-v2 JSONL or league archives into a deduplicated replay file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def records_from_value(value: object) -> Iterable[dict]:
    if not isinstance(value, dict):
        return
    if isinstance(value.get("moves"), list):
        yield value
        return
    games = value.get("games")
    if isinstance(games, list):
        for game in games:
            if isinstance(game, dict) and isinstance(game.get("moves"), list):
                yield game


def read_records(path: Path) -> Iterable[dict]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        yield from records_from_value(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", type=int, help="Reject records for another board size")
    parser.add_argument("--reserve", type=int, help="Reject records for another reserve size")
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    unique: dict[str, dict] = {}
    for path in args.inputs:
        for record in read_records(path):
            board_size = int(record.get("boardSize", 7))
            reserve = int(record.get("reservePerPlayer", 2 * board_size))
            if args.size is not None and board_size != args.size:
                raise SystemExit(f"{path}: expected board size {args.size}, found {board_size}")
            if args.reserve is not None and reserve != args.reserve:
                raise SystemExit(f"{path}: expected reserve {args.reserve}, found {reserve}")
            canonical = json.dumps(record, sort_keys=True, separators=(",", ":"))
            unique[canonical] = record

    ordered = sorted(
        unique.values(),
        key=lambda record: (
            int(record.get("boardSize", 7)),
            int(record.get("reservePerPlayer", 0)),
            int(record.get("seed", 0)),
            json.dumps(record.get("agents", {}), sort_keys=True),
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.output), "games": len(ordered), "inputs": len(args.inputs)}, sort_keys=True))


if __name__ == "__main__":
    main()
