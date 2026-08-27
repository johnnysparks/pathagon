#!/usr/bin/env python3
"""Merge validated local one-game records without overwriting cloud records."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def seed_from_path(path: Path) -> int | None:
    try:
        return int(path.stem.removeprefix("game-"))
    except ValueError:
        return None


def read_record(path: Path) -> tuple[dict[str, Any], dict[str, Any]] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    record = payload.get("record", payload)
    if not isinstance(record, dict) or not isinstance(record.get("moves"), list):
        return None
    return payload, record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging-games", type=Path, required=True)
    parser.add_argument("--target-games", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-end", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    staging = args.staging_games if args.staging_games.is_absolute() else REPO_ROOT / args.staging_games
    target = args.target_games if args.target_games.is_absolute() else REPO_ROOT / args.target_games
    report_path = args.report if args.report.is_absolute() else REPO_ROOT / args.report
    if args.seed_start < 0 or args.seed_end < args.seed_start:
        raise SystemExit("invalid seed range")
    target.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"copied": [], "skippedExisting": [], "skippedOutOfRange": [], "invalid": []}
    for source in sorted(staging.glob("game-*.json")):
        seed = seed_from_path(source)
        if seed is None or not args.seed_start <= seed <= args.seed_end:
            summary["skippedOutOfRange"].append(source.name)
            continue
        destination = target / source.name
        if destination.is_file():
            summary["skippedExisting"].append(seed)
            continue
        loaded = read_record(source)
        if loaded is None:
            summary["invalid"].append({"seed": seed, "source": source.name, "reason": "invalid record"})
            continue
        payload, record = loaded
        if int(record.get("seed", -1)) != seed:
            summary["invalid"].append({"seed": seed, "source": source.name, "reason": "record seed mismatch"})
            continue
        moves = record["moves"]
        q_covered = sum(bool(move.get("actionValues") and move.get("actionVisits")) for move in moves)
        if q_covered != len(moves):
            summary["invalid"].append({"seed": seed, "source": source.name, "reason": f"partial Q coverage {q_covered}/{len(moves)}"})
            continue
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
        summary["copied"].append({"seed": seed, "plies": len(moves), "qCovered": q_covered, "source": str(source)})
    summary["counts"] = {key: len(value) for key, value in summary.items() if isinstance(value, list)}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), **summary["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
