#!/usr/bin/env python3
"""Audit diversity, coverage, and search-target completeness in 7x7 self-play."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Iterable


ROOT_Q_SOURCE = "mcts-root-q-v1"


def records_from_value(value: object) -> Iterable[dict]:
    if not isinstance(value, dict):
        return
    nested = value.get("record")
    if isinstance(nested, dict):
        yield from records_from_value(nested)
        return
    if isinstance(value.get("moves"), list):
        yield value
        return
    games = value.get("games")
    if isinstance(games, list):
        for game in games:
            yield from records_from_value(game)


def read_records(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        text = handle.read()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            yield from records_from_value(value)
        return
    yield from records_from_value(value)


def action_key(action: object) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def trajectory_key(record: dict) -> str:
    payload = "|".join(action_key(move.get("action")) for move in record.get("moves", []))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


def has_complete_q(record: dict) -> bool:
    moves = record.get("moves")
    if not isinstance(moves, list) or not moves:
        return False
    return all(
        isinstance(move, dict)
        and move.get("actionValueSource") == ROOT_Q_SOURCE
        and isinstance(move.get("actionValues"), list)
        and isinstance(move.get("actionVisits"), list)
        and len(move["actionValues"]) == len(move["actionVisits"]) > 0
        for move in moves
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--opening-plies", type=int, default=8)
    args = parser.parse_args()
    if args.opening_plies < 1:
        raise SystemExit("--opening-plies must be positive")

    records: list[dict] = []
    sources = Counter[str]()
    for path in sorted(
        path
        for pattern in ("*.json", "*.jsonl", "*.jsonl.gz")
        for path in args.root.rglob(pattern)
    ):
        for record in read_records(path):
            records.append(record)
            sources[str(path)] += 1

    trajectories = Counter(trajectory_key(record) for record in records)
    openings = Counter(
        "|".join(action_key(move.get("action")) for move in record.get("moves", [])[: args.opening_plies])
        for record in records
    )
    first_actions = Counter(
        action_key(record["moves"][0].get("action"))
        for record in records
        if record.get("moves")
    )
    plies = [len(record.get("moves", [])) for record in records]
    placement_positions = sum(
        1
        for record in records
        for move in record.get("moves", [])
        if (move.get("action") or {}).get("kind") == "place"
    )
    relocation_positions = sum(
        1
        for record in records
        for move in record.get("moves", [])
        if (move.get("action") or {}).get("kind") == "relocate"
    )
    q_complete = sum(has_complete_q(record) for record in records)
    winners = Counter(str(record.get("winner")) for record in records)
    reasons = Counter(str(record.get("reason")) for record in records)
    agents = Counter(
        json.dumps(record.get("agents", {}), sort_keys=True, separators=(",", ":"))
        for record in records
    )

    audit = {
        "schemaVersion": 1,
        "root": str(args.root),
        "openingPlies": args.opening_plies,
        "games": len(records),
        "positions": sum(plies),
        "uniqueTrajectories": len(trajectories),
        "duplicateGames": sum(count - 1 for count in trajectories.values() if count > 1),
        "duplicateTrajectoryGroups": sum(count > 1 for count in trajectories.values()),
        "uniqueOpenings": len(openings),
        "duplicateOpeningGames": sum(count - 1 for count in openings.values() if count > 1),
        "firstActionEntropyBits": entropy(first_actions),
        "openingEntropyBits": entropy(openings),
        "firstActionKinds": Counter(
            (move.get("action") or {}).get("kind")
            for record in records
            for move in record.get("moves", [])[:1]
        ),
        "placementPositions": placement_positions,
        "relocationPositions": relocation_positions,
        "placementFraction": placement_positions / max(1, placement_positions + relocation_positions),
        "qCompleteGames": q_complete,
        "qCoverage": q_complete / max(1, len(records)),
        "plies": {
            "min": min(plies) if plies else 0,
            "median": median(plies) if plies else 0,
            "mean": mean(plies) if plies else 0,
            "max": max(plies) if plies else 0,
        },
        "results": dict(winners),
        "terminationReasons": dict(reasons),
        "agentPairs": dict(agents),
        "sources": dict(sources),
    }
    encoded = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({key: audit[key] for key in ("games", "positions", "uniqueTrajectories", "duplicateGames", "uniqueOpenings", "qCoverage")}, sort_keys=True))


if __name__ == "__main__":
    main()
