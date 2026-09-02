#!/usr/bin/env python3
"""Build a deterministic, duplicate-free mixed teacher corpus from source pools."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("moves"), list):
                raise ValueError(f"{path}:{line_number}: expected a game record")
            records.append(record)
    return records


def sequence_key(record: dict[str, Any]) -> str:
    return json.dumps(
        [move["action"] for move in record["moves"]],
        separators=(",", ":"),
        sort_keys=True,
    )


def select_unique(
    records: list[dict[str, Any]],
    count: int,
    used_sequences: set[str],
    used_seeds: set[int],
) -> tuple[list[dict[str, Any]], int]:
    selected: list[dict[str, Any]] = []
    skipped_duplicates = 0
    for record in records:
        seed = int(record.get("seed", -1))
        key = sequence_key(record)
        if seed in used_seeds or key in used_sequences:
            skipped_duplicates += 1
            continue
        selected.append(record)
        used_seeds.add(seed)
        used_sequences.add(key)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise SystemExit(
            f"source pool contains only {len(selected)} usable unique games; {count} required"
        )
    return selected, skipped_duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strong-input", type=Path, action="append", required=True)
    parser.add_argument("--weak-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=Path, required=True)
    parser.add_argument("--strong-games", type=int, default=8_000)
    parser.add_argument("--weak-games", type=int, default=2_000)
    args = parser.parse_args()
    args.strong_input = [
        path if path.is_absolute() else REPO_ROOT / path for path in args.strong_input
    ]
    if not args.weak_input.is_absolute():
        args.weak_input = REPO_ROOT / args.weak_input
    for name in ("output", "manifest", "expected_seeds"):
        path = getattr(args, name)
        if not path.is_absolute():
            setattr(args, name, REPO_ROOT / path)
    if args.strong_games < 1 or args.weak_games < 1:
        raise SystemExit("strong-games and weak-games must be positive")

    strong_records = sorted(
        [record for path in args.strong_input for record in read_records(path)],
        key=lambda record: int(record["seed"]),
    )
    weak_records = sorted(read_records(args.weak_input), key=lambda record: int(record["seed"]))
    used_sequences: set[str] = set()
    used_seeds: set[int] = set()
    strong, strong_skipped = select_unique(
        strong_records, args.strong_games, used_sequences, used_seeds
    )
    weak, weak_skipped = select_unique(weak_records, args.weak_games, used_sequences, used_seeds)
    records = strong + weak
    if len({int(record["seed"]) for record in records}) != len(records):
        raise SystemExit("duplicate seeds remain after selection")
    if len({sequence_key(record) for record in records}) != len(records):
        raise SystemExit("duplicate full games remain after selection")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
    seeds = [int(record["seed"]) for record in records]
    args.expected_seeds.write_text(json.dumps(seeds, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schemaVersion": 1,
        "status": "complete",
        "archive": str(args.output.relative_to(REPO_ROOT)),
        "archiveSha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "games": len(records),
        "uniqueFullGames": len(used_sequences),
        "seedCount": len(seeds),
        "seedMin": min(seeds),
        "seedMax": max(seeds),
        "sources": {
            "strong": {
                "archives": [str(path.relative_to(REPO_ROOT)) for path in args.strong_input],
                "archiveSha256": {
                    str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in args.strong_input
                },
                "selectedGames": len(strong),
                "skippedDuplicates": strong_skipped,
                "opponentProfile": {"depth": 5, "beam": 256, "nodeBudget": 500_000},
            },
            "weak": {
                "archive": str(args.weak_input.relative_to(REPO_ROOT)),
                "archiveSha256": hashlib.sha256(args.weak_input.read_bytes()).hexdigest(),
                "selectedGames": len(weak),
                "skippedDuplicates": weak_skipped,
                "opponentProfile": {"depth": 3, "beam": 64, "nodeBudget": 12_000},
            },
        },
        "duplicatePolicy": "global action-sequence uniqueness across both source pools",
        "expectedSeeds": str(args.expected_seeds.relative_to(REPO_ROOT)),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
