#!/usr/bin/env python3
"""Build a deduplicated, seed-grouped 7x7 train/held-out benchmark corpus."""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


EXCLUDED_ENGINES = {"typescript-live-cross-play"}
ROOT_Q_SOURCE = "mcts-root-q-v1"


def records_from_value(value: object) -> Iterable[dict]:
    if not isinstance(value, dict):
        return
    nested_record = value.get("record")
    if isinstance(nested_record, dict):
        if value.get("engine") in EXCLUDED_ENGINES:
            return
        yield from records_from_value(nested_record)
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
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            text = handle.read()
    else:
        text = path.read_text(encoding="utf-8")
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


def record_config(record: dict, path: Path) -> tuple[int, int]:
    config = record.get("config") or {}
    size_value = record.get("boardSize", config.get("boardSize"))
    reserve_value = record.get("reservePerPlayer", config.get("reservePerPlayer"))
    if size_value is None and "5x5" in path.name:
        size_value = 5
    if size_value is None and "6x6" in path.name:
        size_value = 6
    # The earliest 7x7 self-play exports predate board-size metadata. The
    # filename exceptions above keep the known 5x5 archive out; remaining
    # legacy files in this benchmark root are historical 7x7 exports.
    if size_value is None:
        size_value = 7
    size = int(size_value)
    reserve = int(reserve_value if reserve_value is not None else 2 * size)
    return size, reserve


def replay_signature(record: dict, size: int, reserve: int, include_agents: bool = True) -> str:
    payload = {
        "size": size,
        "reserve": reserve,
        "winner": record.get("winner"),
        "moves": [move.get("action") for move in record["moves"]],
    }
    if include_agents:
        payload["agents"] = record.get("agents")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def has_complete_action_values(record: dict) -> bool:
    moves = record.get("moves")
    if not isinstance(moves, list) or not moves:
        return False
    return all(
        isinstance(move, dict)
        and isinstance(move.get("actionValues"), list)
        and isinstance(move.get("actionVisits"), list)
        and move.get("actionValueSource") == ROOT_Q_SOURCE
        and len(move["actionValues"]) == len(move["actionVisits"])
        and len(move["actionValues"]) > 0
        for move in moves
    )


def group_key(record: dict, signature: str, opening_plies: int = 0) -> str:
    if opening_plies > 0:
        opening = [move.get("action") for move in record["moves"][:opening_plies]]
        encoded = json.dumps(opening, sort_keys=True, separators=(",", ":"))
        return f"opening:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"
    seed = record.get("seed")
    return f"seed:{seed}" if seed is not None else f"record:{signature}"


def choose_split(groups: dict[str, list[tuple[str, dict]]], heldout_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    heldout: set[str] = set()
    for key in sorted(groups):
        digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
        score = int.from_bytes(digest[:8], "big") / float(1 << 64)
        if score < heldout_fraction:
            heldout.add(key)
    train = set(groups) - heldout
    if not train or not heldout:
        raise ValueError("split produced an empty train or held-out partition")
    return train, heldout


def write_records(path: Path, records: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("research/runs/gnn"))
    parser.add_argument("--output", type=Path, default=Path("research/runs/gnn/benchmark-7x7"))
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument(
        "--opening-plies",
        type=int,
        default=0,
        help="group train/held-out assignment by the first N actions so opening patterns never cross the split",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="directory name to exclude while discovering source archives; may be repeated",
    )
    parser.add_argument(
        "--exclude-path",
        action="append",
        default=[],
        help="relative glob pattern to exclude while discovering source archives; may be repeated",
    )
    parser.add_argument(
        "--require-action-values",
        action="store_true",
        help="keep only complete games whose every move has mcts-root-q-v1 action-value targets",
    )
    parser.add_argument(
        "--dedupe-by-actions",
        action="store_true",
        help="deduplicate identical board trajectories even when the generating agent IDs differ",
    )
    args = parser.parse_args()
    if not 0.0 < args.heldout_fraction < 1.0:
        raise SystemExit("--heldout-fraction must be between 0 and 1")
    if args.opening_plies < 0:
        raise SystemExit("--opening-plies cannot be negative")

    unique: dict[str, tuple[str, dict, int, int]] = {}
    raw_counts: Counter[str] = Counter()
    raw_positions = 0
    skipped = []
    skipped_missing_action_values = 0
    excluded_dirs = set(args.exclude_dir)
    excluded_paths = tuple(args.exclude_path)

    def is_excluded(path: Path) -> bool:
        relative = path.relative_to(args.root).as_posix()
        return any(part in excluded_dirs for part in path.relative_to(args.root).parts) or any(
            fnmatch.fnmatch(relative, pattern) for pattern in excluded_paths
        )

    input_paths = sorted(
        {
            path
            for pattern in ("*.jsonl", "*.jsonl.gz", "*.json")
            for path in args.root.rglob(pattern)
            if not is_excluded(path)
        }
    )
    for path in input_paths:
        for record in read_records(path):
            size, reserve = record_config(record, path)
            if size != 7 or reserve != 14:
                skipped.append({"path": str(path), "size": size, "reserve": reserve})
                continue
            if args.require_action_values and not has_complete_action_values(record):
                skipped_missing_action_values += 1
                continue
            raw_counts[str(path)] += 1
            raw_positions += len(record["moves"])
            signature = replay_signature(record, size, reserve, include_agents=not args.dedupe_by_actions)
            unique.setdefault(signature, (str(path), record, size, reserve))

    groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for signature, (source, record, _size, _reserve) in unique.items():
        groups[group_key(record, signature, args.opening_plies)].append((signature, record))
    train_groups, heldout_groups = choose_split(groups, args.heldout_fraction, args.seed)

    ordered = sorted(unique.items(), key=lambda item: (int(item[1][1].get("seed") or 0), item[0]))
    train_records = [record for signature, (_source, record, _size, _reserve) in ordered if group_key(record, signature, args.opening_plies) in train_groups]
    heldout_records = [record for signature, (_source, record, _size, _reserve) in ordered if group_key(record, signature, args.opening_plies) in heldout_groups]
    args.output.mkdir(parents=True, exist_ok=True)
    write_records(args.output / "all.jsonl", (record for _signature, (_source, record, _size, _reserve) in ordered))
    write_records(args.output / "train.jsonl", train_records)
    write_records(args.output / "heldout.jsonl", heldout_records)

    def position_count(records: Iterable[dict]) -> int:
        return sum(len(record["moves"]) for record in records)

    manifest = {
        "schemaVersion": 1,
        "boardSize": 7,
        "reservePerPlayer": 14,
        "splitSeed": args.seed,
        "heldoutFraction": args.heldout_fraction,
        "openingHoldoutPlies": args.opening_plies,
        "rawGames": sum(raw_counts.values()),
        "rawPositions": raw_positions,
        "uniqueGames": len(ordered),
        "uniquePositions": position_count(record for _signature, (_source, record, _size, _reserve) in ordered),
        "trainGames": len(train_records),
        "trainPositions": position_count(train_records),
        "heldoutGames": len(heldout_records),
        "heldoutPositions": position_count(heldout_records),
        "duplicateGamesRemoved": sum(raw_counts.values()) - len(ordered),
        "requireActionValues": args.require_action_values,
        "dedupeByActions": args.dedupe_by_actions,
        "skippedMissingActionValues": skipped_missing_action_values,
        "trainSeeds": sorted({record.get("seed") for record in train_records if record.get("seed") is not None}),
        "heldoutSeeds": sorted({record.get("seed") for record in heldout_records if record.get("seed") is not None}),
        "trainOpeningGroups": len(train_groups),
        "heldoutOpeningGroups": len(heldout_groups),
        "openingGroupOverlap": len(train_groups & heldout_groups),
        "sources": [{"path": path, "rawGames": count} for path, count in sorted(raw_counts.items())],
        "skippedNon7x7": skipped,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("rawGames", "rawPositions", "uniqueGames", "uniquePositions", "trainGames", "trainPositions", "heldoutGames", "heldoutPositions", "duplicateGamesRemoved")}, sort_keys=True))


if __name__ == "__main__":
    main()
