#!/usr/bin/env python3
"""Generate a deterministic, sharded schema-v2 archive from the Rust teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "research/20260901-strong-teacher-10k-games/workspace/generation"
TEACHER_ID = "rust-pathfinder-teacher-d5-b256-500k-v1"
OPPONENT_ID = "rust-pathfinder-v0.3.0"


@dataclass(frozen=True)
class Shard:
    index: int
    seed: int
    games: int


def split_shards(total: int, shard_count: int, first_seed: int) -> list[Shard]:
    if total < 1 or shard_count < 1:
        raise ValueError("total games and shard count must be positive")
    shard_count = min(shard_count, total)
    base, remainder = divmod(total, shard_count)
    result: list[Shard] = []
    next_seed = first_seed
    for index in range(shard_count):
        games = base + int(index < remainder)
        result.append(Shard(index, next_seed, games))
        next_seed += games
    return result


def load_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("moves"), list):
                raise ValueError(f"{path}:{line_number}: expected a schema-v2 game record")
            records.append(value)
    return records


def inspect_shard(
    path: Path,
    shard: Shard,
    max_plies: int,
    opening_plies: int,
    opponent_id: str,
    opponent_depth: int,
    opponent_nodes: int,
    opponent_beam: int,
) -> dict[str, Any]:
    records = load_records(path)
    expected_seeds = list(range(shard.seed, shard.seed + shard.games))
    actual_seeds = [int(record.get("seed", -1)) for record in records]
    if len(records) != shard.games or actual_seeds != expected_seeds:
        raise ValueError(
            f"{path}: expected {shard.games} ordered games/seeds "
            f"{shard.seed}..{shard.seed + shard.games - 1}, found {len(records)}"
        )
    for record in records:
        config = record.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"{path}: seed {record.get('seed')} is missing config")
        if config.get("boardSize") != 7 or config.get("reservePerPlayer") != 14:
            raise ValueError(f"{path}: seed {record.get('seed')} is not a 7x7/14 game")
        if config.get("maxPlies") != max_plies:
            raise ValueError(f"{path}: seed {record.get('seed')} has an unexpected max-plies")
        if record.get("agents") != {"light": TEACHER_ID, "dark": opponent_id} and record.get("agents") != {
            "light": opponent_id,
            "dark": TEACHER_ID,
        }:
            raise ValueError(f"{path}: seed {record.get('seed')} has unexpected agent IDs")
        moves = record["moves"]
        if record.get("plies") != len(moves):
            raise ValueError(f"{path}: seed {record.get('seed')} has an incorrect ply count")
        if len(moves) < opening_plies:
            raise ValueError(f"{path}: seed {record.get('seed')} ended inside the opening window")
        specifications = record.get("agentSpecifications")
        if not isinstance(specifications, dict):
            raise ValueError(f"{path}: seed {record.get('seed')} is missing agent specifications")
        for player in ("light", "dark"):
            specification = specifications.get(player)
            manifest = specification.get("manifest") if isinstance(specification, dict) else None
            if not isinstance(manifest, dict):
                raise ValueError(f"{path}: seed {record.get('seed')} has incomplete {player} provenance")
            expected = (
                {"depth": 5, "beam": 256, "nodeBudget": 500_000}
                if record["agents"][player] == TEACHER_ID
                else {"depth": opponent_depth, "beam": opponent_beam, "nodeBudget": opponent_nodes}
            )
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise ValueError(f"{path}: seed {record.get('seed')} has unexpected {player} provenance")
        for move in moves[:opening_plies]:
            if move.get("nodes") != 1 or move.get("completedDepth") != 0:
                raise ValueError(f"{path}: seed {record.get('seed')} opening move is not marked as seeded randomness")
    return {
        "index": shard.index,
        "seedStart": shard.seed,
        "seedEnd": shard.seed + shard.games - 1,
        "games": len(records),
        "positions": sum(len(record["moves"]) for record in records),
        "path": str(path.relative_to(REPO_ROOT)),
    }


def run_shard(args: argparse.Namespace, shard: Shard) -> dict[str, Any]:
    output = args.output_dir / f"teacher-{shard.index:02d}.jsonl"
    log = args.output_dir / f"teacher-{shard.index:02d}.log"
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {output}; use --resume for a completed shard")
    if args.resume and output.exists():
        try:
            return inspect_shard(
                output,
                shard,
                args.max_plies,
                args.opening_random_plies,
                args.opponent_id,
                args.opponent_depth,
                args.opponent_nodes,
                args.opponent_beam,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            raise RuntimeError(f"{output} exists but is not a valid completed shard; remove it explicitly before retrying")
    command = [
        str(args.binary),
        "--games", str(shard.games),
        "--seed", str(shard.seed),
        "--max-plies", str(args.max_plies),
        "--opening-random-plies", str(args.opening_random_plies),
        "--board-size", "7",
        "--reserve", "14",
        "--depth", str(args.opponent_depth),
        "--nodes", str(args.opponent_nodes),
        "--beam", str(args.opponent_beam),
        "--candidate-depth", "5",
        "--candidate-nodes", "500000",
        "--candidate-beam", "256",
        "--candidate-id", TEACHER_ID,
        "--opponent", args.opponent,
        "--no-tactical-root-filter",
        "--workers", "1",
        "--progress-every", str(max(1, shard.games // 20)),
        "--jsonl",
    ]
    print(f"starting shard {shard.index}: {shard.games} games, seeds {shard.seed}..{shard.seed + shard.games - 1}", flush=True)
    with output.open("w", encoding="utf-8") as output_handle, log.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(command, cwd=REPO_ROOT, stdout=output_handle, stderr=log_handle, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"teacher shard {shard.index} failed with exit code {completed.returncode}; see {log}")
    result = inspect_shard(
        output,
        shard,
        args.max_plies,
        args.opening_random_plies,
        args.opponent_id,
        args.opponent_depth,
        args.opponent_nodes,
        args.opponent_beam,
    )
    print(f"completed shard {shard.index}: {result['positions']} positions", flush=True)
    return result


def combine(args: argparse.Namespace, shards: list[Shard], shard_stats: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for shard in shards:
        records.extend(load_records(args.output_dir / f"teacher-{shard.index:02d}.jsonl"))
    records.sort(key=lambda record: int(record["seed"]))
    if [int(record["seed"]) for record in records] != list(range(args.seed, args.seed + args.games)):
        raise ValueError("combined archive does not contain the exact contiguous seed range")
    archive = args.output_dir / f"teacher-games-{args.games}.jsonl"
    with archive.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    sequences = {
        json.dumps([move["action"] for move in record["moves"]], separators=(",", ":"), sort_keys=True)
        for record in records
    }
    manifest = {
        "schemaVersion": 1,
        "status": "complete",
        "archive": str(archive.relative_to(REPO_ROOT)),
        "archiveSha256": digest,
        "games": len(records),
        "uniqueActionSequences": len(sequences),
        "positions": sum(len(record["moves"]) for record in records),
        "seedStart": args.seed,
        "seedEnd": args.seed + args.games - 1,
        "teacher": {
            "id": TEACHER_ID,
            "engine": "rust-bitboard",
            "depth": 5,
            "beam": 256,
            "nodeBudget": 500_000,
            "tacticalRootFilter": False,
        },
        "opponent": {
            "id": args.opponent_id,
            "kind": args.opponent,
            "depth": args.opponent_depth,
            "beam": args.opponent_beam,
            "nodeBudget": args.opponent_nodes,
        },
        "openingRandomPlies": args.opening_random_plies,
        "maxPlies": args.max_plies,
        "parallelShards": args.parallel,
        "shards": sorted(shard_stats, key=lambda item: item["index"]),
    }
    (args.output_dir / "generation-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--parallel", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=2026090100)
    parser.add_argument("--max-plies", type=int, default=20)
    parser.add_argument("--opening-random-plies", type=int, default=4)
    parser.add_argument("--opponent", default="deep-search")
    parser.add_argument("--opponent-id", default=OPPONENT_ID)
    parser.add_argument("--opponent-depth", type=int, default=5)
    parser.add_argument("--opponent-nodes", type=int, default=500_000)
    parser.add_argument("--opponent-beam", type=int, default=256)
    parser.add_argument("--binary", type=Path, default=REPO_ROOT / "pathagon/engine-rs/target/release/pathagon-selfplay")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.binary = args.binary if args.binary.is_absolute() else REPO_ROOT / args.binary
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    if not args.binary.is_file():
        raise SystemExit(f"missing release binary: {args.binary}; build pathagon-selfplay first")
    if args.max_plies <= args.opening_random_plies or args.opening_random_plies < 0:
        raise SystemExit("max plies must exceed the non-negative opening-random-plies value")
    if args.opponent_depth < 1 or args.opponent_nodes < 1 or args.opponent_beam < 1:
        raise SystemExit("opponent depth, nodes, and beam must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards = split_shards(args.games, args.shards, args.seed)
    stats: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.parallel, len(shards)))) as executor:
        futures = [executor.submit(run_shard, args, shard) for shard in shards]
        for future in as_completed(futures):
            stats.append(future.result())
    manifest = combine(args, shards, stats)
    print(json.dumps({"status": "complete", **{key: manifest[key] for key in ("games", "positions", "uniqueActionSequences", "archive")}}, sort_keys=True))


if __name__ == "__main__":
    main()
