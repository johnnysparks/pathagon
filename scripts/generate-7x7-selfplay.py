#!/usr/bin/env python3
"""Generate isolated, provenance-stamped 7x7 neural self-play archives."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("research/runs/gnn/benchmark-7x7/generated/batch-20260824-neural-mix")
PLAYERS = {
    "scout": {
        "label": "GNN Scout",
        "checkpoint": Path("research/runs/gnn/benchmark-7x7/small-gnn-warmstart.pt"),
        "architecture": "gnn",
        "agent_id": "python-gnn-puct-scout-v0.1.0",
        "agent_name": "Python GNN PUCT Scout",
        "agent_engine": "python-gnn",
    },
    "learner": {
        "label": "GNN Learner",
        "checkpoint": Path("research/runs/gnn/benchmark-7x7/gnn-warmstart.pt"),
        "architecture": "gnn",
        "agent_id": "python-gnn-puct-learner-v0.1.0",
        "agent_name": "Python GNN PUCT Learner",
        "agent_engine": "python-gnn",
    },
    "cnn": {
        "label": "CNN Learner",
        "checkpoint": Path("research/runs/gnn/benchmark-7x7/cnn-warmstart.pt"),
        "architecture": "cnn",
        "agent_id": "python-cnn-puct-v0.1.0",
        "agent_name": "Python CNN PUCT",
        "agent_engine": "python-cnn",
    },
    "reval-gnn": {
        "label": "Re-evaluated GNN 30k",
        "checkpoint": Path("research/runs/gnn/benchmark-7x7/generated/batch-20260824-neural-reval-20260824/reval-gnn-30k.pt"),
        "architecture": "gnn",
        "agent_id": "python-gnn-puct-reval30k-v0.1.0",
        "agent_name": "Python GNN PUCT Re-evaluated 30k",
        "agent_engine": "python-gnn",
    },
    "reval-cnn": {
        "label": "Re-evaluated CNN 30k",
        "checkpoint": Path("research/runs/gnn/benchmark-7x7/generated/batch-20260824-neural-reval-20260824/reval-cnn-30k.pt"),
        "architecture": "cnn",
        "agent_id": "python-cnn-puct-reval30k-v0.1.0",
        "agent_name": "Python CNN PUCT Re-evaluated 30k",
        "agent_engine": "python-cnn",
    },
}


def parse_players(value: str) -> list[str]:
    players = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in players if item not in PLAYERS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown player(s): {', '.join(unknown)}; choose from {', '.join(PLAYERS)}")
    if not players:
        raise argparse.ArgumentTypeError("at least one player is required")
    if len(set(players)) != len(players):
        raise argparse.ArgumentTypeError("players must be listed once each")
    return players


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def inspect_archive(path: Path, expected_games: int, expected_seed: int, player: dict) -> dict:
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise RuntimeError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)

    if len(records) != expected_games:
        raise RuntimeError(f"{path}: expected {expected_games} games, found {len(records)}")
    expected_seeds = set(range(expected_seed, expected_seed + expected_games))
    actual_seeds = {record.get("seed") for record in records}
    if actual_seeds != expected_seeds:
        raise RuntimeError(f"{path}: seed range mismatch; expected {expected_seed}..{expected_seed + expected_games - 1}")

    agent_ids = {
        record.get("agents", {}).get("light")
        for record in records
    } | {
        record.get("agents", {}).get("dark")
        for record in records
    }
    if agent_ids != {player["agent_id"]}:
        raise RuntimeError(f"{path}: archive agent IDs do not match {player['agent_id']}: {sorted(agent_ids)}")

    return {
        "games": len(records),
        "positions": sum(len(record.get("moves", [])) for record in records),
        "results": dict(sorted(Counter(record.get("result") for record in records).items())),
        "reasons": dict(sorted(Counter(record.get("reason") for record in records).items())),
        "modelHash": records[0]["agentSpecifications"]["light"]["manifest"].get("modelHash"),
        "agentId": player["agent_id"],
    }


def run_player(args: argparse.Namespace, player_name: str, player_index: int, output_dir: Path) -> dict:
    player = PLAYERS[player_name]
    checkpoint = REPO_ROOT / player["checkpoint"]
    if not checkpoint.is_file():
        raise RuntimeError(f"missing checkpoint for {player_name}: {checkpoint}")

    seed = args.seed + player_index * args.games_per_player
    output = output_dir / f"{player_name}-{args.games_per_player}games-seed-{seed}.jsonl"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing archive: {output}")

    command = [
        sys.executable,
        "-m",
        "research.gnn.train",
        "alphazero",
        "--resume",
        str(checkpoint),
        "--architecture",
        player["architecture"],
        "--size",
        "7",
        "--reserve",
        "14",
        "--max-plies",
        str(args.max_plies),
        "--games",
        str(args.games_per_player),
        "--workers",
        str(args.workers),
        "--selfplay-device",
        args.selfplay_device,
        "--device",
        args.device,
        "--simulations",
        str(args.simulations),
        "--temperature-moves",
        str(args.temperature_moves),
        "--updates",
        "0",
        "--seed",
        str(seed),
        "--games-out",
        str(output),
        "--agent-id",
        player["agent_id"],
        "--agent-name",
        player["agent_name"],
        "--agent-engine",
        player["agent_engine"],
    ]

    print(f"\n[{player_name}] {args.games_per_player} games, seeds {seed}..{seed + args.games_per_player - 1}", flush=True)
    print(f"[{player_name}] {shlex.join(command)}", flush=True)
    with tempfile.TemporaryDirectory(prefix=f".{player_name}-", dir=output_dir) as temporary_dir:
        temporary_checkpoint = Path(temporary_dir) / "generation.pt"
        completed = subprocess.run(
            [*command, "--out", str(temporary_checkpoint)],
            cwd=REPO_ROOT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"{player_name} generation failed with exit code {completed.returncode}")

    stats = inspect_archive(output, args.games_per_player, seed, player)
    stats.update({
        "name": player_name,
        "label": player["label"],
        "checkpoint": relative_path(checkpoint),
        "output": relative_path(output),
        "seedStart": seed,
        "seedEnd": seed + args.games_per_player - 1,
        "configuration": {
            "simulations": args.simulations,
            "temperatureMoves": args.temperature_moves,
            "workers": args.workers,
            "maxPlies": args.max_plies,
        },
    })
    print(f"[{player_name}] complete: {stats['games']} games / {stats['positions']} positions", flush=True)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-per-player", type=int, default=1000)
    parser.add_argument("--players", type=parse_players, default=list(PLAYERS))
    parser.add_argument("--seed", type=int, default=2026082400, help="first game seed; each player gets a disjoint contiguous range")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--simulations", type=int, default=4)
    parser.add_argument("--temperature-moves", type=int, default=32)
    parser.add_argument("--max-plies", type=int, default=196)
    parser.add_argument("--selfplay-device", default="cpu")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.games_per_player < 1:
        raise SystemExit("--games-per-player must be positive")
    if args.seed < 0 or args.seed + len(args.players) * args.games_per_player > 4_294_967_296:
        raise SystemExit("seed ranges must fit in an unsigned 32-bit integer")

    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(f"refusing to reuse existing batch directory with manifest: {manifest_path}")

    manifest = {
        "schemaVersion": 1,
        "status": "running",
        "createdAtUtc": datetime.now(timezone.utc).isoformat(),
        "boardSize": 7,
        "reservePerPlayer": 14,
        "gamesPerPlayer": args.games_per_player,
        "playersRequested": args.players,
        "seedStart": args.seed,
        "seedEnd": args.seed + len(args.players) * args.games_per_player - 1,
        "generation": {
            "simulations": args.simulations,
            "temperatureMoves": args.temperature_moves,
            "workers": args.workers,
            "maxPlies": args.max_plies,
            "selfplayDevice": args.selfplay_device,
            "trainingUpdates": 0,
        },
        "players": [],
    }
    write_manifest(manifest_path, manifest)

    try:
        for player_index, player_name in enumerate(args.players):
            stats = run_player(args, player_name, player_index, output_dir)
            manifest["players"].append(stats)
            manifest["completedGames"] = sum(item["games"] for item in manifest["players"])
            manifest["completedPositions"] = sum(item["positions"] for item in manifest["players"])
            write_manifest(manifest_path, manifest)
    except Exception:
        manifest["status"] = "failed"
        write_manifest(manifest_path, manifest)
        raise

    manifest["status"] = "complete"
    manifest["totalGames"] = sum(item["games"] for item in manifest["players"])
    manifest["totalPositions"] = sum(item["positions"] for item in manifest["players"])
    write_manifest(manifest_path, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "outputDir": relative_path(output_dir),
        "totalGames": manifest["totalGames"],
        "totalPositions": manifest["totalPositions"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
