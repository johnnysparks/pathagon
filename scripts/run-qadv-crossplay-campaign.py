#!/usr/bin/env python3
"""Run a weighted, resumable QAdv cross-play campaign in 100-game batches.

The campaign intentionally keeps Pathfinder as the largest opponent slice
while retaining several different opponents for useful action coverage. Each
batch produces a standalone arena report, a JSONL archive candidate, and a
bounded manual-review report. Uploading the aggregate archive is a separate
explicit step.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ARENA = REPO_ROOT / "scripts/run-qadv-arena.py"
ANALYZER = REPO_ROOT / "scripts/analyze-selfplay-batch.py"
DEFAULT_OUTPUT = REPO_ROOT / "research/runs/gnn/benchmark-7x7/generated/qadv-crossplay-campaign-20260826"
DEFAULT_SCHEDULE = "pathfinder:1000,surveyor:500,gnn:400,cnn:300,lunatic:300"


def parse_schedule(value: str, batch_size: int) -> list[tuple[str, int]]:
    schedule: list[tuple[str, int]] = []
    for item in value.split(","):
        key, separator, count_text = item.strip().partition(":")
        if not separator or not key:
            raise SystemExit(f"invalid schedule item: {item!r}; expected opponent:games")
        try:
            count = int(count_text)
        except ValueError as error:
            raise SystemExit(f"invalid schedule count: {item!r}") from error
        if count <= 0 or count % batch_size:
            raise SystemExit(f"schedule count for {key} must be a positive multiple of batch size {batch_size}")
        schedule.append((key, count))
    if not schedule:
        raise SystemExit("schedule must not be empty")
    return schedule


def write_jsonl(path: Path, games: list[dict]) -> None:
    path.write_text("".join(json.dumps(game, sort_keys=True) + "\n" for game in games), encoding="utf-8")


def game_signature(game: dict) -> str:
    """Identify a trajectory independently of seed and archive metadata."""

    return json.dumps(
        [move.get("action") for move in game.get("moves", [])],
        sort_keys=True,
        separators=(",", ":"),
    )


def analyze_batch(batch_jsonl: Path, review_dir: Path, batch_number: int) -> dict:
    review_dir.mkdir(parents=True, exist_ok=True)
    report_path = review_dir / f"batch-{batch_number:04d}.json"
    text_path = review_dir / f"batch-{batch_number:04d}.txt"
    sample_path = review_dir / f"batch-{batch_number:04d}-sample.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            str(batch_jsonl),
            "--opening-plies",
            "6",
            "--sample-games",
            "20",
            "--sample-seed",
            str(20260826 + batch_number),
            "--report",
            str(report_path),
            "--text-report",
            str(text_path),
            "--sample-output",
            str(sample_path),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    return json.loads(report_path.read_text(encoding="utf-8"))


def run_batch(args: argparse.Namespace, batch_number: int, opponent: str, seed: int, report_path: Path) -> None:
    command = [
        sys.executable,
        str(ARENA),
        "--checkpoint",
        str(args.checkpoint),
        "--selector",
        "guided",
        "--qadv-top-k",
        str(args.qadv_top_k),
        "--qadv-reply-k",
        str(args.qadv_reply_k),
        "--temperature-moves",
        str(args.temperature_moves),
        "--policy-temperature",
        str(args.policy_temperature),
        "--opening-moves",
        str(args.opening_moves),
        "--opening-temperature",
        str(args.opening_temperature),
        "--opening-randomness",
        str(args.opening_randomness),
        "--pathfinder-temperature",
        str(args.pathfinder_temperature),
        "--opponents",
        opponent,
        "--games-per-match",
        str(args.batch_size),
        "--baseline-simulations",
        str(args.baseline_simulations),
        "--max-plies",
        str(args.max_plies),
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--out",
        str(report_path),
    ]
    if not args.quiet:
        command.append("--verbose-progress")
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": str(args.threads_per_batch),
        "MKL_NUM_THREADS": str(args.threads_per_batch),
    }
    print(f"[campaign] batch {batch_number:02d}: {args.batch_size} games vs {opponent}", flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=environment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=2026120000)
    parser.add_argument("--qadv-top-k", type=int, default=4)
    parser.add_argument("--qadv-reply-k", type=int, default=3)
    parser.add_argument("--temperature-moves", type=int, default=48)
    parser.add_argument("--policy-temperature", type=float, default=1.15)
    parser.add_argument("--opening-moves", type=int, default=16)
    parser.add_argument("--opening-temperature", type=float, default=1.8)
    parser.add_argument("--opening-randomness", type=float, default=0.30)
    parser.add_argument("--pathfinder-temperature", type=float, default=1.15)
    parser.add_argument("--baseline-simulations", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=196)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads-per-batch", type=int, default=1)
    parser.add_argument("--quiet", action="store_true", help="suppress per-game move digests")
    args = parser.parse_args()
    if args.batch_size < 2 or args.batch_size % 2:
        raise SystemExit("--batch-size must be an even number >= 2")
    if args.threads_per_batch < 1:
        raise SystemExit("--threads-per-batch must be positive")
    args.checkpoint = args.checkpoint.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")
    schedule = parse_schedule(args.schedule, args.batch_size)
    batches: list[tuple[int, str, int]] = []
    batch_number = 0
    for opponent, game_count in schedule:
        for offset in range(0, game_count, args.batch_size):
            batch_number += 1
            batches.append((batch_number, opponent, args.start_seed + (batch_number - 1) * 1_000))
    total_games = sum(game_count for _opponent, game_count in schedule)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch_dir = args.output_dir / "batches"
    review_dir = args.output_dir / "review-batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "campaign-manifest.json"
    manifest = {
        "schemaVersion": 1,
        "mode": "qadv-crossplay-campaign",
        "status": "running",
        "checkpoint": str(args.checkpoint),
        "batchSize": args.batch_size,
        "startSeed": args.start_seed,
        "totalGamesRequested": total_games,
        "schedule": [{"opponent": opponent, "games": count} for opponent, count in schedule],
        "selector": "guided",
        "qadvTopK": args.qadv_top_k,
        "qadvReplyK": args.qadv_reply_k,
        "temperatureMoves": args.temperature_moves,
        "policyTemperature": args.policy_temperature,
        "openingMoves": args.opening_moves,
        "openingTemperature": args.opening_temperature,
        "openingRandomness": args.opening_randomness,
        "pathfinderTemperature": args.pathfinder_temperature,
        "baselineSimulations": args.baseline_simulations,
        "maxPlies": args.max_plies,
        "batches": [],
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        immutable = ("checkpoint", "batchSize", "startSeed", "totalGamesRequested", "schedule")
        if any(existing.get(key) != manifest.get(key) for key in immutable):
            raise SystemExit(f"refusing to reuse incompatible campaign manifest: {manifest_path}")
        manifest = existing
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    recorded = {int(entry["batch"]) for entry in manifest.get("batches", [])}
    all_games: list[dict] = []
    for number, opponent, seed in batches:
        report_path = batch_dir / f"batch-{number:04d}.json"
        jsonl_path = batch_dir / f"batch-{number:04d}.jsonl"
        if number not in recorded:
            run_batch(args, number, opponent, seed, report_path)
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            write_jsonl(jsonl_path, payload.get("games", []))
            analysis = analyze_batch(jsonl_path, review_dir, number)
            entry = {
                "batch": number,
                "opponent": opponent,
                "seed": seed,
                "games": len(payload.get("games", [])),
                "report": str(report_path.relative_to(args.output_dir)),
                "archiveCandidate": str(jsonl_path.relative_to(args.output_dir)),
                "reviewReport": str((review_dir / f"batch-{number:04d}.json").relative_to(args.output_dir)),
                "exactDuplicates": int(analysis.get("exactDuplicateGames", 0)),
                "interestingSeeds": [item.get("seed") for item in analysis.get("interesting", [])[:5]],
                "suspiciousSeeds": [item.get("seed") for item in analysis.get("suspicious", [])[:5]],
            }
            manifest.setdefault("batches", []).append(entry)
            manifest["batches"].sort(key=lambda item: int(item["batch"]))
            manifest["completedGames"] = sum(int(item.get("games", 0)) for item in manifest["batches"])
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(
                f"[campaign] reviewed batch {number:02d}: "
                f"interesting={entry['interestingSeeds']} suspicious={entry['suspiciousSeeds']}",
                flush=True,
            )
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        all_games.extend(payload.get("games", []))

    aggregate = args.output_dir / "all-games.jsonl"
    unique_games: list[dict] = []
    duplicate_seeds: list[int] = []
    seen_signatures: set[str] = set()
    for game in all_games:
        signature = game_signature(game)
        if signature in seen_signatures:
            duplicate_seeds.append(int(game.get("seed", -1)))
            continue
        seen_signatures.add(signature)
        unique_games.append(game)
    write_jsonl(aggregate, unique_games)
    manifest["status"] = "complete"
    manifest["completedGames"] = len(all_games)
    manifest["uniqueGames"] = len(unique_games)
    manifest["duplicateGamesFiltered"] = len(duplicate_seeds)
    manifest["duplicateSeeds"] = duplicate_seeds
    manifest["aggregateArchiveCandidate"] = str(aggregate.relative_to(args.output_dir))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "games": len(all_games), "uniqueGames": len(unique_games), "duplicatesFiltered": len(duplicate_seeds), "aggregate": str(aggregate)}, sort_keys=True))


if __name__ == "__main__":
    main()
