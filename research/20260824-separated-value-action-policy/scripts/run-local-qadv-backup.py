#!/usr/bin/env python3
"""Stage native Rust QAdv games while a remote self-play lane is running.

The runner deliberately writes to a separate staging directory.  A later
merge can keep an AWS record when the same seed arrives from both lanes.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = REPO_ROOT / "pathagon/engine-rs/target/release/pathagon-selfplay"
DEFAULT_GAMES_DIR = REPO_ROOT / "research/runs/gnn/benchmark-7x7/generated/batch-20260826-rust-lambda-20k-20260826/games"
DEFAULT_STAGING_DIR = REPO_ROOT / "research/runs/gnn/benchmark-7x7/generated/batch-20260826-rust-lambda-20k-20260826/local-backup-20260826"


def game_seed(path: Path) -> int | None:
    try:
        return int(path.stem.removeprefix("game-"))
    except ValueError:
        return None


def run_one(binary: Path, model: Path, seed: int) -> dict[str, Any]:
    command = [
        str(binary),
        "--qadv-onnx", str(model),
        "--opponent", "neural",
        "--games", "1",
        "--seed", str(seed),
        "--max-plies", "196",
        "--opening-random-plies", "0",
        "--simulations", "128",
        "--temperature-moves", "48",
        "--policy-temperature", "1.15",
        "--opening-moves", "16",
        "--opening-temperature", "1.8",
        "--opening-randomness", "0.30",
        "--pathfinder-guidance", "0.45",
        "--placement-guidance", "0.30",
        "--pathfinder-temperature", "1.15",
        "--pathfinder-depth", "2",
        "--pathfinder-beam", "8",
        "--pathfinder-nodes", "512",
        "--qadv-weight", "1.0",
        "--workers", "1",
        "--progress-every", "1",
        "--jsonl",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return {"seed": seed, "status": "failed", "error": (completed.stderr or completed.stdout or "native runner failed").strip()}
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        return {"seed": seed, "status": "failed", "error": f"expected one JSONL record, got {len(lines)} lines"}
    try:
        record = json.loads(lines[0])
    except json.JSONDecodeError as error:
        return {"seed": seed, "status": "failed", "error": f"invalid native JSONL: {error}"}
    return {
        "seed": seed,
        "status": "complete",
        "record": record,
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--games-dir", type=Path, default=DEFAULT_GAMES_DIR)
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING_DIR)
    parser.add_argument("--seed-start", type=int, default=2026200000)
    parser.add_argument("--seed-end", type=int, default=2026217499)
    parser.add_argument("--max-games", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.seed_start < 0 or args.seed_end < args.seed_start or args.max_games < 1 or args.workers < 1:
        raise SystemExit("invalid seed range, max-games, or workers")
    binary = args.binary.resolve()
    model = args.model.resolve()
    games_dir = args.games_dir.resolve()
    staging_dir = args.staging_dir.resolve()
    if not binary.is_file() or not model.is_file():
        raise SystemExit("native binary and model must exist")

    present = {seed for path in games_dir.glob("game-*.json") if (seed := game_seed(path)) is not None}
    staged_games_dir = staging_dir / "games"
    staged_games_dir.mkdir(parents=True, exist_ok=True)
    staged = {seed for path in staged_games_dir.glob("game-*.json") if (seed := game_seed(path)) is not None}
    candidates = [
        seed
        for seed in range(args.seed_start, args.seed_end + 1)
        if seed not in present and seed not in staged
    ][: args.max_games]
    if not candidates:
        raise SystemExit("no missing target seeds available for local staging")

    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    print(json.dumps({"selected": len(candidates), "workers": args.workers, "modelSha256": model_hash}, sort_keys=True), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, binary, model, seed): seed for seed in candidates}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            if result["status"] == "complete":
                path = staged_games_dir / f"game-{result['seed']}.json"
                path.write_text(json.dumps({"seed": result["seed"], "record": result["record"]}, separators=(",", ":")) + "\n", encoding="utf-8")
                print(f"local backup: seed={result['seed']} complete seconds={result['seconds']:.1f}", flush=True)
            else:
                print(f"local backup: seed={result['seed']} failed: {result['error']}", flush=True)

    results.sort(key=lambda item: item["seed"])
    manifest = {
        "schemaVersion": 1,
        "status": "complete" if all(item["status"] == "complete" for item in results) else "partial",
        "engine": "rust-native",
        "modelSha256": model_hash,
        "seedStart": args.seed_start,
        "seedEnd": args.seed_end,
        "selectedSeeds": candidates,
        "completedSeeds": [item["seed"] for item in results if item["status"] == "complete"],
        "failed": [item for item in results if item["status"] != "complete"],
        "workers": args.workers,
        "elapsedSeconds": time.perf_counter() - started,
        "recipe": {
            "max_plies": 196,
            "simulations": 128,
            "temperature_moves": 48,
            "policy_temperature": 1.15,
            "opening_moves": 16,
            "opening_temperature": 1.8,
            "opening_randomness": 0.30,
            "pathfinder_guidance": 0.45,
            "placement_guidance": 0.30,
            "pathfinder_temperature": 1.15,
            "pathfinder_depth": 2,
            "pathfinder_beam": 8,
            "pathfinder_nodes": 512,
            "qadv_weight": 1.0,
            "tactical_simulations": 512,
            "tactical_capture_threshold": 2,
        },
    }
    (staging_dir / "local-backup-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "completed": len(manifest["completedSeeds"]), "failed": len(manifest["failed"]), "staging": str(staging_dir)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
