#!/usr/bin/env python3
"""Run bounded one-game Rust Lambda fan-out with rolling review batches.

Each seed is an idempotent unit. The coordinator stores one response per seed,
emits a 20-game review slice, and refuses to schedule new work when a
conservative compute estimate reaches the configured budget guardrail.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYZER = REPO_ROOT / "scripts/analyze-selfplay-batch.py"
DEFAULT_OUTPUT = REPO_ROOT / "research/runs/gnn/benchmark-7x7/generated/batch-20260826-rust-lambda-20k-20260826"
DEFAULT_JOB = {
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
    "tactical_proof_horizon": None,
    "tactical_proof_nodes": 50_000,
}


def invoke_one(
    function_name: str,
    region: str,
    profile: str,
    job: dict[str, Any],
    game_dir: Path,
    retry_attempts: int,
) -> dict[str, Any]:
    seed = int(job["seed"])
    destination = game_dir / f"game-{seed}.json"
    if destination.is_file():
        try:
            response = json.loads(destination.read_text(encoding="utf-8"))
            return {"seed": seed, "status": "existing", "record": response["record"], "duration": 0.0}
        except (OSError, KeyError, json.JSONDecodeError):
            destination.unlink(missing_ok=True)

    command = [
        "aws",
        "lambda",
        "invoke",
        "--function-name",
        function_name,
        "--invocation-type",
        "RequestResponse",
        "--cli-binary-format",
        "raw-in-base64-out",
        "--region",
        region,
        "--profile",
        profile,
        "--cli-connect-timeout",
        "30",
        "--cli-read-timeout",
        "900",
        "--no-cli-pager",
    ]
    started = time.perf_counter()
    last_error = "unknown Lambda invocation failure"
    for attempt in range(retry_attempts + 1):
        with tempfile.NamedTemporaryFile(prefix=f"lambda-{seed}-", suffix=".json", delete=False) as temporary:
            response_path = Path(temporary.name)
        try:
            completed = subprocess.run(
                [*command, "--payload", json.dumps(job, separators=(",", ":")), str(response_path)],
                check=False,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                env={**os.environ, "AWS_MAX_ATTEMPTS": "1"},
            )
            try:
                response = json.loads(response_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                response = None
                last_error = f"invalid Lambda response: {error}"
            if completed.returncode != 0:
                last_error = (completed.stderr or completed.stdout or "aws lambda invoke failed").strip()
            elif isinstance(response, dict) and response.get("FunctionError"):
                last_error = str(response.get("errorMessage") or response.get("FunctionError"))
            elif isinstance(response, dict) and isinstance(response.get("record"), dict):
                destination.write_text(json.dumps(response, separators=(",", ":")) + "\n", encoding="utf-8")
                return {
                    "seed": seed,
                    "status": "complete",
                    "record": response["record"],
                    "duration": time.perf_counter() - started,
                    "attempt": attempt + 1,
                }
        finally:
            response_path.unlink(missing_ok=True)
    return {"seed": seed, "status": "failed", "error": last_error, "duration": time.perf_counter() - started}


def run_review(
    records: list[dict[str, Any]],
    batch_number: int,
    output_dir: Path,
    sample_size: int,
) -> dict[str, Any]:
    review_dir = output_dir / "review-batches"
    review_dir.mkdir(parents=True, exist_ok=True)
    batch_path = review_dir / f"batch-{batch_number:04d}.jsonl"
    batch_path.write_text("\n".join(json.dumps(record, separators=(",", ":")) for record in sorted(records, key=lambda item: item["seed"])) + "\n", encoding="utf-8")
    report_path = batch_path.with_suffix(".json")
    text_path = batch_path.with_suffix(".txt")
    sample_path = batch_path.with_name(batch_path.stem + "-sample.jsonl")
    subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            str(batch_path),
            "--opening-plies",
            "6",
            "--sample-games",
            str(min(sample_size, len(records))),
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


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def count_positions(game_dir: Path, expected_seeds: set[int] | None = None) -> int:
    total = 0
    for path in game_dir.glob("game-*.json"):
        if expected_seeds is not None:
            try:
                seed = int(path.stem.removeprefix("game-"))
            except ValueError:
                continue
            if seed not in expected_seeds:
                continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            total += len(payload.get("record", {}).get("moves", []))
        except (OSError, TypeError, AttributeError, json.JSONDecodeError):
            continue
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function-name", required=True)
    parser.add_argument("--games", type=int, default=20_000)
    parser.add_argument("--start-seed", type=int, default=2026100000)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--review-every", type=int, default=20)
    parser.add_argument("--review-sample-size", type=int, default=20)
    parser.add_argument("--retry-attempts", type=int, default=0)
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--profile", default="pathagon")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget-usd", type=float, default=90.0, help="stop before this conservative compute estimate")
    parser.add_argument("--memory-mb", type=int, default=2048)
    parser.add_argument("--max-seconds-per-game", type=float, default=150.0)
    parser.add_argument("--cost-safety-factor", type=float, default=1.10)
    parser.add_argument("--usd-per-gb-second", type=float, default=0.0000133334)
    args = parser.parse_args()
    if args.games < 1 or args.concurrency < 1 or args.review_every < 1 or args.memory_mb < 128 or args.cost_safety_factor < 1.0:
        raise SystemExit("games, concurrency, review-every, and memory must be positive")
    if args.start_seed < 0 or args.start_seed + args.games > 4_294_967_296:
        raise SystemExit("seed range must fit in an unsigned 32-bit integer")

    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    game_dir = output_dir / "games"
    output_dir.mkdir(parents=True, exist_ok=True)
    game_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "schemaVersion": 1,
        "status": "running",
        "createdAtUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "region": args.region,
        "profile": args.profile,
        "functionName": args.function_name,
        "gamesRequested": args.games,
        "seedStart": args.start_seed,
        "seedEnd": args.start_seed + args.games - 1,
        "concurrency": args.concurrency,
        "reviewEvery": args.review_every,
        "budgetGuardrailUsd": args.budget_usd,
        "costModel": {
            "memoryMb": args.memory_mb,
            "maxSecondsPerGame": args.max_seconds_per_game,
            "costSafetyFactor": args.cost_safety_factor,
            "usdPerGbSecond": args.usd_per_gb_second,
        },
        "recipe": DEFAULT_JOB,
        "completedGames": 0,
        "failedSeeds": [],
        "reviewBatches": [],
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("functionName") != args.function_name or existing.get("seedStart") != args.start_seed or existing.get("gamesRequested") != args.games:
            raise SystemExit(f"refusing to reuse incompatible manifest: {manifest_path}")
        manifest = existing
    write_manifest(manifest_path, manifest)

    expected_seeds = set(range(args.start_seed, args.start_seed + args.games))
    pending = [
        {**DEFAULT_JOB, "seed": seed}
        for seed in range(args.start_seed, args.start_seed + args.games)
        if not (game_dir / f"game-{seed}.json").is_file()
    ]
    total_completed = args.games - len(pending)
    total_positions = count_positions(game_dir, expected_seeds)
    existing_seeds = expected_seeds - {int(job["seed"]) for job in pending}
    # A validated local backup may have filled a seed that an earlier cloud
    # attempt recorded as failed. Reconcile that provenance before scheduling
    # so the final manifest describes the files actually in the corpus.
    manifest["failedSeeds"] = [
        item for item in manifest.get("failedSeeds", []) if int(item.get("seed", -1)) not in existing_seeds
    ]
    manifest["completedGames"] = total_completed
    manifest["completedPositions"] = total_positions
    write_manifest(manifest_path, manifest)
    durations: list[float] = []
    review_buffer: list[dict[str, Any]] = []
    batch_number = len(manifest.get("reviewBatches", []))
    prior_estimated_usd = float(manifest.get("estimatedComputeUsd") or 0.0)
    prior_completed = total_completed
    per_game_ceiling_usd = (args.memory_mb / 1024.0) * args.max_seconds_per_game * args.usd_per_gb_second
    estimated_total_usd = prior_estimated_usd or total_completed * per_game_ceiling_usd
    print(f"lambda fanout: {total_completed}/{args.games} already complete; {len(pending)} pending; conservative ceiling=${estimated_total_usd:.2f}", flush=True)
    if estimated_total_usd >= args.budget_usd:
        raise SystemExit("existing work already reaches the configured budget guardrail")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}
        pending_iter = iter(pending)

        def fill() -> None:
            nonlocal estimated_total_usd
            while len(futures) < args.concurrency:
                try:
                    job = next(pending_iter)
                except StopIteration:
                    return
                observed = statistics.fmean(durations) if durations else args.max_seconds_per_game
                projected = estimated_total_usd + (len(futures) + 1) * max(per_game_ceiling_usd, (args.memory_mb / 1024.0) * observed * args.cost_safety_factor * args.usd_per_gb_second)
                if projected >= args.budget_usd:
                    print(f"budget guardrail reached before seed {job['seed']}: projected=${projected:.2f}", flush=True)
                    return
                futures[executor.submit(invoke_one, args.function_name, args.region, args.profile, job, game_dir, args.retry_attempts)] = job

        fill()
        while futures:
            done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                job = futures.pop(future)
                result = future.result()
                if result["status"] in {"complete", "existing"}:
                    total_completed += 1 if result["status"] == "complete" else 0
                    # A resumed run may be repairing a prior transport timeout.
                    # Once the seed is present, remove any stale failure record
                    # so the final manifest reflects the corpus on disk.
                    manifest["failedSeeds"] = [
                        item for item in manifest.get("failedSeeds", [])
                        if item.get("seed") != result["seed"]
                    ]
                    if result.get("duration", 0.0) > 0:
                        durations.append(float(result["duration"]))
                    review_buffer.append(result["record"])
                    if len(review_buffer) >= args.review_every:
                        batch_number += 1
                        report = run_review(review_buffer[: args.review_every], batch_number, output_dir, args.review_sample_size)
                        manifest.setdefault("reviewBatches", []).append({
                            "batch": batch_number,
                            "games": report["games"],
                            "positions": report["positions"],
                            "exactDuplicateGames": report["exactDuplicateGames"],
                            "stateRepeatGames": report["stateRepeatGames"],
                            "threefoldStateGames": report["threefoldStateGames"],
                            "textReport": f"review-batches/batch-{batch_number:04d}.txt",
                        })
                        del review_buffer[: args.review_every]
                        print(
                            f"review batch {batch_number}: {report['games']} games, "
                            f"{report['positions']} positions, duplicates={report['exactDuplicateGames']}, "
                            f"stateRepeats={report['stateRepeatGames']}, "
                            f"qCoverage={report['qCoverage']:.1%}, "
                            f"interestingSeeds={[item.get('seed') for item in report['interesting'][:3]]}, "
                            f"suspiciousSeeds={[item.get('seed') for item in report['suspicious'][:3]]}",
                            flush=True,
                        )
                else:
                    manifest.setdefault("failedSeeds", [])
                    manifest["failedSeeds"] = [
                        item for item in manifest["failedSeeds"]
                        if item.get("seed") != result["seed"]
                    ]
                    manifest["failedSeeds"].append({"seed": result["seed"], "error": result.get("error", "unknown")})
                    print(f"seed {result['seed']} failed: {result.get('error', 'unknown')}", file=sys.stderr, flush=True)
                if result["status"] in {"complete", "existing"}:
                    total_positions += len(result.get("record", {}).get("moves", []))
                if prior_estimated_usd:
                    estimated_total_usd = prior_estimated_usd + (total_completed - prior_completed) * per_game_ceiling_usd
                else:
                    estimated_total_usd = total_completed * (args.memory_mb / 1024.0) * max(args.max_seconds_per_game, (statistics.fmean(durations) * args.cost_safety_factor if durations else args.max_seconds_per_game)) * args.usd_per_gb_second
                manifest["completedGames"] = total_completed
                manifest["completedPositions"] = total_positions
                manifest["estimatedComputeUsd"] = estimated_total_usd
                write_manifest(manifest_path, manifest)
            fill()

    if review_buffer:
        batch_number += 1
        report = run_review(review_buffer, batch_number, output_dir, args.review_sample_size)
        manifest.setdefault("reviewBatches", []).append({
            "batch": batch_number,
            "games": report["games"],
            "positions": report["positions"],
            "exactDuplicateGames": report["exactDuplicateGames"],
            "stateRepeatGames": report["stateRepeatGames"],
            "threefoldStateGames": report["threefoldStateGames"],
            "textReport": f"review-batches/batch-{batch_number:04d}.txt",
        })
    manifest["status"] = "complete" if not manifest.get("failedSeeds") and manifest.get("completedGames", 0) >= args.games else "partial"
    manifest["completedGames"] = sum(1 for seed in expected_seeds if (game_dir / f"game-{seed}.json").is_file())
    manifest["completedPositions"] = count_positions(game_dir, expected_seeds)
    manifest["averageSecondsPerGame"] = statistics.fmean(durations) if durations else 0.0
    write_manifest(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "completedGames": manifest["completedGames"], "failedSeeds": len(manifest.get("failedSeeds", [])), "estimatedComputeUsd": manifest.get("estimatedComputeUsd", 0.0)}, sort_keys=True))


if __name__ == "__main__":
    main()
