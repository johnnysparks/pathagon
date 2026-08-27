#!/usr/bin/env python3
"""Run a resumable, Lambda-backed Q-Arbiter cross-play campaign.

Each invocation plays one color-balanced game against the requested opponent.
The coordinator writes a durable game file, review batch, and manifest after
every 100-game batch so a long campaign can be inspected or uploaded while it
is still running.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
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
ARCHIVER = REPO_ROOT / "scripts/archive-selfplay.ts"
DEFAULT_OUTPUT = REPO_ROOT / "research/runs/gnn/benchmark-7x7/generated/qadv-lambda-crossplay-20260826"
DEFAULT_SCHEDULE = "pathfinder:1000,surveyor:500,lunatic:500,coin-flip:500"
DEFAULT_JOB: dict[str, Any] = {
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
}


def parse_schedule(value: str, batch_size: int) -> list[tuple[str, int]]:
    schedule: list[tuple[str, int]] = []
    for item in value.split(","):
        opponent, separator, count_text = item.strip().partition(":")
        if not separator or not opponent:
            raise SystemExit(f"invalid schedule item {item!r}; expected opponent:games")
        try:
            count = int(count_text)
        except ValueError as error:
            raise SystemExit(f"invalid schedule count {item!r}") from error
        if count <= 0 or count % batch_size:
            raise SystemExit(f"schedule count for {opponent} must be a positive multiple of batch size {batch_size}")
        schedule.append((opponent, count))
    if not schedule:
        raise SystemExit("schedule must not be empty")
    return schedule


def game_signature(record: dict[str, Any]) -> str:
    return json.dumps(
        [move.get("action") for move in record.get("moves", [])],
        sort_keys=True,
        separators=(",", ":"),
    )


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
            payload = json.loads(destination.read_text(encoding="utf-8"))
            record = payload["record"]
            if isinstance(record, dict) and int(record["seed"]) == seed:
                return {"seed": seed, "opponent": job["opponent"], "status": "existing", "record": record, "duration": 0.0}
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    command = [
        "aws", "lambda", "invoke",
        "--function-name", function_name,
        "--invocation-type", "RequestResponse",
        "--cli-binary-format", "raw-in-base64-out",
        "--region", region,
        "--profile", profile,
        "--cli-connect-timeout", "30",
        "--cli-read-timeout", "900",
        "--no-cli-pager",
    ]
    started = time.perf_counter()
    last_error = "unknown Lambda invocation failure"
    for attempt in range(retry_attempts + 1):
        with tempfile.NamedTemporaryFile(prefix=f"lambda-crossplay-{seed}-", suffix=".json", delete=False) as temporary:
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
                record = response["record"]
                if int(record.get("seed", -1)) != seed:
                    last_error = f"Lambda returned seed {record.get('seed')} for requested seed {seed}"
                elif record.get("agents", {}).get("light") == record.get("agents", {}).get("dark"):
                    last_error = "Lambda returned a self-play record for a cross-play request"
                else:
                    destination.write_text(
                        json.dumps({"seed": seed, "opponent": job["opponent"], "qadvLight": job["qadv_light"], "record": record}, separators=(",", ":")) + "\n",
                        encoding="utf-8",
                    )
                    return {
                        "seed": seed,
                        "opponent": job["opponent"],
                        "status": "complete",
                        "record": record,
                        "duration": time.perf_counter() - started,
                        "attempt": attempt + 1,
                    }
        finally:
            response_path.unlink(missing_ok=True)
    return {"seed": seed, "opponent": job["opponent"], "status": "failed", "error": last_error, "duration": time.perf_counter() - started}


def analyze_batch(records: list[dict[str, Any]], output_dir: Path, batch_number: int) -> dict[str, Any]:
    batch_dir = output_dir / "batches"
    review_dir = output_dir / "review-batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    archive_path = batch_dir / f"batch-{batch_number:04d}.jsonl"
    archive_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    report_path = review_dir / f"batch-{batch_number:04d}.json"
    text_path = review_dir / f"batch-{batch_number:04d}.txt"
    sample_path = review_dir / f"batch-{batch_number:04d}-sample.jsonl"
    subprocess.run(
        [
            sys.executable, str(ANALYZER), str(archive_path),
            "--opening-plies", "6", "--sample-games", "20",
            "--sample-seed", str(20260826 + batch_number),
            "--report", str(report_path), "--text-report", str(text_path),
            "--sample-output", str(sample_path),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    return json.loads(report_path.read_text(encoding="utf-8"))


def upload_batch(archive_path: Path, site_url: str, run_id: str, profile: str) -> None:
    environment = {**os.environ}
    subprocess.run(
        [
            "npx", "tsx", str(ARCHIVER), "--file", str(archive_path),
            "--url", site_url, "--engine", "rust", "--mode", "cross-play",
            "--runId", run_id, "--profile", profile,
        ],
        check=True,
        cwd=REPO_ROOT,
        env=environment,
    )


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function-name", required=True)
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=2026220000)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--retry-attempts", type=int, default=2)
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--profile", default="pathagon")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget-usd", type=float, default=20.0)
    parser.add_argument("--memory-mb", type=int, default=2048)
    parser.add_argument("--max-seconds-per-game", type=float, default=150.0)
    parser.add_argument("--cost-safety-factor", type=float, default=1.10)
    parser.add_argument("--usd-per-gb-second", type=float, default=0.0000133334)
    parser.add_argument("--site-url", help="optionally upload each completed batch to Sites")
    parser.add_argument("--site-run-id", default="qadv-lambda-crossplay-20260826-clean")
    args = parser.parse_args()
    if args.batch_size < 2 or args.concurrency < 1 or args.retry_attempts < 0:
        raise SystemExit("batch-size, concurrency, and retries must be valid positive values")
    if args.start_seed < 0 or args.start_seed > 4_294_967_295:
        raise SystemExit("start seed must fit in an unsigned 32-bit integer")
    schedule = parse_schedule(args.schedule, args.batch_size)
    total_games = sum(count for _opponent, count in schedule)
    if args.start_seed + total_games - 1 > 4_294_967_295:
        raise SystemExit("scheduled seed range must fit in an unsigned 32-bit integer")

    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    game_dir = output_dir / "games"
    output_dir.mkdir(parents=True, exist_ok=True)
    game_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "campaign-manifest.json"
    immutable = {
        "functionName": args.function_name,
        "batchSize": args.batch_size,
        "startSeed": args.start_seed,
        "totalGamesRequested": total_games,
        "schedule": [{"opponent": opponent, "games": count} for opponent, count in schedule],
    }
    manifest = {
        "schemaVersion": 1,
        "mode": "qadv-lambda-crossplay",
        "status": "running",
        "createdAtUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "region": args.region,
        "profile": args.profile,
        **immutable,
        "concurrency": args.concurrency,
        "retryAttempts": args.retry_attempts,
        "budgetGuardrailUsd": args.budget_usd,
        "recipe": DEFAULT_JOB,
        "batches": [],
        "failedSeeds": [],
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in immutable.items()):
            raise SystemExit(f"refusing to reuse incompatible manifest: {manifest_path}")
        manifest = existing
    write_manifest(manifest_path, manifest)

    jobs: list[tuple[int, str, int]] = []
    seed = args.start_seed
    batch_number = 0
    for opponent, count in schedule:
        for offset in range(0, count, args.batch_size):
            batch_number += 1
            jobs.append((batch_number, opponent, seed + offset))
        seed += count

    recorded = {int(item["batch"]) for item in manifest.get("batches", [])}
    per_game_ceiling = (args.memory_mb / 1024.0) * args.max_seconds_per_game * args.usd_per_gb_second * args.cost_safety_factor
    projected_total = total_games * per_game_ceiling
    print(f"lambda cross-play: {total_games} games in {len(jobs)} batches, concurrency={args.concurrency}, conservative ceiling=${projected_total:.2f}", flush=True)
    if projected_total >= args.budget_usd:
        raise SystemExit(f"campaign ceiling ${projected_total:.2f} reaches the ${args.budget_usd:.2f} guardrail")

    durations: list[float] = []
    for number, opponent, first_seed in jobs:
        batch_dir = output_dir / "batches"
        archive_path = batch_dir / f"batch-{number:04d}.jsonl"
        if number in recorded and archive_path.is_file():
            print(f"[lambda cross-play] batch {number:02d} already recorded", flush=True)
            continue
        batch_jobs = [
            {**DEFAULT_JOB, "seed": first_seed + offset, "opponent": opponent, "qadv_light": (first_seed + offset) % 2 == 0}
            for offset in range(args.batch_size)
        ]
        print(f"[lambda cross-play] batch {number:02d}: {args.batch_size} games vs {opponent} (seeds {first_seed}-{first_seed + args.batch_size - 1})", flush=True)
        results: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(invoke_one, args.function_name, args.region, args.profile, job, game_dir, args.retry_attempts) for job in batch_jobs]
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                if result["status"] == "failed":
                    manifest.setdefault("failedSeeds", []).append({"seed": result["seed"], "opponent": opponent, "error": result.get("error", "unknown")})
                    print(f"[lambda cross-play] seed {result['seed']} failed: {result.get('error', 'unknown')}", file=sys.stderr, flush=True)
                else:
                    if result.get("duration", 0.0) > 0:
                        durations.append(float(result["duration"]))
                    print(f"[lambda cross-play] batch {number:02d}: {index}/{args.batch_size} complete", flush=True)
        failures = [item for item in results if item["status"] == "failed"]
        if failures:
            write_manifest(manifest_path, manifest)
            raise SystemExit(f"batch {number} has {len(failures)} failed Lambda invocations; rerun to resume")
        records = [item["record"] for item in sorted(results, key=lambda item: int(item["seed"]))]
        report = analyze_batch(records, output_dir, number)
        entry = {
            "batch": number,
            "opponent": opponent,
            "seedStart": first_seed,
            "seedEnd": first_seed + args.batch_size - 1,
            "games": len(records),
            "archive": str(archive_path.relative_to(output_dir)),
            "reviewReport": f"review-batches/batch-{number:04d}.json",
            "exactDuplicates": int(report.get("exactDuplicateGames", 0)),
            "interestingSeeds": [item.get("seed") for item in report.get("interesting", [])[:5]],
            "suspiciousSeeds": [item.get("seed") for item in report.get("suspicious", [])[:5]],
        }
        manifest.setdefault("batches", []).append(entry)
        manifest["batches"].sort(key=lambda item: int(item["batch"]))
        manifest["completedGames"] = sum(int(item.get("games", 0)) for item in manifest["batches"])
        manifest["meanLambdaSeconds"] = statistics.fmean(durations) if durations else 0.0
        write_manifest(manifest_path, manifest)
        print(f"[lambda cross-play] reviewed batch {number:02d}: duplicates={entry['exactDuplicates']} interesting={entry['interestingSeeds'][:3]} suspicious={entry['suspiciousSeeds'][:3]}", flush=True)
        if args.site_url:
            upload_batch(archive_path, args.site_url, f"{args.site_run_id}-batch-{number:04d}", args.profile)

    all_records: list[dict[str, Any]] = []
    for path in sorted(game_dir.glob("game-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            record = payload["record"]
            if isinstance(record, dict):
                all_records.append(record)
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            continue
    unique_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    duplicates: list[int] = []
    for record in sorted(all_records, key=lambda item: int(item.get("seed", -1))):
        signature = game_signature(record)
        if signature in seen:
            duplicates.append(int(record.get("seed", -1)))
        else:
            seen.add(signature)
            unique_records.append(record)
    aggregate = output_dir / "all-games.jsonl"
    aggregate.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in unique_records), encoding="utf-8")
    manifest["status"] = "complete" if len(unique_records) == total_games and not manifest.get("failedSeeds") else "partial"
    manifest["completedGames"] = len(all_records)
    manifest["uniqueGames"] = len(unique_records)
    manifest["duplicateGamesFiltered"] = len(duplicates)
    manifest["duplicateSeeds"] = duplicates
    manifest["aggregateArchive"] = str(aggregate.relative_to(output_dir))
    write_manifest(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "games": len(all_records), "uniqueGames": len(unique_records), "duplicatesFiltered": len(duplicates), "aggregate": str(aggregate)}, sort_keys=True))


if __name__ == "__main__":
    main()
