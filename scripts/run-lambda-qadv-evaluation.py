#!/usr/bin/env python3
"""Fan out held-out QAdv replay evaluation to bounded AWS Lambda chunks.

The Lambda receives replay records directly, so this coordinator needs no
staging bucket. Payloads are kept below the invoke limit, results are written
per chunk for resumability, and raw metric sums are aggregated only after all
chunks complete.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAMES_DIR = REPO_ROOT / "research/runs/gnn/benchmark-7x7/generated/batch-20260826-rust-lambda-20k-20260826/games"
DEFAULT_OUTPUT = REPO_ROOT / "research/runs/gnn/benchmark-7x7/generated/batch-20260826-rust-lambda-20k-20260826/qadv-lambda-evaluation-heldout-complete"
RAW_KEYS = (
    "positions",
    "visitedActions",
    "visitedPairs",
    "selectedActionIsQMax",
    "selectedActionQRank",
    "selectedActionQPercentile",
    "qSpread",
    "qMse",
    "qMae",
    "qWeight",
    "predictedPairwiseAgreement",
    "predictedPairwisePairs",
    "predictedSelectedActionIsTargetQMax",
    "predictedSelectedActionTargetQRank",
)


def heldout_seeds(seed_start: int, seed_end: int, fraction: float, split_seed: int) -> set[int]:
    seeds = list(range(seed_start, seed_end + 1))
    selected = {
        seed
        for seed in seeds
        if int.from_bytes(hashlib.sha256(f"{split_seed}:{seed}".encode()).digest()[:8], "big") / float(1 << 64)
        < fraction
    }
    if fraction and not selected and len(seeds) > 1:
        selected = {seeds[-1]}
    if fraction and len(selected) == len(seeds) and len(seeds) > 1:
        selected.remove(seeds[-1])
    return selected


def iter_records(games_dir: Path, selected_seeds: set[int]) -> Iterator[dict[str, Any]]:
    for path in sorted(games_dir.glob("game-*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            record = payload.get("record", payload) if isinstance(payload, dict) else payload
            seed = int(record["seed"])
            if seed in selected_seeds:
                if not isinstance(record.get("moves"), list):
                    raise ValueError("record has no moves list")
                yield record
        except (OSError, TypeError, KeyError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot load held-out record {path}: {error}") from error


def iter_chunks(records: Iterable[dict[str, Any]], max_payload_bytes: int) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    current: list[dict[str, Any]] = []
    chunk_number = 0
    for record in records:
        candidate = current + [record]
        chunk_id = f"heldout-{chunk_number + 1:04d}"
        payload_size = len(json.dumps({"chunkId": chunk_id, "games": candidate}, separators=(",", ":")).encode())
        if current and payload_size > max_payload_bytes:
            chunk_number += 1
            yield f"heldout-{chunk_number:04d}", current
            current = [record]
            single_size = len(json.dumps({"chunkId": f"heldout-{chunk_number + 1:04d}", "games": current}, separators=(",", ":")).encode())
            if single_size > max_payload_bytes:
                raise ValueError(f"single replay record exceeds payload limit ({single_size} bytes)")
        else:
            current = candidate
    if current:
        chunk_number += 1
        yield f"heldout-{chunk_number:04d}", current


def zero_raw() -> dict[str, float]:
    return {key: 0.0 for key in RAW_KEYS}


def merge_raw(destination: dict[str, float], source: dict[str, Any]) -> None:
    for key in RAW_KEYS:
        destination[key] += float(source.get(key, 0.0))


def finalize(raw: dict[str, float]) -> dict[str, Any]:
    positions = raw["positions"]
    visited_pairs = raw["visitedPairs"]
    result: dict[str, Any] = {
        "positions": int(positions),
        "visitedActions": raw["visitedActions"] / positions if positions else 0.0,
        "visitedPairs": int(visited_pairs),
        "selectedActionIsQMax": raw["selectedActionIsQMax"] / positions if positions else 0.0,
        "selectedActionQRank": raw["selectedActionQRank"] / positions if positions else 0.0,
        "selectedActionQPercentile": raw["selectedActionQPercentile"] / positions if positions else 0.0,
        "qSpread": raw["qSpread"] / positions if positions else 0.0,
        "predictedSelectedActionIsTargetQMax": raw["predictedSelectedActionIsTargetQMax"] / positions if positions else 0.0,
        "predictedSelectedActionTargetQRank": raw["predictedSelectedActionTargetQRank"] / positions if positions else 0.0,
        "targetPolicyPairwiseAccuracy": None,
        "targetPolicyPairwisePairs": 0,
        "predictedPairwiseAccuracy": raw["predictedPairwiseAgreement"] / raw["predictedPairwisePairs"] if raw["predictedPairwisePairs"] else None,
        "predictedPairwisePairs": int(raw["predictedPairwisePairs"]),
        "qMse": raw["qMse"] / raw["qWeight"] if raw["qWeight"] else None,
        "qMae": raw["qMae"] / raw["qWeight"] if raw["qWeight"] else None,
    }
    return result


def invoke_chunk(
    function_name: str,
    region: str,
    profile: str,
    chunk_id: str,
    games: list[dict[str, Any]],
    result_path: Path,
    retry_attempts: int,
) -> dict[str, Any]:
    if result_path.is_file():
        return {"chunkId": chunk_id, "status": "existing", "response": json.loads(result_path.read_text(encoding="utf-8"))}
    payload = json.dumps({"chunkId": chunk_id, "games": games}, separators=(",", ":"))
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
        "--cli-connect-timeout",
        "30",
        "--cli-read-timeout",
        "900",
        "--no-cli-pager",
        "--profile",
        profile,
        "--region",
        region,
    ]
    started = time.perf_counter()
    last_error = "unknown Lambda invocation failure"
    for attempt in range(retry_attempts + 1):
        with tempfile.NamedTemporaryFile(prefix=f"{chunk_id}-", suffix=".json", delete=False) as request_file:
            request_path = Path(request_file.name)
            request_file.write(payload.encode())
        with tempfile.NamedTemporaryFile(prefix=f"{chunk_id}-response-", suffix=".json", delete=False) as response_file:
            response_path = Path(response_file.name)
        try:
            completed = subprocess.run(
                [*command, "--payload", f"fileb://{request_path}", str(response_path)],
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
            elif isinstance(response, dict) and response.get("schema") == "pathagon-qadv-evaluation-v1":
                result_path.write_text(json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                return {
                    "chunkId": chunk_id,
                    "status": "complete",
                    "response": response,
                    "duration": time.perf_counter() - started,
                    "attempt": attempt + 1,
                }
        finally:
            request_path.unlink(missing_ok=True)
            response_path.unlink(missing_ok=True)
    return {"chunkId": chunk_id, "status": "failed", "error": last_error, "duration": time.perf_counter() - started}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function-name", default="pathagon-qadv-evaluator-20260826")
    parser.add_argument("--games-dir", type=Path, default=DEFAULT_GAMES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-start", type=int, default=2026200000)
    parser.add_argument("--seed-end", type=int, default=2026217499)
    parser.add_argument("--split-seed", type=int, default=2026086601)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--max-payload-bytes", type=int, default=4_500_000)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--retry-attempts", type=int, default=2)
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--profile", default="pathagon")
    args = parser.parse_args()
    if args.seed_end < args.seed_start or args.concurrency < 1 or args.max_payload_bytes < 1_000:
        raise SystemExit("invalid seed range, concurrency, or payload limit")

    games_dir = args.games_dir if args.games_dir.is_absolute() else REPO_ROOT / args.games_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    selected = heldout_seeds(args.seed_start, args.seed_end, args.heldout_fraction, args.split_seed)
    records = iter_records(games_dir, selected)
    futures: dict[concurrent.futures.Future[dict[str, Any]], str] = {}
    completed_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    submitted = 0
    completed = 0

    print(f"lambda evaluation: {len(selected)} held-out games, concurrency={args.concurrency}, payload ceiling={args.max_payload_bytes} bytes", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        for chunk_id, games in iter_chunks(records, args.max_payload_bytes):
            submitted += 1
            result_path = chunks_dir / f"{chunk_id}.json"
            if result_path.is_file():
                result = invoke_chunk(args.function_name, args.region, args.profile, chunk_id, games, result_path, args.retry_attempts)
                done_results = [result]
            else:
                future = executor.submit(
                    invoke_chunk,
                    args.function_name,
                    args.region,
                    args.profile,
                    chunk_id,
                    games,
                    result_path,
                    args.retry_attempts,
                )
                futures[future] = chunk_id
                done_results = []
            while len(futures) >= args.concurrency:
                done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    futures.pop(future)
                    done_results.append(future.result())
            for result in done_results:
                completed += 1
                if result["status"] in {"complete", "existing"}:
                    completed_results.append(result)
                else:
                    failures.append(result)
                print(f"lambda evaluation progress: {completed} finished, {submitted} submitted; latest={result['chunkId']} status={result['status']}", flush=True)
        if futures:
            done, _ = concurrent.futures.wait(futures)
            for future in done:
                futures.pop(future)
                result = future.result()
                completed += 1
                if result["status"] in {"complete", "existing"}:
                    completed_results.append(result)
                else:
                    failures.append(result)
                print(f"lambda evaluation progress: {completed}/{submitted} finished; latest={result['chunkId']} status={result['status']}", flush=True)

    if failures:
        failure_path = output_dir / "failures.json"
        failure_path.write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise SystemExit(f"{len(failures)} Lambda chunks failed; see {failure_path}")

    raw_buckets = {phase: zero_raw() for phase in ("all", "placement", "relocation")}
    games = positions = q_positions = missing_q_positions = invalid_games = 0
    durations: list[float] = []
    for result in completed_results:
        response = result["response"]
        summary = response["summary"]
        games += int(summary.get("games", 0))
        positions += int(summary.get("positions", 0))
        q_positions += int(summary.get("qPositions", 0))
        missing_q_positions += int(summary.get("missingQPositions", 0))
        invalid_games += int(summary.get("invalidGames", 0))
        for phase in raw_buckets:
            merge_raw(raw_buckets[phase], summary["metrics"][phase])
        if result.get("duration") is not None:
            durations.append(float(result["duration"]))

    report = {
        "schema": "pathagon-qadv-lambda-evaluation-report-v1",
        "functionName": args.function_name,
        "region": args.region,
        "gamesDir": str(games_dir),
        "split": "heldout",
        "splitSeed": args.split_seed,
        "heldoutFraction": args.heldout_fraction,
        "heldoutGames": len(selected),
        "games": games,
        "positions": positions,
        "qPositions": q_positions,
        "missingQPositions": missing_q_positions,
        "invalidGames": invalid_games,
        "chunks": submitted,
        "concurrency": args.concurrency,
        "maxPayloadBytes": args.max_payload_bytes,
        "meanChunkSeconds": sum(durations) / len(durations) if durations else None,
        "metrics": {phase: finalize(raw) for phase, raw in raw_buckets.items()},
    }
    report_path = output_dir / "qadv-lambda-evaluation-heldout.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
