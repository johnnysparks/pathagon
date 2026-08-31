#!/usr/bin/env python3
"""Fan out bounded arena games to the private transition-policy Lambda.

The function is intentionally one-game-per-invocation. This runner keeps the
arena's deterministic seed/index mapping and writes one normalized replay per
line so it can be consumed by the existing arena summarizer and native audit.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import subprocess
import tempfile
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--function-name", required=True)
    parser.add_argument("--region", default="us-east-2")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--games", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026083002)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--raw-dir", type=pathlib.Path, required=True)
    parser.add_argument("--max-plies", type=int, default=80)
    parser.add_argument("--opening-random-plies", type=int, default=2)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--nodes", type=int, default=1_000_000)
    parser.add_argument("--beam", type=int, default=32)
    parser.add_argument("--deadline-ms", type=int, default=2_800)
    return parser.parse_args()


def invoke_one(args: argparse.Namespace, index: int) -> tuple[int, dict[str, Any]]:
    seed = args.seed + index
    payload = {
        "seed": seed,
        "candidate_light": index % 2 == 0,
        "max_plies": args.max_plies,
        "opening_random_plies": args.opening_random_plies,
        "depth": args.depth,
        "nodes": args.nodes,
        "beam": args.beam,
        "deadline_ms": args.deadline_ms,
    }
    raw_path = args.raw_dir / f"{index:05d}.json"
    command = [
        "aws",
        "lambda",
        "invoke",
        "--function-name",
        args.function_name,
        "--region",
        args.region,
        "--invocation-type",
        "RequestResponse",
        "--cli-binary-format",
        "raw-in-base64-out",
        "--cli-read-timeout",
        "1000",
        "--payload",
        json.dumps(payload, separators=(",", ":")),
        str(raw_path),
    ]
    env = os.environ.copy()
    env["AWS_MAX_ATTEMPTS"] = "1"
    started = time.monotonic()
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=890,
    )
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"index {index} seed {seed} failed ({completed.returncode}) after "
            f"{elapsed:.1f}s: {completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        envelope = json.loads(raw_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"index {index} seed {seed} invalid Lambda response: {error}") from error
    if envelope.get("FunctionError"):
        raise RuntimeError(f"index {index} seed {seed} Lambda error: {envelope}")
    payload_value = envelope.get("Payload", envelope)
    if isinstance(payload_value, str):
        payload_value = json.loads(payload_value)
    if not isinstance(payload_value, dict) or "record" not in payload_value:
        raise RuntimeError(f"index {index} seed {seed} missing record: {envelope}")
    record = payload_value["record"]
    record["index"] = index
    record["lambdaSeed"] = seed
    record["candidateLight"] = payload_value.get("candidateLight", payload["candidate_light"])
    record["candidateId"] = payload_value.get("candidateId")
    record["lambdaElapsedSeconds"] = round(elapsed, 3)
    return index, record


def main() -> None:
    args = parse_args()
    if args.games <= 0 or args.workers <= 0 or args.start_index < 0:
        raise SystemExit("games and workers must be positive; start-index must be non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    indices = range(args.start_index, args.start_index + args.games)
    records: dict[int, dict[str, Any]] = {}
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(invoke_one, args, index): index for index in indices}
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                completed_index, record = future.result()
            except Exception as error:  # noqa: BLE001 - preserve all invocation diagnostics
                failures.append(str(error))
                print(f"FAIL {error}", flush=True)
            else:
                records[completed_index] = record
                print(
                    f"DONE {completed_index} seed={record['lambdaSeed']} "
                    f"winner={record.get('winner')} plies={len(record.get('actions', []))}",
                    flush=True,
                )
    if failures:
        raise SystemExit("Lambda arena failures:\n" + "\n".join(sorted(failures)))
    with args.output.open("w", encoding="utf-8") as handle:
        for index in sorted(records):
            handle.write(json.dumps(records[index], separators=(",", ":")) + "\n")
    print(f"WROTE {len(records)} records to {args.output}", flush=True)


if __name__ == "__main__":
    main()
