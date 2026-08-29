#!/usr/bin/env python3
"""Run a repeatable native QAdv search benchmark and emit one JSON report."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = REPO_ROOT / "pathagon/engine-rs/target/release/pathagon-selfplay"
DEFAULT_MODEL = REPO_ROOT / "work/rust-qadv-spike/qadv-gnn-qadv-campaign-17k-clean.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026280001)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--max-plies", type=int, default=196)
    parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args()

    binary = args.binary if args.binary.is_absolute() else REPO_ROOT / args.binary
    model = args.model if args.model.is_absolute() else REPO_ROOT / args.model
    if not binary.exists():
        parser.error(f"native binary does not exist: {binary}; build it first")
    if not model.exists():
        parser.error(f"QAdv ONNX model does not exist: {model}")

    command = [
        str(binary),
        "--games",
        str(args.games),
        "--seed",
        str(args.seed),
        "--max-plies",
        str(args.max_plies),
        "--opening-random-plies",
        "2",
        "--simulations",
        str(args.simulations),
        "--qadv-onnx",
        str(model),
        "--guided",
        "--opening-moves",
        "16",
        "--opening-temperature",
        "1.8",
        "--opening-randomness",
        "0.3",
        "--temperature-moves",
        "48",
        "--policy-temperature",
        "1.15",
        "--pathfinder-guidance",
        "0.45",
        "--placement-guidance",
        "0.3",
        "--pathfinder-temperature",
        "1.15",
        "--pathfinder-depth",
        "2",
        "--pathfinder-beam",
        "8",
        "--pathfinder-nodes",
        "512",
        "--qadv-weight",
        "1.0",
        "--progress-every",
        str(args.progress_every),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    wall_seconds = time.perf_counter() - started
    if completed.returncode:
        raise SystemExit(completed.stderr or completed.stdout)

    summary = None
    for line in completed.stdout.splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("schemaVersion") == 2:
            summary = candidate
    if summary is None:
        raise SystemExit("native benchmark did not emit a schemaVersion=2 summary")

    print(
        json.dumps(
            {
                "schema": "pathagon-native-qadv-benchmark-v1",
                "binary": str(binary.relative_to(REPO_ROOT)),
                "model": str(model.relative_to(REPO_ROOT)),
                "wallSeconds": round(wall_seconds, 6),
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
