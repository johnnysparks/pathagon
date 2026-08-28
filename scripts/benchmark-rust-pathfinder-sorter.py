#!/usr/bin/env python3
"""Run the pure-Rust Pathfinder/ONNX-sorter arena.

The executable owns both players, rules, alpha-beta search, and ONNX inference.
This wrapper only supplies a reproducible command and records the emitted
summary and complete game archive; it is not part of the play-time engine. `--sorter-kind qadv` swaps in
the QAdv action-value head, and `--sorter-all-actions` expands the scoring pool
beyond Pathfinder's heuristic beam.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = REPO_ROOT / "engine-rs/target/release/pathagon-selfplay"
DEFAULT_MODEL = REPO_ROOT / "research/experiments/20260827-pathfinder-rust-sorter/artifacts/compact-gnn-policy.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--candidate",
        choices=("sorter", "probe-search", "tt-search", "guard-search", "filter-search"),
        default="sorter",
        help="candidate agent; probe-search tests the exact Rust root scout",
    )
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026084200)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--opening-random-plies", type=int, default=2)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=2_000)
    parser.add_argument("--candidate-depth", type=int, default=0)
    parser.add_argument("--candidate-beam", type=int, default=0)
    parser.add_argument("--candidate-nodes", type=int, default=0)
    parser.add_argument("--sorter-top-k", type=int, default=4)
    parser.add_argument(
        "--sorter-root-limit",
        type=int,
        default=0,
        help="root candidate cap (0 uses twice the Pathfinder beam)",
    )
    parser.add_argument(
        "--sorter-min-margin",
        type=float,
        default=0.0,
        help="minimum ONNX score margin required to reorder the first root candidate",
    )
    parser.add_argument(
        "--sorter-max-heuristic-gap",
        type=int,
        default=0,
        help="only reorder when the ONNX choice is within this Pathfinder score gap (0 disables)",
    )
    parser.add_argument(
        "--sorter-all-actions",
        action="store_true",
        help="score every legal root action with ONNX before keeping the top-k hints",
    )
    parser.add_argument(
        "--opponent",
        choices=("deep-search", "probe-search", "tt-search", "guard-search", "filter-search"),
        default="deep-search",
        help="baseline opponent; probe-search adds a bounded exact root scout",
    )
    parser.add_argument("--probe-depth", type=int, default=2)
    parser.add_argument("--probe-nodes", type=int, default=256)
    parser.add_argument("--probe-actions", type=int, default=8)
    parser.add_argument(
        "--sorter-kind",
        choices=("policy", "qadv"),
        default="policy",
        help="ONNX head used for ordering: policy logits or QAdv action values",
    )
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="report path; complete moves are written beside it as <stem>.games.jsonl",
    )
    args = parser.parse_args()

    binary = args.binary if args.binary.is_absolute() else REPO_ROOT / args.binary
    model = args.model if args.model.is_absolute() else REPO_ROOT / args.model
    if not binary.is_file():
        parser.error(f"Rust binary does not exist: {binary}; build it first")
    if args.candidate == "sorter" and not model.is_file():
        parser.error(f"ONNX sorter does not exist: {model}; export it first")

    command = [
        str(binary),
        "--games", str(args.games),
        "--seed", str(args.seed),
        "--max-plies", str(args.max_plies),
        "--opening-random-plies", str(args.opening_random_plies),
        "--depth", str(args.depth),
        "--beam", str(args.beam),
        "--nodes", str(args.nodes),
        "--sorter-qadv-onnx" if args.sorter_kind == "qadv" else "--sorter-onnx", str(model),
        "--sorter-top-k", str(args.sorter_top_k),
        "--sorter-root-limit", str(args.sorter_root_limit),
        "--sorter-min-margin", str(args.sorter_min_margin),
        "--sorter-max-heuristic-gap", str(args.sorter_max_heuristic_gap),
        "--opponent", args.opponent,
        "--progress-every", str(args.progress_every),
        "--jsonl",
    ]
    if args.candidate == "probe-search":
        command = [
            str(binary),
            "--games", str(args.games),
            "--seed", str(args.seed),
            "--max-plies", str(args.max_plies),
            "--opening-random-plies", str(args.opening_random_plies),
            "--depth", str(args.depth),
            "--beam", str(args.beam),
            "--nodes", str(args.nodes),
            "--root-probe-depth", str(args.probe_depth),
            "--root-probe-nodes", str(args.probe_nodes),
            "--root-probe-actions", str(args.probe_actions),
            "--opponent", args.opponent,
            "--progress-every", str(args.progress_every),
            "--jsonl",
        ]
    elif args.candidate in ("tt-search", "guard-search", "filter-search"):
        command = [
            str(binary),
            "--games", str(args.games),
            "--seed", str(args.seed),
            "--max-plies", str(args.max_plies),
            "--opening-random-plies", str(args.opening_random_plies),
            "--depth", str(args.depth),
            "--beam", str(args.beam),
            "--nodes", str(args.nodes),
            "--opponent", args.opponent,
            "--progress-every", str(args.progress_every),
            "--jsonl",
        ]
        command.insert(
            command.index("--opponent"),
            "--tt-order"
            if args.candidate == "tt-search"
            else "--tactical-root-guard"
            if args.candidate == "guard-search"
            else "--tactical-root-filter",
        )
    if args.candidate_depth > 0:
        command.extend(["--candidate-depth", str(args.candidate_depth)])
    if args.candidate_beam > 0:
        command.extend(["--candidate-beam", str(args.candidate_beam)])
    if args.candidate_nodes > 0:
        command.extend(["--candidate-nodes", str(args.candidate_nodes)])
    if args.opponent == "probe-search":
        command.extend(
            [
                "--root-probe-depth", str(args.probe_depth),
                "--root-probe-nodes", str(args.probe_nodes),
                "--root-probe-actions", str(args.probe_actions),
            ]
        )
    if args.sorter_all_actions and args.candidate == "sorter":
        command.insert(command.index("--opponent"), "--sorter-all-actions")
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise SystemExit(completed.stderr or completed.stdout)
    records: list[str] = []
    for line in completed.stdout.splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            records.append(json.dumps(candidate, sort_keys=True, separators=(",", ":")))

    summary = None
    for line in completed.stderr.splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("schemaVersion") == 2:
            summary = candidate
    if summary is None:
        raise SystemExit("Rust arena did not emit a schemaVersion=2 summary")
    if len(records) != args.games:
        raise SystemExit(f"Rust arena emitted {len(records)} game records; expected {args.games}")

    output = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    archive = output.with_name(f"{output.stem}.games.jsonl")

    report = {
        "schema": "pathagon-rust-pathfinder-sorter-arena-v1",
        "binary": str(binary.relative_to(REPO_ROOT)),
        "model": str(model.relative_to(REPO_ROOT)) if args.candidate == "sorter" else None,
        "gamesArchive": str(archive.relative_to(REPO_ROOT)),
        "elapsedSeconds": round(elapsed, 6),
        "search": {
            "depth": args.depth,
            "beam": args.beam,
            "nodes": args.nodes,
            "sorterTopK": args.sorter_top_k,
            "sorterRootLimit": args.sorter_root_limit,
            "sorterMinMargin": args.sorter_min_margin,
            "sorterMaxHeuristicGap": args.sorter_max_heuristic_gap,
            "sorterKind": args.sorter_kind,
            "sorterAllActions": args.sorter_all_actions,
            "opponent": args.opponent,
            "candidate": args.candidate,
            "candidateDepth": args.candidate_depth,
            "candidateBeam": args.candidate_beam,
            "candidateNodes": args.candidate_nodes,
            "probeDepth": args.probe_depth,
            "probeNodes": args.probe_nodes,
            "probeActions": args.probe_actions,
        },
        "summary": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    archive.write_text("\n".join(records) + "\n", encoding="utf-8")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
