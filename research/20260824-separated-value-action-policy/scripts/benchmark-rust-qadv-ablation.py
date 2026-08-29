#!/usr/bin/env python3
"""Compare QAdv tree guidance, bounded proof, and extra-node baselines.

Every variant uses the same model, seed range, opening policy, opponent, and
game budget. The report is intentionally an ablation record rather than an
Elo claim; use the same seeds again when promoting a candidate to a larger
cross-play run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = REPO_ROOT / "pathagon/engine-rs/target/release/pathagon-selfplay"
DEFAULT_MODEL = REPO_ROOT / "work/rust-qadv-spike/qadv-gnn-qadv-campaign-17k-clean.onnx"


VARIANTS = {
    "baseline": ("Root-only QAdv seeds disabled", lambda sims: ["--no-qadv-tree-seeds", "--tactical-proof-horizon", "0", "--simulations", str(sims)]),
    "tree-seeds": ("QAdv seeds every expanded node", lambda sims: ["--tactical-proof-horizon", "0", "--simulations", str(sims)]),
    "proof-h2": ("Tree seeds plus two-ply proof", lambda sims: ["--tactical-proof-horizon", "2", "--simulations", str(sims)]),
    "proof-h3": ("Tree seeds plus three-ply proof", lambda sims: ["--tactical-proof-horizon", "3", "--simulations", str(sims)]),
    "double-sim": ("Root-only baseline at 2x simulations", lambda sims: ["--no-qadv-tree-seeds", "--tactical-proof-horizon", "0", "--simulations", str(sims * 2)]),
}


def run_variant(binary: Path, model: Path, args: argparse.Namespace, variant: str) -> dict:
    label, options = VARIANTS[variant]
    command = [
        str(binary),
        "--games", str(args.games),
        "--seed", str(args.seed),
        "--max-plies", str(args.max_plies),
        "--opening-random-plies", "2",
        "--qadv-onnx", str(model),
        "--guided",
        "--opening-moves", "16",
        "--opening-temperature", "1.8",
        "--opening-randomness", "0.3",
        "--temperature-moves", "0",
        "--policy-temperature", "1.0",
        "--pathfinder-guidance", "0",
        "--placement-guidance", "0",
        "--qadv-weight", "1.0",
        "--tactical-simulations", str(args.simulations),
        "--tactical-proof-nodes", str(args.proof_nodes),
        "--opponent", args.opponent,
        "--progress-every", "0",
        *options(args.simulations),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode:
        return {"variant": variant, "label": label, "status": "failed", "seconds": elapsed, "error": (completed.stderr or completed.stdout or "native benchmark failed").strip()}
    summary = None
    for line in completed.stdout.splitlines():
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if candidate.get("schemaVersion") == 2:
            summary = candidate
    if summary is None:
        return {"variant": variant, "label": label, "status": "failed", "seconds": elapsed, "error": "missing schemaVersion=2 summary"}
    return {"variant": variant, "label": label, "status": "complete", "seconds": elapsed, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026280001)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--max-plies", type=int, default=196)
    parser.add_argument("--proof-nodes", type=int, default=50_000)
    parser.add_argument("--opponent", default="random")
    parser.add_argument("--variants", default=",".join(VARIANTS), help="comma-separated variant IDs")
    args = parser.parse_args()
    if args.games < 1 or args.simulations < 0 or args.proof_nodes < 0:
        parser.error("games must be positive and budgets must be non-negative")
    binary = args.binary.resolve()
    model = args.model.resolve()
    if not binary.is_file():
        parser.error(f"native binary does not exist: {binary}; build it first")
    if not model.is_file():
        parser.error(f"QAdv ONNX model does not exist: {model}")
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = [item for item in variants if item not in VARIANTS]
    if unknown:
        parser.error(f"unknown variants: {', '.join(unknown)}")

    started = time.perf_counter()
    results = [run_variant(binary, model, args, variant) for variant in variants]
    report = {
        "schema": "pathagon-rust-qadv-ablation-v1",
        "binary": str(binary.relative_to(REPO_ROOT)) if binary.is_relative_to(REPO_ROOT) else str(binary),
        "model": str(model.relative_to(REPO_ROOT)) if model.is_relative_to(REPO_ROOT) else str(model),
        "controls": {
            "games": args.games,
            "seed": args.seed,
            "simulations": args.simulations,
            "maxPlies": args.max_plies,
            "proofNodes": args.proof_nodes,
            "opponent": args.opponent,
        },
        "graphReuse": "not-implemented; keep this field explicit until the transposition experiment lands",
        "seconds": round(time.perf_counter() - started, 6),
        "results": results,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
