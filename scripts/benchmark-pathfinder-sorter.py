#!/usr/bin/env python3
"""Run a matched Pathfinder-vs-compact-sorter arena.

The candidate uses the same alpha-beta depth, beam, and node ceiling as the
deeper Pathfinder baseline. Its extra signals are a compact GNN policy for root
ordering plus a transposition-aware tactical extension in that same search
budget. This keeps the experiment about smarter search rather than silently
buying more nodes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from research.gnn.game import BoardConfig
from research.gnn.league import (
    AgentSpec,
    HeuristicAgent,
    SorterPathfinderAgent,
    SorterOnlyPathfinderAgent,
    checkpoint_hash,
    load_model,
    play_game,
    summarize,
)
from research.gnn.train import choose_device
from research.gnn.contract import agent_manifest


DEFAULT_SORTER = REPO_ROOT / "research/runs/gnn/benchmark-7x7/small-gnn-warmstart.pt"


def run(args: argparse.Namespace) -> dict:
    device = choose_device(args.device)
    config = BoardConfig(args.size, args.reserve, args.max_plies)
    sorter_path = args.sorter if args.sorter.is_absolute() else REPO_ROOT / args.sorter
    if not sorter_path.is_file():
        raise FileNotFoundError(f"compact sorter checkpoint does not exist: {sorter_path}")
    model = load_model(sorter_path, device)
    model.eval()
    sorter_hash = checkpoint_hash(sorter_path)

    baseline = AgentSpec(
        "pathfinder-deep-10k-7x7",
        "Pathfinder + deeper search",
        "heuristic",
        HeuristicAgent(depth=args.depth, beam_width=args.beam, max_nodes=args.nodes),
        agent_manifest(runtime="python", depth=args.depth, beam=args.beam, node_budget=args.nodes),
    )
    candidate_class = SorterPathfinderAgent if args.variant == "smarter" else SorterOnlyPathfinderAgent
    candidate_depth = args.candidate_depth if args.candidate_depth is not None else args.depth
    candidate_beam = args.candidate_beam if args.candidate_beam is not None else args.beam
    candidate_nodes = args.candidate_nodes if args.candidate_nodes is not None else args.nodes
    candidate_label = (
        "Pathfinder + compact sorter + smarter search"
        if args.variant == "smarter"
        else "Pathfinder + compact root sorter"
    )
    candidate = AgentSpec(
        "pathfinder-compact-sorter-7x7",
        candidate_label,
        "heuristic",
        candidate_class(
            model,
            depth=candidate_depth,
            beam_width=candidate_beam,
            max_nodes=candidate_nodes,
            top_k=args.top_k,
        ),
        agent_manifest(
            runtime="python",
            depth=candidate_depth,
            beam=candidate_beam,
            node_budget=candidate_nodes,
            model_hash=sorter_hash,
        ),
    )

    records = []
    started = time.perf_counter()
    for index in range(args.games):
        light, dark = (candidate, baseline) if index % 2 == 0 else (baseline, candidate)
        records.append(play_game(light, dark, config, args.seed + index, args.opening_random_plies))
    elapsed = time.perf_counter() - started

    candidate_summary = summarize(records, candidate.id)
    baseline_summary = summarize(records, baseline.id)
    candidate_wins_by_color = {
        "light": sum(record["winner"] == "light" and record["agents"]["light"] == candidate.id for record in records),
        "dark": sum(record["winner"] == "dark" and record["agents"]["dark"] == candidate.id for record in records),
    }
    return {
        "schema": "pathagon-pathfinder-sorter-arena-v1",
        "boardSize": args.size,
        "reservePerPlayer": config.reserve_per_player,
        "maxPlies": args.max_plies,
        "openingRandomPlies": args.opening_random_plies,
        "seed": args.seed,
        "games": args.games,
        "elapsedSeconds": round(elapsed, 6),
        "search": {
            "variant": args.variant,
            "depth": args.depth,
            "beam": args.beam,
            "nodes": args.nodes,
            "candidateDepth": candidate_depth,
            "candidateBeam": candidate_beam,
            "candidateNodes": candidate_nodes,
            "sorterTopK": args.top_k,
        },
        "sorter": {"checkpoint": str(sorter_path.relative_to(REPO_ROOT)), "sha256": sorter_hash},
        "candidate": {**candidate_summary, "winsByColor": candidate_wins_by_color},
        "baseline": baseline_summary,
        "pairedScore": round((candidate_summary["wins"] + 0.5 * candidate_summary["draws"]) / max(1, args.games), 4),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sorter", type=Path, default=DEFAULT_SORTER)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026082700)
    parser.add_argument("--size", type=int, default=7)
    parser.add_argument("--reserve", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--opening-random-plies", type=int, default=2)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--beam", type=int, default=16)
    parser.add_argument("--nodes", type=int, default=10_000)
    parser.add_argument("--candidate-depth", type=int)
    parser.add_argument("--candidate-beam", type=int)
    parser.add_argument("--candidate-nodes", type=int)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--variant", choices=("sorter", "smarter"), default="smarter")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = run(args)
    if args.out:
        output = args.out if args.out.is_absolute() else REPO_ROOT / args.out
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"records"}}, sort_keys=True))


if __name__ == "__main__":
    main()
