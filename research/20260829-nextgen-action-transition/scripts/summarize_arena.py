#!/usr/bin/env python3
"""Summarize a paired candidate-vs-incumbent arena JSONL."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def candidate_result(record: dict, candidate: str) -> tuple[str, str]:
    color = "light" if record["agents"]["light"] == candidate else "dark"
    winner = record.get("winner")
    if winner is None:
        return color, "draw"
    return color, "win" if winner == color else "loss"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arena", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.arena.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise SystemExit("arena is empty")
    outcomes = [candidate_result(record, args.candidate) for record in records]
    overall = Counter(result for _, result in outcomes)
    by_color = {
        color: Counter(result for actual_color, result in outcomes if actual_color == color)
        for color in ("light", "dark")
    }
    plies = sum(int(record.get("plies", 0)) for record in records)
    points = overall["win"] + 0.5 * overall["draw"]
    candidate_nodes = []
    candidate_depths = []
    for record in records:
        color, _ = candidate_result(record, args.candidate)
        candidate_nodes.extend(
            move["nodes"] for move in record.get("moves", []) if move.get("player") == color
        )
        candidate_depths.extend(
            move["completedDepth"] for move in record.get("moves", []) if move.get("player") == color
        )
    mean_nodes = sum(candidate_nodes) / len(candidate_nodes) if candidate_nodes else 0.0
    mean_depth = sum(candidate_depths) / len(candidate_depths) if candidate_depths else 0.0
    rate = points / len(records)
    standard_error = math.sqrt(rate * (1.0 - rate) / len(records)) if records else 0.0
    print(
        json.dumps(
            {
                "candidate": args.candidate,
                "games": len(records),
                "wins": overall["win"],
                "losses": overall["loss"],
                "draws": overall["draw"],
                "points": points,
                "pointRate": rate,
                "pointRateStandardError": standard_error,
                "byColor": {color: dict(by_color[color]) for color in ("light", "dark")},
                "plies": plies,
                "meanCandidateNodes": mean_nodes,
                "meanCandidateCompletedDepth": mean_depth,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
