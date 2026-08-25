#!/usr/bin/env python3
"""Score a 7x7 policy/value checkpoint on held-out replay positions."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import torch

# Allow direct execution from the repository root as well as ``python -m``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.gnn.data import ReplayExample, load_replay_examples
from research.gnn.game import BoardConfig
from research.gnn.train import choose_device, load_model


def select_examples(examples: List[ReplayExample], limit: int, seed: int) -> List[ReplayExample]:
    if limit <= 0 or limit >= len(examples):
        return examples
    rng = random.Random(seed)
    selected = rng.sample(examples, limit)
    selected.sort(key=lambda example: (example.seed, example.state.ply))
    return selected


def phase_name(example: ReplayExample) -> str:
    return "placement" if example.action.kind == 0 else "relocation"


def empty_metrics() -> Dict[str, float]:
    return {
        "examples": 0,
        "policyNll": 0.0,
        "policyTop1": 0.0,
        "policyTop5": 0.0,
        "valueMse": 0.0,
        "valueMae": 0.0,
        "valueSignAccuracy": 0.0,
        "nonDrawValueExamples": 0,
    }


def score(model, examples: Iterable[ReplayExample]) -> Dict[str, object]:
    metrics: Dict[str, Dict[str, float]] = {
        "all": empty_metrics(),
        "placement": empty_metrics(),
        "relocation": empty_metrics(),
    }
    target_counts = {"win": 0, "loss": 0, "draw": 0}

    with torch.no_grad():
        for example in examples:
            phase = phase_name(example)
            actions = list(example.state.legal_actions())
            target_index = actions.index(example.action)
            logits, value = model.policy_value(example.state, actions)
            log_probabilities = torch.log_softmax(logits, dim=0)
            order = torch.argsort(logits, descending=True)
            predicted_value = float(value.detach().cpu())
            expected_value = example.value
            target_bucket = "draw" if expected_value == 0.0 else ("win" if expected_value > 0 else "loss")
            target_counts[target_bucket] += 1

            for bucket in (metrics["all"], metrics[phase]):
                bucket["examples"] += 1
                bucket["policyNll"] += float(-log_probabilities[target_index].detach().cpu())
                bucket["policyTop1"] += float(int(int(order[0]) == target_index))
                bucket["policyTop5"] += float(int(target_index in order[:5].tolist()))
                bucket["valueMse"] += (predicted_value - expected_value) ** 2
                bucket["valueMae"] += abs(predicted_value - expected_value)
                if expected_value:
                    bucket["nonDrawValueExamples"] += 1
                    bucket["valueSignAccuracy"] += float(
                        int((predicted_value >= 0.0) == (expected_value > 0.0))
                    )

    for bucket in metrics.values():
        count = bucket["examples"]
        non_draw = bucket["nonDrawValueExamples"]
        if count:
            for key in ("policyNll", "policyTop1", "policyTop5", "valueMse", "valueMae"):
                bucket[key] /= count
        if non_draw:
            bucket["valueSignAccuracy"] /= non_draw

    return {"metrics": metrics, "targetCounts": target_counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--size", type=int, default=7)
    parser.add_argument("--reserve", type=int, default=14)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    config = BoardConfig(args.size, args.reserve)
    examples = load_replay_examples(Path(args.data), config=config)
    examples = select_examples(examples, args.max_examples, args.seed)
    model = load_model(Path(args.checkpoint), device)
    model.eval()
    report = score(model, examples)
    report.update(
        {
            "checkpoint": args.checkpoint,
            "data": args.data,
            "device": str(device),
            "boardSize": args.size,
            "reservePerPlayer": args.reserve,
            "games": len({example.seed for example in examples}),
            "examples": len(examples),
            "maxExamples": args.max_examples,
            "seed": args.seed,
        }
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
