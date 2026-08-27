#!/usr/bin/env python3
"""Audit root-Q action ranking and optionally score a trained Q/advantage model."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from research.gnn.data import ReplayExample
from research.gnn.game import Action, BoardConfig
from research.gnn.train import choose_device, load_model, load_replay_source, split_replay_examples


def phase_name(example: ReplayExample) -> str:
    return "placement" if example.action.kind == 0 else "relocation"


def rank_of(values: list[float], index: int) -> int:
    target = values[index]
    return 1 + sum(value > target + 1.0e-6 for value in values)


def pairwise_agreement(left: list[float], right: list[float], indices: list[int]) -> tuple[int, int]:
    agreements = pairs = 0
    for left_index, left_action in enumerate(indices):
        for right_action in indices[left_index + 1 :]:
            left_delta = left[left_action] - left[right_action]
            right_delta = right[left_action] - right[right_action]
            if abs(left_delta) < 1.0e-6 or abs(right_delta) < 1.0e-6:
                continue
            pairs += 1
            agreements += int(left_delta * right_delta > 0)
    return agreements, pairs


def empty_metrics() -> dict[str, float]:
    return {
        "positions": 0,
        "visitedActions": 0,
        "visitedPairs": 0,
        "targetPolicyPairwiseAgreement": 0,
        "targetPolicyPairwisePairs": 0,
        "selectedActionIsQMax": 0,
        "selectedActionQRank": 0.0,
        "selectedActionQPercentile": 0.0,
        "qSpread": 0.0,
        "qMse": 0.0,
        "qMae": 0.0,
        "predictedPairwiseAgreement": 0,
        "predictedPairwisePairs": 0,
        "predictedSelectedActionIsTargetQMax": 0,
        "predictedSelectedActionTargetQRank": 0.0,
    }


def add_position(metrics: dict[str, float], example: ReplayExample, predicted_q: list[float] | None) -> None:
    if example.action_values is None or example.action_visits is None or example.action_value_actions is None:
        return
    actions = list(example.state.legal_actions())
    values_by_action = dict(zip(example.action_value_actions, example.action_values))
    visits_by_action = dict(zip(example.action_value_actions, example.action_visits))
    target_q = [float(values_by_action[action]) for action in actions]
    visits = [int(visits_by_action[action]) for action in actions]
    visited = [index for index, count in enumerate(visits) if count > 0]
    selected_index = actions.index(example.action)
    target_rank = rank_of(target_q, selected_index)
    target_max = int(target_rank == 1)
    percentile = 1.0 if len(target_q) == 1 else 1.0 - (target_rank - 1) / (len(target_q) - 1)
    bucket = metrics
    bucket["positions"] += 1
    bucket["visitedActions"] += len(visited)
    bucket["selectedActionIsQMax"] += target_max
    bucket["selectedActionQRank"] += target_rank
    bucket["selectedActionQPercentile"] += percentile
    bucket["qSpread"] += max(target_q) - min(target_q)
    policy = [0.0] * len(actions)
    if example.policy is not None and example.policy_actions is not None:
        policy_by_action = dict(zip(example.policy_actions, example.policy))
        policy = [float(policy_by_action[action]) for action in actions]
        agreements, pairs = pairwise_agreement(policy, target_q, visited)
        bucket["targetPolicyPairwiseAgreement"] += agreements
        bucket["targetPolicyPairwisePairs"] += pairs
    bucket["visitedPairs"] += sum(1 for left, left_index in enumerate(visited) for right_index in visited[left + 1 :])

    if predicted_q is None:
        return
    predicted = [float(value) for value in predicted_q]
    agreements, pairs = pairwise_agreement(predicted, target_q, visited)
    bucket["predictedPairwiseAgreement"] += agreements
    bucket["predictedPairwisePairs"] += pairs
    weighted = [index for index in visited if visits[index] > 0]
    predicted_index = max(weighted or list(range(len(actions))), key=lambda index: (predicted[index], -index))
    predicted_rank = rank_of(target_q, predicted_index)
    bucket["predictedSelectedActionIsTargetQMax"] += int(predicted_rank == 1)
    bucket["predictedSelectedActionTargetQRank"] += predicted_rank
    if visited:
        for index in visited:
            weight = visits[index] ** 0.5
            bucket["qMse"] += weight * (predicted[index] - target_q[index]) ** 2
            bucket["qMae"] += weight * abs(predicted[index] - target_q[index])


def finalize(metrics: dict[str, float]) -> dict[str, float]:
    positions = metrics["positions"]
    visited_actions = metrics["visitedActions"]
    visited_pairs = metrics["visitedPairs"]
    weighted_count = metrics.pop("_weightedCount", 0.0)
    if positions:
        for key in ("selectedActionIsQMax", "selectedActionQRank", "selectedActionQPercentile", "qSpread", "predictedSelectedActionIsTargetQMax", "predictedSelectedActionTargetQRank"):
            metrics[key] /= positions
    if visited_actions:
        metrics["visitedActions"] = visited_actions / positions
    if visited_pairs:
        metrics["targetPolicyPairwiseAccuracy"] = metrics.pop("targetPolicyPairwiseAgreement") / metrics.pop("targetPolicyPairwisePairs") if metrics["targetPolicyPairwisePairs"] else None
        metrics["predictedPairwiseAccuracy"] = metrics.pop("predictedPairwiseAgreement") / metrics.pop("predictedPairwisePairs") if metrics["predictedPairwisePairs"] else None
    else:
        metrics["targetPolicyPairwiseAccuracy"] = None
        metrics["predictedPairwiseAccuracy"] = None
    if weighted_count:
        metrics["qMse"] /= weighted_count
        metrics["qMae"] /= weighted_count
    else:
        metrics["qMse"] = None
        metrics["qMae"] = None
    return metrics


def score_raw(examples: Iterable[ReplayExample], model=None) -> dict[str, dict[str, float]]:
    buckets = {name: empty_metrics() for name in ("all", "placement", "relocation")}
    with torch.no_grad():
        for example in examples:
            predicted_q = None
            if model is not None:
                actions = list(example.state.legal_actions())
                _logits, _value, q_values, _advantages = model.policy_value_q(example.state, actions)
                predicted_q = q_values.detach().cpu().tolist()
            add_position(buckets["all"], example, predicted_q)
            add_position(buckets[phase_name(example)], example, predicted_q)
            if predicted_q is not None and example.action_visits:
                buckets["all"]["_weightedCount"] = buckets["all"].get("_weightedCount", 0.0) + sum(count ** 0.5 for count in example.action_visits if count > 0)
                buckets[phase_name(example)]["_weightedCount"] = buckets[phase_name(example)].get("_weightedCount", 0.0) + sum(count ** 0.5 for count in example.action_visits if count > 0)
    return buckets


def evaluate(examples: Iterable[ReplayExample], model=None) -> dict:
    buckets = score_raw(examples, model)
    return {name: finalize(metrics) for name, metrics in buckets.items()}


_WORKER_EXAMPLES: list[ReplayExample] = []
_WORKER_MODEL = None


def _init_worker(checkpoint: str, device_name: str) -> None:
    global _WORKER_MODEL
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _WORKER_MODEL = load_model(Path(checkpoint), torch.device(device_name), qadv=True)
    _WORKER_MODEL.eval()


def _score_worker(bounds: tuple[int, int]) -> tuple[tuple[int, int], dict[str, dict[str, float]]]:
    start, end = bounds
    return bounds, score_raw(_WORKER_EXAMPLES[start:end], _WORKER_MODEL)


def evaluate_parallel(examples: list[ReplayExample], checkpoint: Path, device: torch.device, workers: int) -> dict:
    if device.type != "cpu":
        raise ValueError("parallel evaluation currently requires --device cpu")
    global _WORKER_EXAMPLES
    _WORKER_EXAMPLES = examples
    chunk_size = max(1, (len(examples) + workers * 4 - 1) // (workers * 4))
    bounds = [(start, min(start + chunk_size, len(examples))) for start in range(0, len(examples), chunk_size)]
    combined = {name: empty_metrics() for name in ("all", "placement", "relocation")}
    context = mp.get_context("fork")
    with context.Pool(
        processes=min(workers, len(bounds)),
        initializer=_init_worker,
        initargs=(str(checkpoint), str(device)),
    ) as pool:
        for completed, (start_end, partial) in enumerate(pool.imap_unordered(_score_worker, bounds), 1):
            for name, metrics in partial.items():
                for key, value in metrics.items():
                    combined[name][key] = combined[name].get(key, 0.0) + value
            print(
                f"evaluation progress: {completed}/{len(bounds)} chunks "
                f"({start_end[1]}/{len(examples)} positions)",
                file=sys.stderr,
                flush=True,
            )
    return {name: finalize(metrics) for name, metrics in combined.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Q-target JSONL or generated batch directory")
    parser.add_argument("--checkpoint", help="optional qadv checkpoint to score against the pilot")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--seed-start", type=int, help="include only replay records at or after this seed")
    parser.add_argument("--seed-end", type=int, help="include only replay records at or before this seed")
    parser.add_argument("--split", choices=("all", "train", "heldout"), default="all")
    parser.add_argument("--split-seed", type=int, default=20260823, help="seed used for game-grouped train/heldout splitting")
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=1, help="parallel CPU workers for checkpoint evaluation")
    parser.add_argument("--output")
    args = parser.parse_args()
    if (args.seed_start is None) != (args.seed_end is None):
        raise SystemExit("--seed-start and --seed-end must be provided together")
    examples = load_replay_source(Path(args.data), BoardConfig(7, 14), args.seed_start, args.seed_end)
    total_examples = len(examples)
    if args.split != "all":
        train_examples, heldout_examples = split_replay_examples(examples, args.heldout_fraction, args.split_seed)
        examples = train_examples if args.split == "train" else heldout_examples
    if args.max_examples:
        examples = examples[: args.max_examples]
    model = None
    device = choose_device(args.device)
    if args.checkpoint:
        if args.workers > 1:
            if device.type != "cpu":
                raise SystemExit("--workers requires --device cpu")
        else:
            model = load_model(Path(args.checkpoint), device, qadv=True)
            model.eval()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.workers > 1 and not args.checkpoint:
        raise SystemExit("--workers requires --checkpoint")
    scored_metrics = (
        evaluate_parallel(examples, Path(args.checkpoint).resolve(), device, args.workers)
        if args.workers > 1
        else evaluate(examples, model)
    )
    report = {
        "data": args.data,
        "checkpoint": args.checkpoint,
        "seedStart": args.seed_start,
        "seedEnd": args.seed_end,
        "split": args.split,
        "splitSeed": args.split_seed,
        "heldoutFraction": args.heldout_fraction,
        "boardSize": 7,
        "reservePerPlayer": 14,
        "games": len({example.seed for example in examples}),
        "examples": len(examples),
        "totalExamples": total_examples,
        "workers": args.workers,
        "metrics": scored_metrics,
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
