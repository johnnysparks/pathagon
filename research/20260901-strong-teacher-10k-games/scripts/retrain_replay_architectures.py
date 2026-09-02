#!/usr/bin/env python3
"""Retrain the existing replay learners on a frozen teacher-game split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_ROOT = REPO_ROOT / "research/20260824-gnn-cnn-lab"
TEACHER_ID = "rust-pathfinder-teacher-d5-b256-500k-v1"
sys.path.insert(0, str(LEGACY_ROOT))

from python.game import BoardConfig  # noqa: E402
from python.train import (  # noqa: E402
    build_model,
    choose_device,
    load_replay_source,
    model_state_hash,
    save_model,
    train_qadv_replay,
    train_replay,
)


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("moves"), list):
                raise ValueError(f"{path}:{line_number}: expected a game record")
            records.append(value)
    return records


def split_records(records: list[dict[str, Any]], heldout_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout fraction must be between zero and one")
    train: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    for record in records:
        game_seed = int(record["seed"])
        digest = hashlib.sha256(f"{seed}\0{game_seed}".encode("utf-8")).digest()
        (heldout if digest[0] < int(heldout_fraction * 256) else train).append(record)
    # Keep smoke runs useful too; the full 10k split is comfortably larger
    # than this deterministic boundary case.
    if not heldout and len(train) > 1:
        heldout.append(train.pop())
    if not train and len(heldout) > 1:
        train.append(heldout.pop())
    if not train or not heldout:
        raise ValueError("game split produced an empty partition")
    return train, heldout


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")


def opponent_profile_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        agents = record.get("agents", {})
        specifications = record.get("agentSpecifications", {})
        opponent_players = [player for player in ("light", "dark") if agents.get(player) != TEACHER_ID]
        if not opponent_players:
            # The stable teacher ID is defined below; retaining this fallback
            # makes malformed records visible in the report instead of hiding
            # them behind a KeyError.
            key = "unknown"
        else:
            manifest = specifications.get(opponent_players[0], {}).get("manifest", {})
            key = f"{manifest.get('depth', 0)}:{manifest.get('beam', 0)}:{manifest.get('nodeBudget', 0)}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def evaluate_model(model: Any, examples: list[Any], max_examples: int, seed: int) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot evaluate an empty replay split")
    rng = random.Random(seed)
    if len(examples) > max_examples:
        selected = rng.sample(examples, max_examples)
    else:
        selected = examples
    model.eval()
    top1 = 0
    top3 = 0
    value_squared_error = 0.0
    value_sign_correct = 0
    with torch.no_grad():
        for example in selected:
            actions = list(example.state.legal_actions())
            if getattr(model, "qadv", False):
                logits, value, _q_values, _advantages = model.policy_value_q(example.state, actions)
            else:
                logits, value = model.policy_value(example.state, actions)
            expected_index = actions.index(example.action)
            ordering = torch.argsort(logits, descending=True).detach().cpu().tolist()
            top1 += int(ordering[0] == expected_index)
            top3 += int(expected_index in ordering[:3])
            actual_value = float(value.detach().cpu())
            value_squared_error += (actual_value - float(example.value)) ** 2
            value_sign_correct += int((actual_value >= 0.0) == (float(example.value) >= 0.0))
    count = len(selected)
    return {
        "examples": count,
        "top1": top1,
        "top1Rate": top1 / count,
        "top3": top3,
        "top3Rate": top3 / count,
        "valueMse": value_squared_error / count,
        "valueSignRate": value_sign_correct / count,
    }


def train_one(spec: dict[str, Any], train_examples: list[Any], heldout_examples: list[Any], args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    seed = args.seed + int(spec["seedOffset"])
    torch.manual_seed(seed)
    model = build_model(
        spec["architecture"],
        spec["hidden"],
        spec["layers"],
        spec["blocks"],
        7,
        qadv=bool(spec["qadv"]),
    ).to(device)
    started = time.perf_counter()
    if spec["qadv"]:
        training = train_qadv_replay(
            model,
            train_examples,
            args.steps,
            args.learning_rate,
            seed,
            q_weight=1.0,
            advantage_weight=0.5,
            rank_weight=0.25,
            symmetry_augmentation=True,
        )
    else:
        policy_loss, value_loss = train_replay(
            model,
            train_examples,
            args.steps,
            args.learning_rate,
            seed,
            symmetry_augmentation=True,
            value_weight=1.0,
        )
        training = {"policyLoss": policy_loss, "valueLoss": value_loss}
    model.eval()
    checkpoint_hash = model_state_hash(model)
    report = {
        "name": spec["name"],
        "architecture": spec["architecture"],
        "qadv": spec["qadv"],
        "hidden": spec["hidden"],
        "layers": spec["layers"],
        "blocks": spec["blocks"],
        "seed": seed,
        "steps": args.steps,
        "learningRate": args.learning_rate,
        "device": str(device),
        "trainingSeconds": time.perf_counter() - started,
        "training": training,
        "train": evaluate_model(model, train_examples, args.max_eval_examples, seed + 100),
        "heldout": evaluate_model(model, heldout_examples, args.max_eval_examples, seed + 200),
        "modelHash": checkpoint_hash,
        "qTargetPositions": sum(1 for example in train_examples if example.action_values is not None),
    }
    checkpoint = args.output_dir / f"{spec['name']}.pt"
    save_model(model, checkpoint, {
        "mode": "strong-teacher-replay-retrain",
        "data": str(args.input),
        "split": str(args.split_dir),
        "architecture": spec["architecture"],
        "qadv": spec["qadv"],
        "steps": args.steps,
        "seed": seed,
        "teacher": {"depth": 5, "beam": 256, "nodeBudget": 500_000},
        "modelHash": checkpoint_hash,
        "heldout": report["heldout"],
    })
    report["checkpoint"] = str(checkpoint.relative_to(REPO_ROOT))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=2026090101)
    parser.add_argument("--seed", type=int, default=2026090102)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-eval-examples", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    args.input = args.input if args.input.is_absolute() else REPO_ROOT / args.input
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    args.split_dir = args.split_dir or args.output_dir / "split"
    args.split_dir = args.split_dir if args.split_dir.is_absolute() else REPO_ROOT / args.split_dir
    if args.steps < 1 or args.learning_rate <= 0.0 or args.max_eval_examples < 1:
        raise SystemExit("steps, learning rate, and max-eval-examples must be positive")
    records = read_records(args.input)
    train_records, heldout_records = split_records(records, args.heldout_fraction, args.split_seed)
    args.split_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.split_dir / "train.jsonl"
    heldout_path = args.split_dir / "heldout.jsonl"
    write_records(train_path, train_records)
    write_records(heldout_path, heldout_records)
    config = BoardConfig(7, 14, 20)
    print(f"loading replay: {len(train_records)} train games / {len(heldout_records)} heldout games", file=sys.stderr, flush=True)
    train_examples = load_replay_source(train_path, config, None, None)
    heldout_examples = load_replay_source(heldout_path, config, None, None)
    if not train_examples or not heldout_examples:
        raise SystemExit("replay split produced no examples")
    device = choose_device(args.device)
    specs = [
        {"name": "gnn-policy-value", "architecture": "gnn", "hidden": 64, "layers": 8, "blocks": 4, "qadv": False, "seedOffset": 0},
        {"name": "cnn-policy-value", "architecture": "cnn", "hidden": 64, "layers": 8, "blocks": 4, "qadv": False, "seedOffset": 1},
        {"name": "gnn-qadv-replay", "architecture": "gnn", "hidden": 64, "layers": 8, "blocks": 4, "qadv": True, "seedOffset": 2},
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for spec in specs:
        print(f"training {spec['name']} for {args.steps} steps on {device}", file=sys.stderr, flush=True)
        reports.append(train_one(spec, train_examples, heldout_examples, args, device))
        print(json.dumps(reports[-1], sort_keys=True), flush=True)
    report = {
        "schemaVersion": 1,
        "status": "complete",
        "input": str(args.input),
        "inputSha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "teacher": {"depth": 5, "beam": 256, "nodeBudget": 500_000},
        "games": len(records),
        "opponentProfiles": opponent_profile_counts(records),
        "trainGames": len(train_records),
        "heldoutGames": len(heldout_records),
        "trainExamples": len(train_examples),
        "heldoutExamples": len(heldout_examples),
        "splitSeed": args.split_seed,
        "split": {"train": str(train_path), "heldout": str(heldout_path)},
        "models": reports,
    }
    (args.output_dir / "retraining-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "models": len(reports), "heldoutExamples": len(heldout_examples)}, sort_keys=True))


if __name__ == "__main__":
    main()
