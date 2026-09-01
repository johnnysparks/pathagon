#!/usr/bin/env python3
"""Measure learner/heuristic recall of Rust Pathfinder teacher actions."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch


def load_trainer():
    path = Path(__file__).parents[2] / "20260829-action-transition-policy" / "scripts" / "train_transition_policy.py"
    spec = importlib.util.spec_from_file_location("transition_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import trainer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model(path: Path, trainer):
    document = json.loads(path.read_text(encoding="utf-8"))
    hidden = len(document["layers"][0]["bias"])
    model = trainer.make_model(len(trainer.FEATURE_NAMES), hidden)
    state = {}
    for index, layer in enumerate(document["layers"]):
        state[f"{index * 2}.weight"] = torch.tensor(layer["weights"], dtype=torch.float32)
        state[f"{index * 2}.bias"] = torch.tensor(layer["bias"], dtype=torch.float32)
    model.load_state_dict(state)
    model.eval()
    mean = torch.tensor(document["mean"], dtype=torch.float32)
    scale = torch.tensor(document["scale"], dtype=torch.float32)
    return model, mean, scale


def order(row, scores, trainer):
    safe = [index for index, action in enumerate(row["actions"]) if action["safe"]]
    pool = safe if safe and len(safe) < len(row["actions"]) else list(range(len(row["actions"])))
    return sorted(pool, key=lambda index: (scores[index], -index), reverse=True)


def evaluate(rows, model, mean, scale, trainer):
    result = {
        "roots": len(rows),
        "top1": 0,
        "top3": 0,
        "top8": 0,
        "top16": 0,
        "top32": 0,
        "unsafeSelections": 0,
        "teacherRankSum": 0,
        "teacherRankP90": 0,
    }
    ranks = []
    for row in rows:
        features = [
            trainer.action_features(row, action, False)
            for action in row["actions"]
        ]
        if model is None:
            scores = [float(trainer.baseline_score(action)) for action in row["actions"]]
        else:
            tensor = (torch.tensor(features, dtype=torch.float32) - mean) / scale
            with torch.no_grad():
                scores = model(tensor).flatten().tolist()
        ordering = order(row, scores, trainer)
        teacher = next(
            index for index, action in enumerate(row["actions"])
            if trainer.identity(action) == trainer.identity({"action": row["teacherAction"]})
        )
        rank = ordering.index(teacher) + 1 if teacher in ordering else len(ordering) + 1
        ranks.append(rank)
        selected = ordering[0]
        result["top1"] += int(rank <= 1)
        result["top3"] += int(rank <= 3)
        result["top8"] += int(rank <= 8)
        result["top16"] += int(rank <= 16)
        result["top32"] += int(rank <= 32)
        result["unsafeSelections"] += int(not row["actions"][selected]["safe"])
        result["teacherRankSum"] += rank
    result["top1Rate"] = result["top1"] / result["roots"]
    for limit in (3, 8, 16, 32):
        result[f"top{limit}Rate"] = result[f"top{limit}"] / result["roots"]
    result["teacherRankMean"] = result["teacherRankSum"] / result["roots"]
    result["teacherRankP90"] = sorted(ranks)[int(0.9 * (len(ranks) - 1))]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    trainer = load_trainer()
    rows = [
        json.loads(line)
        for line in args.targets.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model = mean = scale = None
    if args.model:
        model, mean, scale = load_model(args.model, trainer)
    for partition in ("train", "heldout"):
        subset = [row for row in rows if row["partition"] == partition]
        print(json.dumps({
            "model": str(args.model) if args.model else "heuristic-baseline",
            "partition": partition,
            "metrics": evaluate(subset, model, mean, scale, trainer),
        }, sort_keys=True))


if __name__ == "__main__":
    main()
