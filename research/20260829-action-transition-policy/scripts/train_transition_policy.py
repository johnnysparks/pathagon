#!/usr/bin/env python3
"""Train a compact action scorer from the million-node Pathfinder labels."""

from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

import torch


BASELINE = (241, 112, 887, 40, 154, 74)
FEATURE_NAMES = (
    "path", "material", "capture", "structure", "threat", "edge",
    "capture_count", "immediate_win", "safe", "relocate", "to_row",
    "to_column", "from_row", "from_column", "own_progress", "own_from_progress",
    "center_distance", "edge_square", "corner_square", "dark_to_move",
    "own_pieces", "opponent_pieces", "own_reserve", "opponent_reserve",
    "legal_action_fraction", "last_capture", "ply_fraction", "last_player_same",
    "opening_phase", "placement_phase", "movement_phase", "late_phase",
)


def identity(action: dict) -> tuple:
    value = action["action"]
    return value.get("kind"), value.get("from"), value.get("to")


def decode_radix(value: str) -> int:
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
    result = 0
    for character in value:
        result = result * 64 + alphabet.index(character)
    return result


def action_features(row: dict, action: dict, virtual_source: bool) -> list[float]:
    value = action["action"]
    relocate = float(value.get("kind") == "relocate")
    destination = int(value["to"])
    to_row, to_column = divmod(destination, 7)
    if relocate:
        source = int(value["from"])
        from_row, from_column = divmod(source, 7)
    else:
        from_row = from_column = 7 if virtual_source else -1
    dark = row["state"].split(".")[4] == "D"
    state_fields = row["state"].split(".")
    light_mask = decode_radix(state_fields[0])
    dark_mask = decode_radix(state_fields[1])
    light_pieces = light_mask.bit_count()
    dark_pieces = dark_mask.bit_count()
    light_reserve = decode_radix(state_fields[2])
    dark_reserve = decode_radix(state_fields[3])
    own_pieces, opponent_pieces = (dark_pieces, light_pieces) if dark else (light_pieces, dark_pieces)
    own_reserve, opponent_reserve = (dark_reserve, light_reserve) if dark else (light_reserve, dark_reserve)
    last_capture = decode_radix(state_fields[8])
    ply = decode_radix(state_fields[10])
    occupied = light_pieces + dark_pieces
    reserves = light_reserve + dark_reserve
    phase_flags = (
        float(occupied < 8),
        float(occupied >= 8 and reserves > 0 and occupied < 20),
        float(reserves == 0),
        float(occupied >= 20 and reserves > 0),
    )
    own_progress = (6 - to_row if not dark else to_column) / 6.0
    own_from_progress = ((6 - from_row if not dark else from_column) / 6.0) if relocate else 0.0
    edge = float(to_row in (0, 6) or to_column in (0, 6))
    corner = float(to_row in (0, 6) and to_column in (0, 6))
    return [
        *(float(item) for item in action["features"]),
        float(action["captureCount"]) / 4.0,
        float(action["immediateWin"]),
        float(action["safe"]),
        relocate,
        to_row / 6.0,
        to_column / 6.0,
        from_row / 6.0 if relocate or virtual_source else 0.0,
        from_column / 6.0 if relocate or virtual_source else 0.0,
        own_progress,
        own_from_progress,
        (abs(to_row - 3) + abs(to_column - 3)) / 6.0,
        edge,
        corner,
        float(dark),
        own_pieces / 14.0,
        opponent_pieces / 14.0,
        own_reserve / 14.0,
        opponent_reserve / 14.0,
        len(row["actions"]) / 2401.0,
        last_capture / 4.0,
        ply / 196.0,
        float(state_fields[9] == ("D" if dark else "L")),
        *phase_flags,
    ]


def load_rows(pattern: str) -> list[dict]:
    rows: dict[str, dict] = {}
    for path in sorted(glob.glob(pattern)):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    if not rows:
        raise ValueError(f"no target rows matched {pattern}")
    return list(rows.values())


def baseline_index(row: dict) -> int:
    return max(
        range(len(row["actions"])),
        key=lambda index: (baseline_score(row["actions"][index]), -index),
    )


def baseline_score(action: dict) -> int:
    if action["immediateWin"]:
        return 2_000_000_000
    return int(action["captureCount"]) * 10_000 + sum(
        int(feature) * weight for feature, weight in zip(action["features"], BASELINE)
    )


def make_model(input_size: int, hidden: int) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, 1),
    )


def choose_indices(model: torch.nn.Module, rows: list[dict], features: list[torch.Tensor]) -> list[int]:
    choices = []
    with torch.no_grad():
        for row, action_features_tensor in zip(rows, features):
            scores = model(action_features_tensor).flatten()
            safe = [index for index, action in enumerate(row["actions"]) if action["safe"]]
            pool = safe if safe and len(safe) < len(row["actions"]) else list(range(len(row["actions"])))
            choices.append(max(pool, key=lambda index: (float(scores[index]), -index)))
    return choices


def ranked_indices(model: torch.nn.Module | None, row: dict, feature_tensor: torch.Tensor) -> list[int]:
    """Return the model's legal/tactical-safe ordering for one root."""
    if model is None:
        scores = [float(baseline_score(action)) for action in row["actions"]]
    else:
        with torch.no_grad():
            scores = model(feature_tensor).flatten().tolist()
    safe = [index for index, action in enumerate(row["actions"]) if action["safe"]]
    pool = safe if safe and len(safe) < len(row["actions"]) else list(range(len(row["actions"])))
    return sorted(pool, key=lambda index: (scores[index], -index), reverse=True)


def teacher_pairwise_loss(scores: torch.Tensor, teacher_index: int, margin: float = 0.25) -> torch.Tensor:
    """Make the labeled teacher action outrank every legal alternative."""
    teacher = scores[teacher_index]
    alternatives = torch.cat((scores[:teacher_index], scores[teacher_index + 1:]))
    if alternatives.numel() == 0:
        return scores.new_zeros(())
    return torch.nn.functional.relu(margin - teacher + alternatives).mean()


def metrics(model: torch.nn.Module | None, rows: list[dict], features: list[torch.Tensor], teacher_indices: list[int]) -> dict:
    if model is None:
        selected = [baseline_index(row) for row in rows]
    else:
        selected = choose_indices(model, rows, features)
    result = {
        "roots": len(rows),
        "top1": 0,
        "top3": 0,
        "unsafeSelections": 0,
        "teacherRankSum": 0,
        "teacherRankP90": 0,
        "byPhase": {},
        "byTurn": {},
    }
    ranks = []
    for row, index, teacher, feature_tensor in zip(rows, selected, teacher_indices, features):
        ordering = ranked_indices(model, row, feature_tensor)
        rank = ordering.index(teacher) + 1 if teacher in ordering else len(ordering) + 1
        ranks.append(rank)
        result["top1"] += int(index == teacher)
        result["top3"] += int(rank <= 3)
        result["teacherRankSum"] += rank
        result["unsafeSelections"] += int(not row["actions"][index]["safe"])
        for name, key in ((row["phase"], "byPhase"), (row["state"].split(".")[4], "byTurn")):
            bucket = result[key].setdefault(name, {"roots": 0, "top1": 0})
            bucket["roots"] += 1
            bucket["top1"] += int(index == teacher)
    result["top1Rate"] = result["top1"] / result["roots"] if result["roots"] else 0.0
    result["top3Rate"] = result["top3"] / result["roots"] if result["roots"] else 0.0
    result["teacherRankMean"] = result["teacherRankSum"] / result["roots"] if result["roots"] else 0.0
    if ranks:
        result["teacherRankP90"] = sorted(ranks)[int(0.9 * (len(ranks) - 1))]
    return result


def tensor_to_nested(tensor: torch.Tensor) -> list[list[float]]:
    return tensor.detach().cpu().tolist()


def train(args: argparse.Namespace) -> None:
    rows = load_rows(args.targets)
    if args.min_completed_depth > 0:
        rows = [row for row in rows if row["completedDepth"] >= args.min_completed_depth]
    train_rows = [row for row in rows if row["partition"] == "train"]
    heldout_rows = [row for row in rows if row["partition"] == "heldout"]
    if not train_rows or not heldout_rows:
        raise ValueError("both train and heldout roots are required")
    all_features = [[action_features(row, action, args.virtual_source) for action in row["actions"]] for row in rows]
    train_flat = torch.tensor(
        [feature for row, features in zip(rows, all_features) if row["partition"] == "train" for feature in features],
        dtype=torch.float32,
    )
    mean = train_flat.mean(dim=0)
    scale = train_flat.std(dim=0).clamp_min(0.2)
    tensors = [
        (torch.tensor(features, dtype=torch.float32) - mean) / scale for features in all_features
    ]
    teacher_indices = [
        next(index for index, action in enumerate(row["actions"]) if identity(action) == identity({"action": row["teacherAction"]}))
        for row in rows
    ]
    train_indices = [index for index, row in enumerate(rows) if row["partition"] == "train"]
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = make_model(len(FEATURE_NAMES), args.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model.train()
    losses = []
    for epoch in range(args.epochs):
        random.Random(args.seed * 1000 + epoch).shuffle(train_indices)
        epoch_loss = 0.0
        for index in train_indices:
            scores = model(tensors[index]).flatten()
            teacher_index = teacher_indices[index]
            loss = torch.nn.functional.cross_entropy(scores.unsqueeze(0), torch.tensor([teacher_index]))
            if args.rank_weight > 0.0:
                loss = loss + args.rank_weight * teacher_pairwise_loss(scores, teacher_index, args.rank_margin)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach())
        losses.append(epoch_loss / len(train_indices))
    model.eval()
    ordered_train_rows = train_rows
    ordered_train_features = [tensors[index] for index, row in enumerate(rows) if row["partition"] == "train"]
    ordered_train_teacher = [teacher_indices[index] for index, row in enumerate(rows) if row["partition"] == "train"]
    ordered_heldout_features = [tensors[index] for index, row in enumerate(rows) if row["partition"] == "heldout"]
    ordered_heldout_teacher = [teacher_indices[index] for index, row in enumerate(rows) if row["partition"] == "heldout"]
    report = {
        "schemaVersion": 1,
        "model": "tanh-unified-move-policy-v2" if args.virtual_source else "tanh-action-state-transition-policy-v2",
        "seed": args.seed,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "learningRate": args.learning_rate,
        "weightDecay": args.weight_decay,
        "rankWeight": args.rank_weight,
        "rankMargin": args.rank_margin,
        "minCompletedDepth": args.min_completed_depth,
        "featureOrder": list(FEATURE_NAMES),
        "training": {"roots": len(train_rows), "actions": int(train_flat.shape[0])},
        "heldout": {"roots": len(heldout_rows)},
        "loss": {"first": losses[0], "last": losses[-1]},
        "baseline": {
            "train": metrics(None, train_rows, ordered_train_features, ordered_train_teacher),
            "heldout": metrics(None, heldout_rows, ordered_heldout_features, ordered_heldout_teacher),
        },
        "candidate": {
            "train": metrics(model, train_rows, ordered_train_features, ordered_train_teacher),
            "heldout": metrics(model, heldout_rows, ordered_heldout_features, ordered_heldout_teacher),
        },
    }
    layers = []
    for layer in model:
        if isinstance(layer, torch.nn.Linear):
            layers.append({"weights": tensor_to_nested(layer.weight), "bias": layer.bias.detach().cpu().tolist()})
    document = {
        "schemaVersion": 1,
        "model": "tanh-unified-move-policy-v2" if args.virtual_source else "tanh-action-state-transition-policy-v2",
        "encoding": "virtual-offboard-source" if args.virtual_source else "explicit-source-kind",
        "featureOrder": list(FEATURE_NAMES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "layers": layers,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "transition-policy.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"train": report["candidate"]["train"], "heldout": report["candidate"]["heldout"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.002)
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--rank-margin", type=float, default=0.25)
    parser.add_argument("--min-completed-depth", type=int, default=0)
    parser.add_argument("--virtual-source", action="store_true", help="encode placement with a virtual off-board source")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
