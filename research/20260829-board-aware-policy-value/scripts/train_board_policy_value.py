#!/usr/bin/env python3
"""Train a board-aware policy/value model on super-deep Pathfinder roots.

The target files are the source-disjoint 1M-node/depth-7 labels produced by
the super-deep contextual study. The model is intentionally trained as a
research root-ordering hint; Rust remains the legal-move and search authority.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
LAB_ROOT = REPO_ROOT / "research/20260824-gnn-cnn-lab"
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from python.game import Action, BoardConfig, GameState, Player  # type: ignore  # noqa: E402
from python.model import PathagonGNN  # type: ignore  # noqa: E402
from python.symmetry import sample_symmetry, transform_action, transform_state  # type: ignore  # noqa: E402


ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
V05_WEIGHTS = (241, 112, 887, 40, 154, 74)


def decode_radix(value: str) -> int:
    result = 0
    for character in value:
        result = result * 64 + ALPHABET.index(character)
    return result


def parse_state(text: str) -> GameState:
    fields = text.split(".")
    if len(fields) != 11:
        raise ValueError(f"state has {len(fields)} fields, expected 11")

    def optional_square(value: str) -> int | None:
        return None if value == "-" else decode_radix(value)

    turn = Player.LIGHT if fields[4] == "L" else Player.DARK
    last_player = None if fields[9] == "-" else (Player.LIGHT if fields[9] == "L" else Player.DARK)
    return GameState(
        config=BoardConfig(size=7, reserve_per_player=14, ply_limit=196),
        light=decode_radix(fields[0]),
        dark=decode_radix(fields[1]),
        reserves=(decode_radix(fields[2]), decode_radix(fields[3])),
        turn=turn,
        forbidden=decode_radix(fields[5]),
        last_relocated_to=(optional_square(fields[6]), optional_square(fields[7])),
        last_capture=decode_radix(fields[8]),
        last_player=last_player,
        winner=None,
        ply=decode_radix(fields[10]),
    )


def parse_action(value: dict[str, Any]) -> Action:
    raw = value["action"] if "action" in value else value
    if raw["kind"] == "place":
        return Action.place(int(raw["to"]))
    return Action.relocate(int(raw["from"]), int(raw["to"]))


def action_key(action: Action) -> tuple[int, int, int]:
    return action.kind, action.from_square, action.to


def teacher_score_target(score: int) -> float:
    """Map the teacher's side-to-move score into the value head's range.

    Terminal scores are intentionally saturated; non-terminal scores retain a
    smooth signal around zero. This target is auxiliary to the policy loss.
    """

    return math.tanh(float(score) / 5000.0)


def teacher_pairwise_loss(scores: torch.Tensor, teacher_index: int) -> torch.Tensor:
    """Rank the teacher action above every legal alternative."""

    if scores.numel() <= 1:
        return scores.sum() * 0.0
    teacher = scores[teacher_index]
    alternatives = torch.cat((scores[:teacher_index], scores[teacher_index + 1 :]))
    return F.softplus(-(teacher - alternatives)).mean()


def load_rows(pattern: str) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(glob.glob(pattern)):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    if not rows:
        raise ValueError(f"no rows matched {pattern}")
    return list(rows.values())


def prepare_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared = []
    for row in rows:
        state = parse_state(row["state"])
        actions = tuple(parse_action(action) for action in row["actions"])
        legal = state.legal_actions()
        if actions != legal:
            raise ValueError(f"{row['id']}: target action order does not match rules engine")
        teacher = parse_action(row["teacherAction"])
        try:
            teacher_index = actions.index(teacher)
        except ValueError as error:
            raise ValueError(f"{row['id']}: teacher action is not legal") from error
        prepared.append(
            {
                "row": row,
                "state": state,
                "actions": actions,
                "teacher_index": teacher_index,
                "value_target": teacher_score_target(int(row["teacherScore"])),
            }
        )
    return prepared


def model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return f"sha256:{digest.hexdigest()}"


def choose_policy(model: PathagonGNN, item: dict[str, Any]) -> tuple[int, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        if model.qadv:
            _policy_logits, value, logits, _centered = model.policy_value_q(
                item["state"], list(item["actions"])
            )
        else:
            logits, value = model.policy_value(item["state"], list(item["actions"]))
    index = int(torch.argmax(logits).detach().cpu())
    return index, logits, value


def safe_pool(item: dict[str, Any]) -> list[int]:
    safe = [index for index, action in enumerate(item["row"]["actions"]) if action["safe"]]
    return safe if safe and len(safe) < len(item["actions"]) else list(range(len(item["actions"])))


def offline_metrics(model: PathagonGNN | None, items: list[dict[str, Any]]) -> dict[str, Any]:
    top1 = 0
    safe_top1 = 0
    unsafe = 0
    value_errors: list[float] = []
    phase = Counter()
    turn = Counter()
    for item in items:
        if model is None:
            selected = max(
                range(len(item["actions"])),
                key=lambda index: (
                    int(item["row"]["actions"][index]["immediateWin"]),
                    int(item["row"]["actions"][index]["captureCount"]) * 10_000
                    + sum(int(feature) * weight for feature, weight in zip(item["row"]["actions"][index]["features"], V05_WEIGHTS)),
                    -index,
                ),
            )
            predicted_value = 0.0
        else:
            selected, logits, value = choose_policy(model, item)
            predicted_value = float(value.detach().cpu())
            if not math.isfinite(predicted_value):
                raise ValueError("model produced a non-finite value")
        pool = safe_pool(item)
        safe_selected = (
            max(
                pool,
                key=lambda index: (
                    float(logits[index].detach().cpu()),
                    -index,
                ),
            )
            if model is not None
            else selected
        )
        top1 += int(selected == item["teacher_index"])
        safe_top1 += int(safe_selected == item["teacher_index"])
        unsafe += int(not item["row"]["actions"][selected]["safe"])
        if model is not None:
            value_errors.append((predicted_value - item["value_target"]) ** 2)
        phase[item["row"]["phase"]] += int(selected == item["teacher_index"])
        turn[item["state"].turn.name[0]] += int(selected == item["teacher_index"])
    result: dict[str, Any] = {
        "roots": len(items),
        "top1": top1,
        "top1Rate": top1 / len(items),
        "safePoolTop1": safe_top1,
        "safePoolTop1Rate": safe_top1 / len(items),
        "unsafeSelections": unsafe,
        "byPhaseTop1": dict(phase),
        "byTurnTop1": dict(turn),
    }
    if value_errors:
        result["valueMse"] = sum(value_errors) / len(value_errors)
    return result


def train(args: argparse.Namespace) -> None:
    rows = load_rows(args.targets)
    items = prepare_rows(rows)
    train_items = [item for item in items if item["row"]["partition"] == "train"]
    heldout_items = [item for item in items if item["row"]["partition"] == "heldout"]
    if not train_items or not heldout_items:
        raise ValueError("both train and heldout roots are required")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    model = PathagonGNN(hidden_size=args.hidden, message_layers=args.layers, qadv=args.qadv).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model.train()
    policy_loss_total = 0.0
    value_loss_total = 0.0
    q_loss_total = 0.0
    rank_loss_total = 0.0
    steps = 0
    for epoch in range(args.epochs):
        order = list(range(len(train_items)))
        random.Random(args.seed * 1000 + epoch).shuffle(order)
        for index in order:
            item = train_items[index]
            if args.symmetry_augmentation:
                symmetry = sample_symmetry(random.Random(args.seed * 1_000_000 + epoch * len(order) + index))
                state = transform_state(item["state"], symmetry)
                actions = tuple(transform_action(action, item["state"].config, symmetry) for action in item["actions"])
                teacher_index = item["teacher_index"]
            else:
                state = item["state"]
                actions = item["actions"]
                teacher_index = item["teacher_index"]
            if args.qadv:
                policy_logits, value, q_values, _centered = model.policy_value_q(state, list(actions))
                logits = policy_logits
            else:
                policy_logits, value = model.policy_value(state, list(actions))
                logits = policy_logits
            policy_loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([teacher_index], device=device))
            q_loss = (
                F.cross_entropy(q_values.unsqueeze(0), torch.tensor([teacher_index], device=device))
                if args.qadv
                else policy_logits.sum() * 0.0
            )
            rank_loss = (
                teacher_pairwise_loss(q_values, teacher_index)
                if args.qadv
                else policy_logits.sum() * 0.0
            )
            value_target = torch.tensor(item["value_target"], dtype=value.dtype, device=device)
            value_loss = F.smooth_l1_loss(value, value_target)
            loss = (
                policy_loss
                + args.q_weight * q_loss
                + args.rank_weight * rank_loss
                + args.value_weight * value_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            policy_loss_total += float(policy_loss.detach().cpu())
            q_loss_total += float(q_loss.detach().cpu())
            rank_loss_total += float(rank_loss.detach().cpu())
            value_loss_total += float(value_loss.detach().cpu())
            steps += 1
        if args.progress:
            print(json.dumps({"epoch": epoch + 1, "epochs": args.epochs, "policyLoss": policy_loss_total / steps, "valueLoss": value_loss_total / steps}), flush=True)
    model.eval()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "board-policy-value.pt"
    metadata = {
        "schemaVersion": 1,
        "targetPattern": args.targets,
        "initCheckpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "hidden": args.hidden,
        "messageLayers": args.layers,
        "valueWeight": args.value_weight,
        "qWeight": args.q_weight,
        "rankWeight": args.rank_weight,
        "qadv": args.qadv,
        "symmetryAugmentation": args.symmetry_augmentation,
        "trainRoots": len(train_items),
        "heldoutRoots": len(heldout_items),
        "modelHash": model_hash(model),
        "teacherDepths": dict(Counter(item["row"]["completedDepth"] for item in items)),
    }
    torch.save({"model_config": model.config_dict(), "state_dict": model.state_dict(), "metadata": metadata}, checkpoint)
    report = {
        **metadata,
        "checkpoint": str(checkpoint),
        "loss": {
            "policy": policy_loss_total / steps,
            "q": q_loss_total / steps,
            "rank": rank_loss_total / steps,
            "value": value_loss_total / steps,
        },
        "baseline": {"train": offline_metrics(None, train_items), "heldout": offline_metrics(None, heldout_items)},
        "candidate": {"train": offline_metrics(model, train_items), "heldout": offline_metrics(model, heldout_items)},
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(checkpoint), "modelHash": metadata["modelHash"], "heldout": report["candidate"]["heldout"]}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", default=str(REPO_ROOT / "research/20260829-superdeep-contextual-evaluator/workspace/targets-turn-1m-1920-*.jsonl"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "research/20260829-board-aware-policy-value/workspace/model-gnn-v1"))
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-weight", type=float, default=0.25)
    parser.add_argument("--q-weight", type=float, default=1.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--qadv", action="store_true", help="train the board-aware transition Q/advantage head")
    parser.add_argument("--seed", type=int, default=2026082901)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--no-symmetry-augmentation", dest="symmetry_augmentation", action="store_false")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
