#!/usr/bin/env python3
"""Train an action-conditioned board model from calibrated quiet targets."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

LAB_ROOT = Path(__file__).resolve().parents[1] / "20260824-gnn-cnn-lab"
sys.path.insert(0, str(LAB_ROOT))

from python.game import Action, BoardConfig, GameState, Player  # noqa: E402
from python.model import PathagonGNN  # noqa: E402
from python.symmetry import sample_symmetry, transform_action, transform_state  # noqa: E402
from python.train import build_model, model_state_hash, save_model  # noqa: E402


ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"


def decode_radix(value: str) -> int:
    result = 0
    for character in value:
        result = result * 64 + ALPHABET.index(character)
    return result


def decode_action(token: str) -> Action:
    if len(token) != 2:
        raise ValueError(f"invalid action token {token!r}")
    code = decode_radix(token)
    if code < 49:
        return Action.place(code)
    relocation = code - 49
    return Action.relocate(relocation // 49, relocation % 49)


def decode_state(text: str) -> GameState:
    fields = text.split(".")
    if len(fields) != 11:
        raise ValueError(f"invalid compact state {text!r}")
    optional = lambda value: None if value == "-" else decode_radix(value)
    player = {"L": Player.LIGHT, "D": Player.DARK}
    config = BoardConfig(size=7, reserve_per_player=14, ply_limit=196)
    return GameState(
        config=config,
        light=decode_radix(fields[0]),
        dark=decode_radix(fields[1]),
        reserves=(decode_radix(fields[2]), decode_radix(fields[3])),
        turn=player[fields[4]],
        forbidden=decode_radix(fields[5]),
        last_relocated_to=(optional(fields[6]), optional(fields[7])),
        last_capture=decode_radix(fields[8]),
        last_player=None if fields[9] == "-" else player[fields[9]],
        winner=None,
        ply=decode_radix(fields[10]),
    )


def load_rows(path: Path) -> list[dict]:
    rows: dict[str, dict] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schemaVersion") != 1:
            raise ValueError(f"{path}:{line_number}: unsupported target schema")
        rows[row["id"]] = row
    if not rows:
        raise ValueError(f"no target rows in {path}")
    return list(rows.values())


def prepared(
    row: dict,
) -> tuple[GameState, tuple[Action, ...], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state = decode_state(row["state"])
    legal = tuple(decode_action(token) for token in row["legalActions"])
    candidates = tuple(decode_action(token) for token in row["candidateActions"])
    actual_legal = state.legal_actions()
    if tuple(actual_legal) != legal:
        raise ValueError(f"{row['id']}: Rust/Python legal action order mismatch")
    if any(action not in legal for action in candidates) or len(set(candidates)) != len(candidates):
        raise ValueError(f"{row['id']}: candidate actions are not a legal unique subset")
    values = torch.tensor(row["teacherValues"], dtype=torch.float32)
    policy = torch.tensor(row["softPolicy"], dtype=torch.float32)
    outcomes = torch.tensor(row["continuationValues"], dtype=torch.float32)
    value_target = torch.tensor(row["valueTarget"], dtype=torch.float32)
    if not (len(candidates) == len(values) == len(policy) == len(outcomes)):
        raise ValueError(f"{row['id']}: target vectors are not aligned")
    if not math.isclose(float(policy.sum()), 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError(f"{row['id']}: soft policy does not sum to one")
    if not -1.0 <= float(value_target) <= 1.0:
        raise ValueError(f"{row['id']}: calibrated value target is out of range")
    return state, candidates, values, policy, outcomes, value_target


def transformed(
    state: GameState,
    actions: tuple[Action, ...],
    symmetry_augmentation: bool,
    rng: random.Random,
) -> tuple[GameState, tuple[Action, ...]]:
    if not symmetry_augmentation:
        return state, actions
    symmetry = sample_symmetry(rng)
    return transform_state(state, symmetry), tuple(
        transform_action(action, state.config, symmetry) for action in actions
    )


def pairwise_rank_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for left in range(len(target)):
        for right in range(left + 1, len(target)):
            difference = target[left] - target[right]
            if abs(float(difference.detach())) < 1.0e-6:
                continue
            losses.append(F.softplus(-torch.sign(difference) * (predicted[left] - predicted[right])))
    return torch.stack(losses).mean() if losses else predicted.sum() * 0.0


def train(
    model: PathagonGNN,
    rows: list[dict],
    steps: int,
    learning_rate: float,
    seed: int,
    outcome_weight: float,
    value_weight: float,
    symmetry_augmentation: bool,
) -> dict[str, float]:
    if not rows:
        raise ValueError("training rows are empty")
    prepared_rows = [prepared(row) for row in rows]
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    model.train()
    totals = Counter()
    for _step in range(steps):
        row_index = rng.randrange(len(rows))
        state, actions, teacher_values, soft_policy, outcomes, value_target = prepared_rows[row_index]
        state, actions = transformed(state, actions, symmetry_augmentation, rng)
        logits, value, q_values, advantages = model.policy_value_q(state, list(actions))
        teacher_values = teacher_values.to(logits.device)
        soft_policy = soft_policy.to(logits.device)
        outcomes = outcomes.to(logits.device)
        value_target = value_target.to(logits.device)
        policy_loss = -(soft_policy * F.log_softmax(logits, dim=0)).sum()
        q_loss = F.smooth_l1_loss(q_values, teacher_values)
        advantage_target = teacher_values - teacher_values.mean()
        advantage_loss = F.smooth_l1_loss(advantages, advantage_target)
        rank_loss = pairwise_rank_loss(q_values, teacher_values)
        outcome_loss = F.smooth_l1_loss(q_values, outcomes)
        value_loss = F.mse_loss(value, value_target)
        loss = (
            policy_loss
            + q_loss
            + 0.5 * advantage_loss
            + 0.25 * rank_loss
            + outcome_weight * outcome_loss
            + value_weight * value_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        for key, item in {
            "loss": loss,
            "policy_loss": policy_loss,
            "q_loss": q_loss,
            "advantage_loss": advantage_loss,
            "rank_loss": rank_loss,
            "outcome_loss": outcome_loss,
            "value_loss": value_loss,
        }.items():
            totals[key] += float(item.detach().cpu())
    return {key: value / steps for key, value in totals.items()}


@torch.no_grad()
def evaluate(model: PathagonGNN, rows: list[dict]) -> dict:
    model.eval()
    q_abs = 0.0
    value_abs = 0.0
    pairwise_correct = 0
    pairwise_total = 0
    teacher_top = 0
    outcome_top = 0
    outcome_roots = 0
    phase_counts: Counter[str] = Counter()
    target_class_counts: Counter[str] = Counter()
    value_bias = 0.0
    value_squared = 0.0
    class_abs: Counter[str] = Counter()
    class_count: Counter[str] = Counter()
    for row in rows:
        state, actions, teacher_values, soft_policy, outcomes, value_target = prepared(row)
        logits, value, q_values, _advantages = model.policy_value_q(state, list(actions))
        teacher_values = teacher_values.to(q_values.device)
        soft_policy = soft_policy.to(logits.device)
        outcomes = outcomes.to(q_values.device)
        value_target = value_target.to(value.device)
        q_abs += float(torch.abs(q_values - teacher_values).mean().cpu())
        error = float(value.cpu()) - float(value_target.cpu())
        value_abs += abs(error)
        value_bias += error
        value_squared += error * error
        target_class = row.get("targetClass", "unknown")
        target_class_counts[target_class] += 1
        class_abs[target_class] += abs(error)
        class_count[target_class] += 1
        teacher_top += int(int(q_values.argmax()) == int(teacher_values.argmax()))
        outcome_best = outcomes.max()
        if bool(torch.any(outcomes != outcomes[0])):
            outcome_roots += 1
            outcome_top += int(outcomes[int(q_values.argmax())] == outcome_best)
        for left in range(len(teacher_values)):
            for right in range(left + 1, len(teacher_values)):
                difference = teacher_values[left] - teacher_values[right]
                if abs(float(difference.cpu())) < 1.0e-6:
                    continue
                pairwise_total += 1
                pairwise_correct += int(torch.sign(q_values[left] - q_values[right]) == torch.sign(difference))
        phase_counts[row["phase"]] += 1
    count = max(1, len(rows))
    return {
        "rows": len(rows),
        "qMae": q_abs / count,
        "valueMae": value_abs / count,
        "valueBias": value_bias / count,
        "valueRmse": math.sqrt(value_squared / count),
        "valueMaeByClass": {
            key: class_abs[key] / class_count[key] for key in sorted(class_count)
        },
        "teacherTop1": teacher_top / count,
        "pairwise": pairwise_correct / max(1, pairwise_total),
        "outcomeTop": outcome_top / max(1, outcome_roots),
        "outcomeRoots": outcome_roots,
        "phaseCounts": dict(sorted(phase_counts.items())),
        "targetClassCounts": dict(sorted(target_class_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--outcome-weight", type=float, default=0.5)
    parser.add_argument("--value-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-symmetry", action="store_true")
    parser.add_argument("--value-only", action="store_true", help="fit only the state value head")
    parser.add_argument(
        "--action-head-only",
        action="store_true",
        help="freeze the board encoder/value head and fit policy/Q action heads",
    )
    args = parser.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be positive")
    device = torch.device("mps" if args.device == "auto" and torch.backends.mps.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    rows = load_rows(args.targets)
    train_rows = [row for row in rows if row["partition"] == "train"]
    heldout_rows = [row for row in rows if row["partition"] == "heldout"]
    if not train_rows or not heldout_rows:
        raise SystemExit("targets need both train and heldout game partitions")
    if args.resume:
        from python.train import load_model

        model = load_model(args.resume, device, qadv=True)
    else:
        model = build_model("gnn", 64, 8, 4, 7, qadv=True).to(device)
    if not isinstance(model, PathagonGNN) or not model.qadv:
        raise SystemExit("quiet value training requires a QAdv-enabled GNN")
    if args.value_only and args.action_head_only:
        raise SystemExit("--value-only and --action-head-only are mutually exclusive")
    if args.value_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("value_head.")
    if args.action_head_only:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "head" in name and not name.startswith("value_head.")
    before = evaluate(model, heldout_rows)
    losses = train(
        model,
        train_rows,
        args.steps,
        args.learning_rate,
        args.seed,
        args.outcome_weight,
        args.value_weight,
        not args.no_symmetry,
    )
    after = evaluate(model, heldout_rows)
    metadata = {
        "mode": "relative-regret-action-head",
        "data": str(args.targets),
        "rows": len(rows),
        "trainRows": len(train_rows),
        "heldoutRows": len(heldout_rows),
        "trainGames": len({row["gameKey"] for row in train_rows}),
        "heldoutGames": len({row["gameKey"] for row in heldout_rows}),
        "steps": args.steps,
        "learningRate": args.learning_rate,
        "outcomeWeight": args.outcome_weight,
        "valueWeight": args.value_weight,
        "seed": args.seed,
        "symmetryAugmentation": not args.no_symmetry,
        "valueOnly": args.value_only,
        "actionHeadOnly": args.action_head_only,
        "architecture": model.config_dict(),
        "before": before,
        "after": after,
        "losses": losses,
        "modelHash": model_state_hash(model),
    }
    save_model(model, args.output, metadata)
    print(json.dumps(metadata | {"output": str(args.output), "device": str(device)}, sort_keys=True))


if __name__ == "__main__":
    main()
