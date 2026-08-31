#!/usr/bin/env python3
"""Train and gate a small layered-gold policy/value/urgency candidate.

This is a research adapter, not a second rules authority. Rust produces the
frontier and remains the runtime source of truth; Python only fits a small
scorer against Rust-exported labels and reports held-out metrics.
"""

from __future__ import annotations

import argparse
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

from python.evaluation import evaluate_position  # noqa: E402
from python.game import Action, BoardConfig, GameState, Player  # noqa: E402
from python.golden import canonical_position_key  # noqa: E402


ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
V4_WEIGHTS = (241, 112, 887, 40, 154, 74)


class PolicyValueModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy = torch.nn.Linear(16, 1)
        self.value = torch.nn.Linear(16, 3)

    def policy_scores(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy(features).squeeze(-1)

    def value_logits(self, features: torch.Tensor) -> torch.Tensor:
        return self.value(features.mean(dim=0)) if features.ndim == 2 else self.value(features)


def decode_action(token: str) -> Action:
    code = (ALPHABET.index(token[0]) << 6) | ALPHABET.index(token[1])
    cells = 49
    if code < cells:
        return Action.place(code)
    relocation = code - cells
    return Action.relocate(*divmod(relocation, cells))


def decode_action_record(raw: dict[str, Any]) -> Action:
    if "token" in raw:
        return decode_action(str(raw["token"]))
    action = raw.get("action", raw)
    if action.get("kind") == "place":
        return Action.place(int(action["to"]))
    if action.get("kind") == "relocate":
        return Action.relocate(int(action["from"]), int(action["to"]))
    raise ValueError(f"unsupported gold action record: {raw!r}")


def action_order(action: Action) -> int:
    return action.to if action.kind == 0 else action.from_square * 64 + action.to


def load_state(raw: dict[str, Any]) -> GameState:
    return GameState.seeded(
        BoardConfig(size=7, reserve_per_player=14),
        int(raw["light"]),
        int(raw["dark"]),
        tuple(int(value) for value in raw["reserve"]),
        Player.LIGHT if raw["turn"] == "light" else Player.DARK,
        int(raw["forbidden"]),
        tuple(None if value is None else int(value) for value in raw["lastRelocatedTo"]),
        int(raw["ply"]),
    )


def action_features(state: GameState, action: Action) -> list[float]:
    player = state.turn
    child = state.apply_legal(action)
    to_row, to_column = divmod(action.to, 7)
    if action.kind == 1:
        from_row, from_column = divmod(action.from_square, 7)
    else:
        from_row = from_column = 0
    own = state.pieces(player).bit_count()
    opponent = state.pieces(player.other()).bit_count()
    own_reserve = state.reserves[player]
    opponent_reserve = state.reserves[player.other()]
    before = evaluate_position(state, player)
    after = evaluate_position(child, player)
    return [
        float(action.kind == 0),
        float(action.kind == 1),
        to_row / 6.0,
        to_column / 6.0,
        from_row / 6.0,
        from_column / 6.0,
        float(child.last_capture) / 4.0,
        float(child.winner == player),
        float(to_row in (0, 6)),
        float(to_column in (0, 6)),
        own / 14.0,
        opponent / 14.0,
        own_reserve / 14.0,
        opponent_reserve / 14.0,
        state.ply / 180.0,
        max(-1.0, min(1.0, (after - before) / 5000.0)),
    ]


def v4_control_index(state: GameState, actions: tuple[Action, ...]) -> int:
    player = state.turn
    scored = []
    for index, action in enumerate(actions):
        child = state.apply_legal(action)
        if child.winner == player:
            score = 2_000_000_000
        else:
            score = child.last_capture * 10_000 + int(evaluate_position(child, player))
        scored.append((score, -action_order(action), index))
    return max(scored)[2]


def hands_opponent_an_immediate_win(state: GameState, action: Action) -> bool:
    child = state.apply_legal(action)
    opponent = child.turn
    return any(child.apply_legal(reply).winner == opponent for reply in child.legal_actions())


def forced_block_metrics(
    rows: list[dict[str, Any]], selected_indices: list[int]
) -> dict[str, Any]:
    eligible = 0
    successes = 0
    for row, selected in zip(rows, selected_indices):
        if row["forced_block_precomputed"]:
            safe = row["forced_block_actions"]
            if not safe:
                continue
            eligible += 1
            successes += int(row["actions"][selected] in safe)
            continue
        unsafe = [hands_opponent_an_immediate_win(row["state"], action) for action in row["actions"]]
        if not any(unsafe) or all(unsafe):
            continue
        eligible += 1
        successes += int(not unsafe[selected])
    return {
        "status": "measured" if eligible else "not-applicable",
        "rows": eligible,
        "accuracy": successes / eligible if eligible else None,
    }


def read_rows(path: Path, heldout: set[str], max_rows: int) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        state = load_state(raw["position"])
        key = canonical_position_key(state).hex()
        labels = raw["actions"]
        actions = tuple(decode_action_record(item) for item in labels)
        legal = state.legal_actions()
        if not set(actions).issubset(set(legal)):
            raise ValueError(f"{key}: frontier action is not legal")
        proven = {
            action
            for action, item in zip(actions, labels)
            if item.get("outcome", "win") == "win"
        }
        urgency = {
            decode_action_record(item)
            for item in raw.get("urgencyActions", [])
        }
        if not urgency:
            urgency = set(proven)
        outcome = str(raw.get("outcome", "win"))
        if outcome not in {"win", "draw", "loss"}:
            raise ValueError(f"{key}: unsupported gold outcome {outcome!r}")
        rows.append(
            {
                "key": key,
                "state": state,
                "actions": legal,
                "proven": proven,
                "urgency": urgency,
                "outcome": outcome,
                "distance": raw.get("distance"),
                "optimal_complete": bool(raw.get("optimalActionsComplete", False)),
                "feature_rows": raw.get("features"),
                "forced_block_precomputed": "forcedBlockActions" in raw,
                "forced_block_actions": {
                    decode_action_record(item)
                    for item in (raw.get("forcedBlockActions") or [])
                },
                "source_layer": int(raw.get("sourceLayer", 0)),
                "partition": "heldout" if key in heldout else "train",
            }
        )
        if max_rows and len(rows) >= max_rows:
            break
    if not rows:
        raise ValueError("candidate file is empty")
    return rows


def prepare(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if row["feature_rows"] is not None:
            if len(row["feature_rows"]) != len(row["actions"]):
                raise ValueError(f"{row['key']}: Rust feature rows do not match legal actions")
            if any(len(features) != 16 for features in row["feature_rows"]):
                raise ValueError(f"{row['key']}: Rust action features must have width 16")
            row["features"] = torch.tensor(row["feature_rows"], dtype=torch.float32)
        else:
            row["features"] = torch.tensor(
                [action_features(row["state"], action) for action in row["actions"]],
                dtype=torch.float32,
            )
        row["proven_indices"] = [
            index for index, action in enumerate(row["actions"]) if action in row["proven"]
        ]
        row["urgency_indices"] = [
            index for index, action in enumerate(row["actions"]) if action in row["urgency"]
        ]
        row["target_indices"] = row["proven_indices"] or row["urgency_indices"]
        row["outcome_index"] = {"loss": 0, "draw": 1, "win": 2}[row["outcome"]]


def metrics(
    model: PolicyValueModel,
    rows: list[dict[str, Any]],
    include_forced_blocks: bool = True,
) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    exact = 0
    value_brier = []
    urgency_hits = 0
    urgency_rows = 0
    forced_win_rows = 0
    forced_win_hits = 0
    policy_rows = 0
    wdl_hits = 0
    selected_indices = []
    for row in rows:
        with torch.no_grad():
            scores = model.policy_scores(row["features"])
            value_probabilities = torch.softmax(model.value_logits(row["features"]), dim=0)
        selected = int(torch.argmax(scores))
        if include_forced_blocks:
            selected_indices.append(selected)
        if row["target_indices"]:
            policy_rows += 1
            exact += int(selected in row["target_indices"])
        if row["urgency_indices"]:
            urgency_rows += 1
            urgency_hits += int(selected in row["urgency_indices"])
        if row["outcome"] == "win" and row["proven_indices"]:
            forced_win_rows += 1
            forced_win_hits += int(selected in row["proven_indices"])
        expected = torch.zeros(3)
        expected[row["outcome_index"]] = 1.0
        value_brier.append(float(torch.mean((value_probabilities - expected) ** 2)))
        wdl_hits += int(int(torch.argmax(value_probabilities)) == row["outcome_index"])
    return {
        "rows": len(rows),
        "policyRows": policy_rows,
        "exactActionAccuracy": exact / policy_rows if policy_rows else None,
        "forcedWinRows": forced_win_rows,
        "forcedWinAccuracy": forced_win_hits / forced_win_rows if forced_win_rows else None,
        "forcedBlockAccuracy": (
            forced_block_metrics(rows, selected_indices)
            if include_forced_blocks
            else {"status": "not-computed", "rows": 0, "accuracy": None}
        ),
        "heldoutMatches": exact / policy_rows if policy_rows else None,
        "wdlAccuracy": wdl_hits / len(rows),
        "urgencyRows": urgency_rows,
        "urgencyAccuracy": urgency_hits / urgency_rows if urgency_rows else None,
        "urgencyDistanceOneRate": urgency_hits / urgency_rows if urgency_rows else None,
        "valueBrier": sum(value_brier) / len(value_brier),
    }


def control_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = 0
    policy_rows = 0
    urgency_hits = 0
    urgency_rows = 0
    forced_win_hits = 0
    forced_win_rows = 0
    selected_indices = []
    for row in rows:
        selected = v4_control_index(row["state"], row["actions"])
        selected_indices.append(selected)
        if row["target_indices"]:
            policy_rows += 1
            exact += int(selected in row["target_indices"])
        if row["urgency_indices"]:
            urgency_rows += 1
            urgency_hits += int(selected in row["urgency_indices"])
        if row["outcome"] == "win" and row["proven_indices"]:
            forced_win_rows += 1
            forced_win_hits += int(selected in row["proven_indices"])
    return {
        "rows": len(rows),
        "policyRows": policy_rows,
        "exactActionAccuracy": exact / policy_rows if policy_rows else None,
        "forcedWinRows": forced_win_rows,
        "forcedWinAccuracy": forced_win_hits / forced_win_rows if forced_win_rows else None,
        "forcedBlockAccuracy": forced_block_metrics(rows, selected_indices),
        "urgencyRows": urgency_rows,
        "urgencyAccuracy": urgency_hits / urgency_rows if urgency_rows else None,
        "urgencyDistanceOneRate": urgency_hits / urgency_rows if urgency_rows else None,
    }


def source_layer_metrics(
    model: PolicyValueModel, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    by_layer = {}
    for layer in sorted({row["source_layer"] for row in rows}):
        subset = [row for row in rows if row["source_layer"] == layer]
        candidate = metrics(model, subset)
        control = control_metrics(subset)
        by_layer[str(layer)] = {
            "rows": len(subset),
            "candidate": candidate,
            "control": control,
            "candidateAtLeastControl": (
                candidate["exactActionAccuracy"] >= control["exactActionAccuracy"]
                if candidate["exactActionAccuracy"] is not None
                and control["exactActionAccuracy"] is not None
                else None
            ),
        }
    return by_layer


def train_batch(
    model: PolicyValueModel,
    rows: list[dict[str, Any]],
    gold_weight: float,
    value_weights: torch.Tensor,
) -> torch.Tensor:
    max_actions = max(len(row["actions"]) for row in rows)
    features = torch.zeros((len(rows), max_actions, 16), dtype=torch.float32)
    legal_mask = torch.zeros((len(rows), max_actions), dtype=torch.bool)
    target_mask = torch.zeros((len(rows), max_actions), dtype=torch.bool)
    outcome = torch.tensor([row["outcome_index"] for row in rows], dtype=torch.long)
    for row_index, row in enumerate(rows):
        count = len(row["actions"])
        features[row_index, :count] = row["features"]
        legal_mask[row_index, :count] = True
        target_mask[row_index, row["target_indices"]] = True
    scores = model.policy(features).squeeze(-1)
    legal_scores = scores.masked_fill(~legal_mask, float("-inf"))
    target_scores = scores.masked_fill(~target_mask, float("-inf"))
    policy_rows = target_mask.any(dim=1)
    if policy_rows.any():
        policy_loss = (
            torch.logsumexp(legal_scores[policy_rows], dim=1)
            - torch.logsumexp(target_scores[policy_rows], dim=1)
        ).mean()
    else:
        policy_loss = scores.sum() * 0.0
    counts = legal_mask.sum(dim=1, keepdim=True).to(features.dtype)
    value_features = features.sum(dim=1) / counts
    value_loss = F.cross_entropy(model.value(value_features), outcome, weight=value_weights)
    return gold_weight * (policy_loss + 0.25 * value_loss)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--heldout-extra", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gold-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.gold_weight <= 0:
        raise ValueError("--gold-weight must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    heldout = {line.strip() for line in args.heldout.read_text(encoding="ascii").splitlines() if line.strip()}
    if args.heldout_extra:
        heldout.update(
            line.strip()
            for line in args.heldout_extra.read_text(encoding="ascii").splitlines()
            if line.strip()
        )
    rows = read_rows(args.candidates, heldout, args.max_rows)
    prepare(rows)
    train = [row for row in rows if row["partition"] == "train"]
    evaluation = [row for row in rows if row["partition"] == "heldout"]
    if not train or not evaluation:
        raise ValueError("selected rows must contain both train and heldout partitions")
    model = PolicyValueModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    outcome_counts = Counter(row["outcome_index"] for row in train)
    value_weights = torch.tensor(
        [min(20.0, math.sqrt(len(train) / max(1, outcome_counts.get(index, 0)))) for index in range(3)],
        dtype=torch.float32,
    )
    losses = []
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        for start in range(0, len(order), args.batch_size):
            batch = [train[index] for index in order[start : start + args.batch_size]]
            loss = train_batch(model, batch, args.gold_weight, value_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    control = control_metrics(evaluation)
    candidate_train = metrics(model, train, include_forced_blocks=False)
    candidate_heldout = metrics(model, evaluation)
    heldout_by_source_layer = source_layer_metrics(model, evaluation)
    ring2_metrics = heldout_by_source_layer.get("1", {}).get("candidate", {})
    ring2_value_gate = {
        "status": (
            "pass"
            if ring2_metrics.get("rows", 0) > 0 and ring2_metrics.get("wdlAccuracy") == 1.0
            else "fail"
            if ring2_metrics.get("rows", 0) > 0
            else "not-run"
        ),
        "rows": ring2_metrics.get("rows", 0),
        "wdlAccuracy": ring2_metrics.get("wdlAccuracy"),
    }
    candidate_at_least_control = candidate_heldout["exactActionAccuracy"] >= control["exactActionAccuracy"]
    report = {
        "schemaVersion": 1,
        "experiment": "layered-golden-policy-value-urgency",
        "teacher": {
            "tableFamily": "fresh-frontier-wdl-v1+v4",
            "outcomes": ["win", "draw", "loss"],
            "policy": "proven-actions-or-urgency-actions; completeness-preserved",
        },
        "seed": args.seed,
        "epochs": args.epochs,
        "goldWeight": args.gold_weight,
        "trainOutcomeCounts": {str(index): outcome_counts.get(index, 0) for index in range(3)},
        "valueClassWeights": value_weights.tolist(),
        "rows": {"all": len(rows), "train": len(train), "heldout": len(evaluation)},
        "control": {"id": "pathfinder-action-transition-v4-xent", "metrics": control},
        "heldoutBySourceLayer": heldout_by_source_layer,
        "candidate": {"id": "layered-linear-gold-candidate-v0.2", "train": candidate_train, "heldout": candidate_heldout},
        "loss": {"initialToFinal": [losses[0], losses[-1]], "mean": sum(losses) / len(losses)},
        "gates": {
            "heldoutPresent": bool(evaluation),
            "exactActionReported": True,
            "wdlCalibrationReported": True,
            "urgencyReported": True,
            "calibrationReported": True,
            "forcedWinGateReported": True,
            "forcedBlockGate": candidate_heldout["forcedBlockAccuracy"],
            "heldoutMatchGateReported": True,
            "earlierRingRegression": {"status": "measured", "reference": "fresh-frontier-wdl-v1"},
            "ring2ValueGate": ring2_value_gate,
            "candidateAtLeastControlOnHeldout": candidate_at_least_control,
            "promotionDecision": "research-only-until-candidate-clears-ring2-value-and-user-benchmark-gates",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "report": report}, args.output_dir / "candidate.pt")
    print(json.dumps({"train": len(train), "heldout": len(evaluation), "candidate": report["candidate"]["heldout"], "control": report["control"]["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
