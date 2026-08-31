#!/usr/bin/env python3
"""Train and gate a small gold-aware policy/value/urgency candidate.

This is a research adapter, not a second rules authority. Rust produces the
frontier and remains the runtime source of truth; Python only prepares a
small scorer against the verified Ring-1 labels and reports held-out metrics.
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
        return self.value(features.mean(dim=0))


def decode_action(token: str) -> Action:
    code = (ALPHABET.index(token[0]) << 6) | ALPHABET.index(token[1])
    cells = 49
    if code < cells:
        return Action.place(code)
    relocation = code - cells
    return Action.relocate(*divmod(relocation, cells))


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
        actions = tuple(decode_action(item["token"]) for item in raw["actions"])
        legal = state.legal_actions()
        if not set(actions).issubset(set(legal)):
            raise ValueError(f"{key}: frontier action is not legal")
        rows.append(
            {
                "key": key,
                "state": state,
                "actions": legal,
                "proven": set(actions),
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
        row["features"] = torch.tensor(
            [action_features(row["state"], action) for action in row["actions"]],
            dtype=torch.float32,
        )
        row["proven_indices"] = [
            index for index, action in enumerate(row["actions"]) if action in row["proven"]
        ]


def metrics(model: PolicyValueModel, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"rows": 0}
    exact = 0
    value_brier = []
    urgency_hits = 0
    selected_indices = []
    for row in rows:
        with torch.no_grad():
            scores = model.policy_scores(row["features"])
            value_probabilities = torch.softmax(model.value_logits(row["features"]), dim=0)
        selected = int(torch.argmax(scores))
        selected_indices.append(selected)
        exact += int(selected in row["proven_indices"])
        urgency_hits += int(selected in row["proven_indices"])
        expected = torch.tensor([0.0, 0.0, 1.0])
        value_brier.append(float(torch.mean((value_probabilities - expected) ** 2)))
    return {
        "rows": len(rows),
        "exactActionAccuracy": exact / len(rows),
        "forcedWinAccuracy": exact / len(rows),
        "forcedBlockAccuracy": forced_block_metrics(rows, selected_indices),
        "heldoutMatches": exact / len(rows),
        "wdlAccuracy": 1.0 if all(torch.argmax(model.value_logits(row["features"])) == 2 for row in rows) else 0.0,
        "urgencyDistanceOneRate": urgency_hits / len(rows),
        "valueBrier": sum(value_brier) / len(value_brier),
    }


def control_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact = 0
    selected_indices = []
    for row in rows:
        selected = v4_control_index(row["state"], row["actions"])
        selected_indices.append(selected)
        exact += int(selected in row["proven_indices"])
    return {
        "rows": len(rows),
        "exactActionAccuracy": exact / len(rows) if rows else None,
        "forcedWinAccuracy": exact / len(rows) if rows else None,
        "forcedBlockAccuracy": forced_block_metrics(rows, selected_indices),
        "heldoutMatches": exact / len(rows) if rows else None,
        "urgencyDistanceOneRate": exact / len(rows) if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--gold-weight", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.gold_weight <= 0:
        raise ValueError("--gold-weight must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    heldout = {line.strip() for line in args.heldout.read_text(encoding="ascii").splitlines() if line.strip()}
    rows = read_rows(args.candidates, heldout, args.max_rows)
    prepare(rows)
    train = [row for row in rows if row["partition"] == "train"]
    evaluation = [row for row in rows if row["partition"] == "heldout"]
    if not train or not evaluation:
        raise ValueError("selected rows must contain both train and heldout partitions")
    model = PolicyValueModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    losses = []
    for epoch in range(args.epochs):
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
        for index in order:
            row = train[index]
            scores = model.policy_scores(row["features"])
            positive = torch.tensor(row["proven_indices"], dtype=torch.long)
            policy_loss = -torch.logsumexp(scores[positive], dim=0) + torch.logsumexp(scores, dim=0)
            value_loss = F.cross_entropy(
                model.value_logits(row["features"])[None, :],
                torch.tensor([2], dtype=torch.long),
            )
            loss = args.gold_weight * (policy_loss + 0.25 * value_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    control = control_metrics(evaluation)
    candidate_train = metrics(model, train)
    candidate_heldout = metrics(model, evaluation)
    candidate_at_least_control = candidate_heldout["exactActionAccuracy"] >= control["exactActionAccuracy"]
    report = {
        "schemaVersion": 1,
        "experiment": "ring-1-golden-policy-value-urgency",
        "teacher": {
            "tableFamily": "fresh-frontier-wdl-v1",
            "outcome": "win",
            "distance": 1,
            "policy": "proven-actions-allowed; complete-optimal-set-unknown",
        },
        "seed": args.seed,
        "epochs": args.epochs,
        "goldWeight": args.gold_weight,
        "rows": {"all": len(rows), "train": len(train), "heldout": len(evaluation)},
        "control": {"id": "pathfinder-action-transition-v4-xent", "metrics": control},
        "candidate": {"id": "ring1-linear-gold-candidate-v0.1", "train": candidate_train, "heldout": candidate_heldout},
        "loss": {"initialToFinal": [losses[0], losses[-1]], "mean": sum(losses) / len(losses)},
        "gates": {
            "heldoutPresent": bool(evaluation),
            "exactActionReported": True,
            "wdlCalibrationReported": True,
            "urgencyReported": True,
            "calibrationReported": True,
            "forcedWinGateReported": True,
            "forcedBlockGate": {"status": "not-applicable", "reason": "Ring-1 teacher contains no forced-loss roots"},
            "heldoutMatchGateReported": True,
            "earlierRingRegression": {"status": "not-applicable", "reason": "Ring-1 is the first promoted ring"},
            "candidateAtLeastControlOnHeldout": candidate_at_least_control,
            "promotionDecision": "research-only-until-candidate-beats-control-on-a-nontrivial-ring",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "report": report}, args.output_dir / "candidate.pt")
    print(json.dumps({"train": len(train), "heldout": len(evaluation), "candidate": report["candidate"]["heldout"], "control": report["control"]["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
