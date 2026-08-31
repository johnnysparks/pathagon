#!/usr/bin/env python3
"""Compare independent Python and Rust oracle labels on 3x3 and 4x4 roots."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LAB_ROOT = REPO_ROOT / "research/20260824-gnn-cnn-lab"
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from python.game import BoardConfig, GameState, Player  # noqa: E402


ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"


def mask(squares: tuple[int, ...]) -> int:
    return sum(1 << square for square in squares)


def action_token(action) -> str:
    cells = 49  # The shared corpus token namespace is fixed-width.
    code = action.to if action.kind == 0 else cells + action.from_square * cells + action.to
    return ALPHABET[code >> 6] + ALPHABET[code & 63]


def record(identifier: str, state: GameState) -> dict:
    return {
        "id": identifier,
        "boardSize": state.config.size,
        "reservePerPlayer": state.config.reserve_per_player,
        "maxPlies": state.config.max_plies,
        "light": state.light,
        "dark": state.dark,
        "reserve": list(state.reserves),
        "turn": "light" if state.turn is Player.LIGHT else "dark",
        "forbidden": state.forbidden,
        "lastRelocatedTo": list(state.last_relocated_to),
        "ply": state.ply,
    }


def repetition_key(state: GameState) -> tuple:
    return (
        state.light,
        state.dark,
        state.reserves,
        state.turn,
        state.forbidden,
        state.last_relocated_to,
    )


def strict_value(state: GameState, counts: dict[tuple, int], depth: int):
    """Independent strict minimax: cutoff is Unknown, never an accidental draw."""

    if state.winner is not None:
        return -1
    position = repetition_key(state)
    if counts.get(position, 0) >= 3 or state.ply >= state.config.max_plies:
        return 0
    if depth == 0:
        return None
    actions = state.legal_actions()
    if not actions:
        return 0
    labels = []
    for action in actions:
        child = state.apply_legal(action)
        child_counts = dict(counts)
        child_position = repetition_key(child)
        child_counts[child_position] = child_counts.get(child_position, 0) + 1
        child_value = strict_value(child, child_counts, depth - 1)
        labels.append(None if child_value is None else -child_value)
    if 1 in labels:
        return 1
    if any(label is None for label in labels):
        return None
    if all(label == -1 for label in labels):
        return -1
    return 0


def strict_analysis(state: GameState, depth: int) -> tuple[str, dict[str, str]]:
    counts = {repetition_key(state): 1}
    labels = {}
    for action in state.legal_actions():
        child = state.apply_legal(action)
        child_counts = dict(counts)
        child_position = repetition_key(child)
        child_counts[child_position] = child_counts.get(child_position, 0) + 1
        child_value = strict_value(child, child_counts, depth - 1)
        labels[action_token(action)] = "unknown" if child_value is None else {1: "win", 0: "draw", -1: "loss"}[-child_value]
    root = "win" if "win" in labels.values() else "unknown" if "unknown" in labels.values() else "loss" if labels and all(value == "loss" for value in labels.values()) else "draw"
    return root, labels


def fixtures() -> list[tuple[dict, GameState]]:
    small = BoardConfig(3, 3, 12)
    four = BoardConfig(4, 5, 64)
    states = [
        (
            "3x3-relocation-root",
            GameState(small, 1 << 6, 1 << 2, (2, 2), Player.LIGHT, ply=4),
        ),
        (
            "4x4-immediate",
            GameState(
                four,
                mask((4, 8, 12, 2, 10)),
                mask((1, 3, 6, 9, 14)),
                (0, 0),
                Player.LIGHT,
                ply=20,
            ),
        ),
        (
            "4x4-block",
            GameState(
                four,
                mask((5, 7, 9, 11, 15)),
                mask((1, 2, 3, 6, 10)),
                (0, 0),
                Player.LIGHT,
                ply=20,
            ),
        ),
        (
            "4x4-fork",
            GameState(
                four,
                mask((4, 5, 8, 10, 15)),
                mask((2, 3, 6, 9, 14)),
                (0, 0),
                Player.LIGHT,
                ply=20,
            ),
        ),
    ]
    return [(record(identifier, state), state) for identifier, state in states]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rust-binary",
        type=Path,
        default=REPO_ROOT / "pathagon/engine-rs/target/debug/pathagon-endgame-oracle",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=REPO_ROOT / "research/20260830-endgame-retrograde-frontier/workspace",
    )
    args = parser.parse_args()
    if not args.rust_binary.exists():
        raise SystemExit(f"Rust oracle binary does not exist: {args.rust_binary}")
    args.workspace.mkdir(parents=True, exist_ok=True)
    fixture_rows = fixtures()
    input_path = args.workspace / "small-agreement.jsonl"
    input_path.write_text("".join(json.dumps(raw) + "\n" for raw, _ in fixture_rows), encoding="utf-8")
    result = subprocess.run(
        [str(args.rust_binary), "--input", str(input_path), "--horizon", "3"],
        check=True,
        capture_output=True,
        text=True,
    )
    rust_rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    if len(rust_rows) != len(fixture_rows):
        raise AssertionError("Rust oracle returned the wrong number of fixture rows")

    comparisons = []
    for (raw, state), rust in zip(fixture_rows, rust_rows):
        python_root, python_actions = strict_analysis(state, 3)
        rust_actions = {item["token"]: item["outcome"] for item in rust["actions"]}
        if python_root != rust["outcome"]:
            raise AssertionError(f"{raw['id']}: root outcome differs")
        if python_actions != rust_actions:
            raise AssertionError(f"{raw['id']}: action outcomes differ")
        comparisons.append(
            {
                "id": raw["id"],
                "boardSize": raw["boardSize"],
                "actions": len(python_actions),
                "outcome": rust["outcome"],
                "status": "pass",
            }
        )
    report = {
        "schemaVersion": 1,
        "experiment": "small-board-rust-python-ground-truth-agreement",
        "horizon": 3,
        "fixtures": comparisons,
        "status": "pass",
    }
    report_path = args.workspace / "small-agreement-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
