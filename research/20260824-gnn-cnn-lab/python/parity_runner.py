"""Emit normalized rules results for the generated cross-runtime fixture."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .game import Action, BoardConfig, GameState, Player


def player(value: str) -> Player:
    return Player.LIGHT if value == "light" else Player.DARK


def action_value(action: Action) -> dict:
    if action.kind == 0:
        return {"kind": "place", "to": action.to}
    return {"kind": "relocate", "from": action.from_square, "to": action.to}


def state_value(state: GameState) -> dict:
    return {
        "board": [None if state.board_at(square) is None else ("light" if state.board_at(square) is Player.LIGHT else "dark") for square in range(state.config.cell_count)],
        "reserve": {"light": state.reserves[Player.LIGHT], "dark": state.reserves[Player.DARK]},
        "turn": "light" if state.turn is Player.LIGHT else "dark",
        "forbidden": list(bits(state.forbidden)),
        "lastRelocatedTo": {
            "light": state.last_relocated_to[Player.LIGHT],
            "dark": state.last_relocated_to[Player.DARK],
        },
        "winner": None if state.winner is None else ("light" if state.winner is Player.LIGHT else "dark"),
        "ply": state.ply,
    }


def bits(mask: int):
    while mask:
        lowest = mask & -mask
        yield lowest.bit_length() - 1
        mask ^= lowest


def make_state(raw: dict) -> GameState:
    config_value = raw["config"]
    config = BoardConfig(config_value["boardSize"], config_value["reservePerPlayer"], config_value["maxPlies"])
    board = raw["board"]
    light = sum(1 << square for square, piece in enumerate(board) if piece == "light")
    dark = sum(1 << square for square, piece in enumerate(board) if piece == "dark")
    reserve = (raw["reserve"]["light"], raw["reserve"]["dark"])
    last = raw["lastRelocatedTo"]
    return GameState(
        config,
        light,
        dark,
        reserve,
        player(raw["turn"]),
        sum(1 << square for square in raw["forbidden"]),
        (last["light"], last["dark"]),
        winner=None if raw["winner"] is None else player(raw["winner"]),
        ply=raw["ply"],
    )


def run_case(case: dict) -> dict:
    state = make_state(case["position"])
    actions = state.legal_actions()
    return {
        "name": case["name"],
        "config": {
            "rulesVersion": "pathagon-rules-v1",
            "boardSize": state.config.size,
            "reservePerPlayer": state.config.reserve_per_player,
            "maxPlies": state.config.max_plies,
            "repetitionLimit": 3,
        },
        "state": state_value(state),
        "legalActions": [action_value(action) for action in actions],
        "transitions": [
            {"action": action_value(action), "state": state_value(state.apply_legal(action))}
            for action in actions
        ],
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m research.gnn.parity_runner <fixture.json>")
    fixture = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if fixture.get("fixtureVersion") != 1:
        raise ValueError("unsupported parity fixture version")
    json.dump([run_case(case) for case in fixture["cases"]], sys.stdout, separators=(",", ":"), sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
