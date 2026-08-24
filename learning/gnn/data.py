"""Replay import and supervised warm-start examples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

from .game import Action, BoardConfig, GameState, Player, action_from_record


@dataclass(frozen=True)
class ReplayExample:
    state: GameState
    action: Action
    value: float
    seed: int


def iter_records(path: Path) -> Iterable[dict]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        record: Any = value.get("record", value) if isinstance(value, dict) else value
        if isinstance(record, str):
            record = json.loads(record)
        if isinstance(record, dict) and isinstance(record.get("moves"), list):
            yield record
        elif isinstance(record, dict) and isinstance(record.get("games"), list):
            for nested in record["games"]:
                if isinstance(nested, dict) and isinstance(nested.get("moves"), list):
                    yield nested


def load_replay_examples(
    path: Path,
    config: Optional[BoardConfig] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> List[ReplayExample]:
    examples: List[ReplayExample] = []
    records = 0
    for record in iter_records(path):
        board = config or BoardConfig(
            size=int(record.get("boardSize", 7)),
            reserve_per_player=int(record.get("reservePerPlayer", 14)),
        )
        state = GameState.initial(board)
        seed = int(record["seed"])
        for move in record["moves"]:
            action = action_from_record(move["action"])
            legal = state.legal_actions()
            if action not in legal:
                raise ValueError(f"seed {seed}: illegal action {action.short()} at ply {state.ply}")
            examples.append(ReplayExample(state, action, winner_value_for_record(record, state), seed))
            state = state.apply_legal(action)
        expected = record.get("winner")
        actual = None if state.winner is None else ("light" if state.winner is Player.LIGHT else "dark")
        if actual != expected:
            raise ValueError(f"seed {seed}: replay winner mismatch ({actual!r} != {expected!r})")
        records += 1
        if progress is not None:
            progress(records, len(examples))
    return examples


def winner_value_for_record(record: dict, state: GameState) -> float:
    winner = record.get("winner")
    if winner is None:
        return 0.0
    winner_player = Player.LIGHT if winner == "light" else Player.DARK
    return 1.0 if winner_player is state.turn else -1.0


def action_index(state: GameState, action: Action) -> int:
    legal = list(state.legal_actions())
    try:
        return legal.index(action)
    except ValueError as error:
        raise ValueError(f"action {action.short()} is not legal in state at ply {state.ply}") from error
