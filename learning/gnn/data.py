"""Replay import and supervised warm-start examples."""

from __future__ import annotations

import json
import math
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
    policy: Optional[tuple[float, ...]] = None
    policy_actions: Optional[tuple[Action, ...]] = None


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
        record_config = record.get("config") if isinstance(record.get("config"), dict) else {}
        board = config or BoardConfig(
            size=int(record_config.get("boardSize", record.get("boardSize", 7))),
            reserve_per_player=int(record_config.get("reservePerPlayer", record.get("reservePerPlayer", 14))),
            ply_limit=int(record_config.get("maxPlies", record.get("maxPlies", 0))),
        )
        state = GameState.initial(board)
        seed = int(record["seed"])
        for move in record["moves"]:
            action = action_from_record(move["action"])
            legal = state.legal_actions()
            if action not in legal:
                raise ValueError(f"seed {seed}: illegal action {action.short()} at ply {state.ply}")
            policy = parse_policy(move.get("policy"), legal, seed, state.ply)
            examples.append(
                ReplayExample(
                    state,
                    action,
                    winner_value_for_record(record, state),
                    seed,
                    policy,
                    tuple(legal) if policy is not None else None,
                )
            )
            state = state.apply_legal(action)
        expected = record.get("winner")
        actual = None if state.winner is None else ("light" if state.winner is Player.LIGHT else "dark")
        if actual != expected:
            raise ValueError(f"seed {seed}: replay winner mismatch ({actual!r} != {expected!r})")
        records += 1
        if progress is not None:
            progress(records, len(examples))
    return examples


def parse_policy(raw: Any, legal: tuple[Action, ...], seed: int, ply: int) -> Optional[tuple[float, ...]]:
    """Validate an optional soft target aligned to the state's legal actions."""

    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != len(legal):
        raise ValueError(f"seed {seed}: policy length mismatch at ply {ply}")
    policy = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in policy):
        raise ValueError(f"seed {seed}: policy contains an invalid probability at ply {ply}")
    if sum(policy) <= 0.0:
        raise ValueError(f"seed {seed}: policy has no positive probability at ply {ply}")
    return policy


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
