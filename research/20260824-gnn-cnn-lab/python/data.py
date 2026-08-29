"""Replay import and supervised warm-start examples."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

from .game import Action, BoardConfig, GameState, Player, action_from_record
from .contract import ROOT_Q_SOURCE


@dataclass(frozen=True)
class ReplayExample:
    state: GameState
    action: Action
    value: float
    seed: int
    policy: Optional[tuple[float, ...]] = None
    policy_actions: Optional[tuple[Action, ...]] = None
    action_values: Optional[tuple[float, ...]] = None
    action_visits: Optional[tuple[int, ...]] = None
    action_value_actions: Optional[tuple[Action, ...]] = None
    rank_scores: Optional[tuple[float, ...]] = None
    rank_actions: Optional[tuple[Action, ...]] = None


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
        state = initial_state_from_record(record, board)
        seed = int(record["seed"])
        for move in record["moves"]:
            action = action_from_record(move["action"])
            legal = state.legal_actions()
            if action not in legal:
                raise ValueError(f"seed {seed}: illegal action {action.short()} at ply {state.ply}")
            policy = parse_policy(move.get("policy"), legal, seed, state.ply)
            action_values, action_visits = parse_action_values(
                move.get("actionValues"), move.get("actionVisits"), legal, seed, state.ply
            )
            rank_actions, rank_scores = parse_rank_targets(
                move.get("rankActions"), move.get("rankScores"), legal, seed, state.ply
            )
            if action_values is not None and move.get("actionValueSource") != ROOT_Q_SOURCE:
                raise ValueError(f"seed {seed}: unsupported action value source at ply {state.ply}")
            if action_values is None and "actionValueSource" in move:
                raise ValueError(f"seed {seed}: action value source has no Q target at ply {state.ply}")
            examples.append(
                ReplayExample(
                    state,
                    action,
                    winner_value_for_record(record, state),
                    seed,
                    policy,
                    tuple(legal) if policy is not None else None,
                    action_values,
                    action_visits,
                    tuple(legal) if action_values is not None else None,
                    rank_scores,
                    rank_actions,
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


def initial_state_from_record(record: dict[str, Any], config: BoardConfig) -> GameState:
    """Return the declared replay root, or the ordinary empty-board root."""

    raw = record.get("initialPosition")
    if raw is None:
        return GameState.initial(config)
    if not isinstance(raw, dict):
        raise ValueError("initialPosition must be an object")
    root_config = raw.get("config") if isinstance(raw.get("config"), dict) else {}
    if int(root_config.get("boardSize", config.size)) != config.size or int(root_config.get("reservePerPlayer", config.reserve_per_player)) != config.reserve_per_player:
        raise ValueError("initialPosition config does not match replay config")
    board = raw.get("board")
    if not isinstance(board, list) or len(board) != config.cell_count:
        raise ValueError("initialPosition board does not match replay config")
    light = 0
    dark = 0
    for square, piece in enumerate(board):
        if piece == "light":
            light |= 1 << square
        elif piece == "dark":
            dark |= 1 << square
        elif piece is not None:
            raise ValueError("initialPosition contains an unknown piece")
    reserve = raw.get("reserve")
    if not isinstance(reserve, dict):
        raise ValueError("initialPosition reserve is missing")
    turn = raw.get("turn")
    if turn not in ("light", "dark"):
        raise ValueError("initialPosition turn is invalid")
    forbidden_values = raw.get("forbidden", [])
    if not isinstance(forbidden_values, list) or any(
        not isinstance(square, int) or isinstance(square, bool) or not 0 <= square < config.cell_count
        for square in forbidden_values
    ):
        raise ValueError("initialPosition forbidden squares are invalid")
    forbidden = sum(1 << square for square in forbidden_values)
    markers = raw.get("lastRelocatedTo", {"light": None, "dark": None})
    if not isinstance(markers, dict):
        raise ValueError("initialPosition relocation markers are invalid")
    marker_values = (markers.get("light"), markers.get("dark"))
    if any(
        marker is not None
        and (not isinstance(marker, int) or isinstance(marker, bool) or not 0 <= marker < config.cell_count)
        for marker in marker_values
    ):
        raise ValueError("initialPosition relocation markers are invalid")
    ply = raw.get("ply", 0)
    if not isinstance(ply, int) or isinstance(ply, bool):
        raise ValueError("initialPosition ply is invalid")
    reserve_values = (reserve.get("light"), reserve.get("dark"))
    if any(not isinstance(value, int) or isinstance(value, bool) for value in reserve_values):
        raise ValueError("initialPosition reserve values are invalid")
    return GameState.seeded(
        config,
        light,
        dark,
        (reserve_values[0], reserve_values[1]),
        Player.LIGHT if turn == "light" else Player.DARK,
        forbidden=forbidden,
        last_relocated_to=marker_values,
        ply=ply,
    )


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


def parse_action_values(
    raw_values: Any,
    raw_visits: Any,
    legal: tuple[Action, ...],
    seed: int,
    ply: int,
) -> tuple[Optional[tuple[float, ...]], Optional[tuple[int, ...]]]:
    """Validate root Q targets and their confidence counts."""

    if raw_values is None and raw_visits is None:
        return None, None
    if not isinstance(raw_values, list) or not isinstance(raw_visits, list):
        raise ValueError(f"seed {seed}: action values and visits must be lists at ply {ply}")
    if len(raw_values) != len(legal) or len(raw_visits) != len(legal):
        raise ValueError(f"seed {seed}: action value length mismatch at ply {ply}")
    values = tuple(float(value) for value in raw_values)
    if any(not math.isfinite(value) or value < -1.0 or value > 1.0 for value in values):
        raise ValueError(f"seed {seed}: action values contain an invalid Q estimate at ply {ply}")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in raw_visits):
        raise ValueError(f"seed {seed}: action visits contain an invalid count at ply {ply}")
    return values, tuple(raw_visits)


def parse_rank_targets(
    raw_actions: Any,
    raw_scores: Any,
    legal: tuple[Action, ...],
    seed: int,
    ply: int,
) -> tuple[Optional[tuple[Action, ...]], Optional[tuple[float, ...]]]:
    """Validate a partial Pathfinder score vector used for sorter ranking."""

    if raw_actions is None and raw_scores is None:
        return None, None
    if not isinstance(raw_actions, list) or not isinstance(raw_scores, list):
        raise ValueError(f"seed {seed}: rank actions and scores must be lists at ply {ply}")
    if len(raw_actions) != len(raw_scores) or not raw_actions:
        raise ValueError(f"seed {seed}: rank target length mismatch at ply {ply}")
    actions = tuple(action_from_record(value) for value in raw_actions)
    if any(action not in legal for action in actions):
        raise ValueError(f"seed {seed}: rank target contains an illegal action at ply {ply}")
    if len(set(actions)) != len(actions):
        raise ValueError(f"seed {seed}: rank target contains duplicate actions at ply {ply}")
    scores = tuple(float(value) for value in raw_scores)
    if any(not math.isfinite(value) for value in scores):
        raise ValueError(f"seed {seed}: rank target contains a non-finite score at ply {ply}")
    return actions, scores


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
