"""Deterministic transition features for the Q/advantage action head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from .evaluation import (
    capture_opportunities,
    connection_distance,
    edge_presence,
    largest_component,
)
from .game import Action, GameState, Player, count_bits, neighbors


TRANSITION_FEATURE_NAMES = (
    "place",
    "relocate",
    "from_row",
    "from_column",
    "to_row",
    "to_column",
    "captured",
    "own_path_gain",
    "opponent_path_loss",
    "own_component_gain",
    "opponent_component_loss",
    "own_threat_gain",
    "opponent_threat_loss",
    "own_edge_gain",
    "opponent_edge_loss",
    "own_piece_delta",
    "opponent_piece_loss",
    "own_reserve_delta",
    "opponent_reserve_gain",
    "mobility_delta",
    "immediate_win",
    "destination_own_neighbors",
    "destination_opponent_neighbors",
    "destination_empty_neighbors",
)
TRANSITION_FEATURES = len(TRANSITION_FEATURE_NAMES)


@dataclass(frozen=True)
class _PositionSignals:
    own_distance: int
    opponent_distance: int
    own_component: int
    opponent_component: int
    own_threats: int
    opponent_threats: int
    own_edges: int
    opponent_edges: int
    own_pieces: int
    opponent_pieces: int
    own_reserve: int
    opponent_reserve: int
    mobility: int


def _signals(state: GameState, player: Player) -> _PositionSignals:
    opponent = player.other()
    return _PositionSignals(
        own_distance=connection_distance(state, player),
        opponent_distance=connection_distance(state, opponent),
        own_component=largest_component(state, player),
        opponent_component=largest_component(state, opponent),
        own_threats=capture_opportunities(state, player),
        opponent_threats=capture_opportunities(state, opponent),
        own_edges=edge_presence(state, player),
        opponent_edges=edge_presence(state, opponent),
        own_pieces=count_bits(state.pieces(player)),
        opponent_pieces=count_bits(state.pieces(opponent)),
        own_reserve=state.reserves[player],
        opponent_reserve=state.reserves[opponent],
        mobility=len(state.legal_actions()),
    )


def _normalized_delta(value: int, scale: int) -> float:
    return float(value) / float(max(1, scale))


def _neighbor_counts(state: GameState, square: int, player: Player) -> tuple[int, int, int]:
    own = opponent = empty = 0
    for neighbor in neighbors(state.config, square):
        piece = state.board_at(neighbor)
        if piece is player:
            own += 1
        elif piece is player.other():
            opponent += 1
        else:
            empty += 1
    return own, opponent, empty


def transition_features(
    state: GameState,
    actions: Sequence[Action] | Iterable[Action] | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Encode legal actions by their deterministic state transition effects.

    Every row is from the side-to-move perspective and follows the action
    sequence supplied by the caller. The feature set intentionally contains
    inspectable afterstate deltas rather than learned board-only summaries.
    """

    legal = tuple(actions) if actions is not None else state.legal_actions()
    if not legal:
        return torch.empty((0, TRANSITION_FEATURES), dtype=torch.float32, device=device)

    player = state.turn
    before = _signals(state, player)
    size_scale = state.config.cell_count
    coordinate_scale = float(max(1, state.config.size - 1))
    rows: list[list[float]] = []
    for action in legal:
        next_state = state.apply_legal(action)
        after = _signals(next_state, player)
        row, column = divmod(action.to, state.config.size)
        if action.kind == 0:
            from_row = from_column = 0.0
        else:
            from_row, from_column = divmod(action.from_square, state.config.size)
            from_row /= coordinate_scale
            from_column /= coordinate_scale
        own_neighbors, opponent_neighbors, empty_neighbors = _neighbor_counts(next_state, action.to, player)
        rows.append([
            float(action.kind == 0),
            float(action.kind == 1),
            float(from_row),
            float(from_column),
            row / coordinate_scale,
            column / coordinate_scale,
            _normalized_delta(next_state.last_capture, 4),
            _normalized_delta(before.own_distance - after.own_distance, size_scale),
            _normalized_delta(after.opponent_distance - before.opponent_distance, size_scale),
            _normalized_delta(after.own_component - before.own_component, size_scale),
            _normalized_delta(before.opponent_component - after.opponent_component, size_scale),
            _normalized_delta(after.own_threats - before.own_threats, 4),
            _normalized_delta(before.opponent_threats - after.opponent_threats, 4),
            _normalized_delta(after.own_edges - before.own_edges, 2),
            _normalized_delta(before.opponent_edges - after.opponent_edges, 2),
            _normalized_delta(after.own_pieces - before.own_pieces, size_scale),
            _normalized_delta(before.opponent_pieces - after.opponent_pieces, size_scale),
            _normalized_delta(after.own_reserve - before.own_reserve, state.config.reserve_per_player),
            _normalized_delta(after.opponent_reserve - before.opponent_reserve, state.config.reserve_per_player),
            _normalized_delta(after.mobility - before.mobility, size_scale * size_scale),
            float(next_state.winner is player),
            _normalized_delta(own_neighbors, 4),
            _normalized_delta(opponent_neighbors, 4),
            _normalized_delta(empty_neighbors, 4),
        ])
    return torch.tensor(rows, dtype=torch.float32, device=device)
