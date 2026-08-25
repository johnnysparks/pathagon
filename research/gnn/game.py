"""Small, variable-size Pathagon rules adapter used by the GNN lab.

The production Rust and TypeScript engines remain the rules authorities. This
module deliberately mirrors their current 7x7 rules while making board size
and placement reserves explicit, so 5x5 curriculum experiments cannot hide a
fixed-size assumption inside the learner.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Iterable, List, Optional, Tuple


class Player(IntEnum):
    LIGHT = 0
    DARK = 1

    def other(self) -> "Player":
        return Player.DARK if self is Player.LIGHT else Player.LIGHT


@dataclass(frozen=True)
class BoardConfig:
    size: int = 7
    reserve_per_player: int = 0
    ply_limit: int = 0

    def __post_init__(self) -> None:
        if self.size < 3:
            raise ValueError("Pathagon boards need at least 3 rows")
        reserve = self.reserve_per_player or 2 * self.size
        if reserve < 1:
            raise ValueError("reserve_per_player must be positive")
        if self.ply_limit < 0:
            raise ValueError("ply_limit cannot be negative")
        object.__setattr__(self, "reserve_per_player", reserve)

    @property
    def cell_count(self) -> int:
        return self.size * self.size

    @property
    def max_plies(self) -> int:
        return self.ply_limit or self.cell_count * 4


@dataclass(frozen=True, order=True)
class Action:
    """A placement has ``kind=0`` and ``from_square=-1``."""

    kind: int
    from_square: int
    to: int

    @classmethod
    def place(cls, to: int) -> "Action":
        return cls(0, -1, to)

    @classmethod
    def relocate(cls, from_square: int, to: int) -> "Action":
        return cls(1, from_square, to)

    def short(self) -> str:
        return f"P{self.to}" if self.kind == 0 else f"R{self.from_square}>{self.to}"


@dataclass(frozen=True)
class GameState:
    config: BoardConfig
    light: int
    dark: int
    reserves: Tuple[int, int]
    turn: Player
    forbidden: int = 0
    last_relocated_to: Tuple[Optional[int], Optional[int]] = (None, None)
    last_capture: int = 0
    last_player: Optional[Player] = None
    winner: Optional[Player] = None
    ply: int = 0

    @classmethod
    def initial(cls, config: BoardConfig = BoardConfig()) -> "GameState":
        reserve = config.reserve_per_player
        return cls(config, 0, 0, (reserve, reserve), Player.LIGHT)

    def pieces(self, player: Player) -> int:
        return self.light if player is Player.LIGHT else self.dark

    def board_at(self, square: int) -> Optional[Player]:
        mask = 1 << square
        if self.light & mask:
            return Player.LIGHT
        if self.dark & mask:
            return Player.DARK
        return None

    def legal_actions(self) -> Tuple[Action, ...]:
        if self.winner is not None:
            return ()
        full = (1 << self.config.cell_count) - 1
        occupied = self.light | self.dark
        destinations = full & ~(occupied | self.forbidden)
        if self.reserves[self.turn] > 0:
            return tuple(Action.place(square) for square in bits(destinations))

        sources = self.pieces(self.turn)
        previous = self.last_relocated_to[self.turn]
        if previous is not None:
            sources &= ~(1 << previous)
        destination_squares = tuple(bits(destinations))
        return tuple(
            Action.relocate(source, destination)
            for source in bits(sources)
            for destination in destination_squares
        )

    def apply(self, action: Action) -> "GameState":
        if action not in self.legal_actions():
            raise ValueError(f"illegal Pathagon action: {action.short()}")
        return self.apply_legal(action)

    def apply_legal(self, action: Action) -> "GameState":
        player = self.turn
        opponent = player.other()
        light, dark = self.light, self.dark
        reserves = list(self.reserves)
        relocated = list(self.last_relocated_to)
        if action.kind == 0:
            reserves[player] -= 1
            relocated[player] = None
        else:
            source_mask = 1 << action.from_square
            if player is Player.LIGHT:
                light &= ~source_mask
            else:
                dark &= ~source_mask
            relocated[player] = action.to

        destination_mask = 1 << action.to
        if player is Player.LIGHT:
            light |= destination_mask
        else:
            dark |= destination_mask
        provisional = GameState(
            self.config,
            light,
            dark,
            tuple(reserves),
            player,
            self.forbidden,
            tuple(relocated),
            self.last_capture,
            self.last_player,
            self.winner,
            self.ply,
        )
        captured = captures_from(provisional, action.to, player)
        if opponent is Player.LIGHT:
            light &= ~captured
        else:
            dark &= ~captured
        reserves[opponent] += count_bits(captured)
        next_state = GameState(
            self.config,
            light,
            dark,
            tuple(reserves),
            opponent,
            captured,
            tuple(relocated),
            count_bits(captured),
            player,
            None,
            self.ply + 1,
        )
        if has_winning_path(next_state, player):
            next_state = GameState(
                next_state.config,
                next_state.light,
                next_state.dark,
                next_state.reserves,
                next_state.turn,
                next_state.forbidden,
                next_state.last_relocated_to,
                next_state.last_capture,
                next_state.last_player,
                player,
                next_state.ply,
            )
        return next_state


def bits(mask: int) -> Iterable[int]:
    while mask:
        lowest = mask & -mask
        yield lowest.bit_length() - 1
        mask ^= lowest


def count_bits(mask: int) -> int:
    return bin(mask).count("1")


def neighbors(config: BoardConfig, square: int) -> Iterable[int]:
    row, column = divmod(square, config.size)
    if row:
        yield square - config.size
    if row + 1 < config.size:
        yield square + config.size
    if column:
        yield square - 1
    if column + 1 < config.size:
        yield square + 1


def has_winning_path(state: GameState, player: Player) -> bool:
    size = state.config.size
    pieces = state.pieces(player)
    if player is Player.LIGHT:
        near = sum(1 << square for square in range((size - 1) * size, size * size))
        far = sum(1 << square for square in range(size))
    else:
        near = sum(1 << (row * size) for row in range(size))
        far = sum(1 << (row * size + size - 1) for row in range(size))
    frontier = pieces & near
    visited = frontier
    while frontier:
        if frontier & far:
            return True
        adjacent = 0
        for square in bits(frontier):
            for neighbor in neighbors(state.config, square):
                adjacent |= 1 << neighbor
        frontier = adjacent & pieces & ~visited
        visited |= frontier
    return False


def captures_from(state: GameState, origin: int, player: Player) -> int:
    opponent = player.other()
    row, column = divmod(origin, state.config.size)
    captured = 0
    for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        near_row = row + row_delta
        near_column = column + column_delta
        far_row = row + 2 * row_delta
        far_column = column + 2 * column_delta
        if not (0 <= far_row < state.config.size and 0 <= far_column < state.config.size):
            continue
        near = near_row * state.config.size + near_column
        far = far_row * state.config.size + far_column
        if state.board_at(near) is opponent and state.board_at(far) is player:
            captured |= 1 << near
    return captured


def action_from_record(raw: Any) -> Action:
    if raw["kind"] == "place":
        return Action.place(int(raw["to"]))
    return Action.relocate(int(raw["from"]), int(raw["to"]))


def winner_value(state: GameState, perspective: Player) -> float:
    if state.winner is None:
        return 0.0
    return 1.0 if state.winner is perspective else -1.0


def repetition_key(state: GameState) -> tuple:
    """Return the rule-relevant position identity used for repetition draws.

    This mirrors the shared Rust/TypeScript engines. ``ply`` and capture
    metadata are intentionally omitted: they record how the position arose,
    not the position used by the threefold rule. The forbidden square and
    relocation markers remain because they affect legal moves.
    """

    return (
        state.light,
        state.dark,
        state.reserves,
        state.turn,
        state.forbidden,
        state.last_relocated_to,
    )
