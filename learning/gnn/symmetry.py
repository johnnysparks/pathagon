"""Exact square-board symmetries for rules-preserving augmentation.

Pathagon has two different connection axes: Light connects top-to-bottom and
Dark connects left-to-right.  The four axis-preserving D4 transforms can keep
player identities unchanged.  The other four transforms exchange the axes,
so they also exchange Light and Dark.  That gives eight legal symmetries of
the complete game, not merely eight geometric transforms of the bitmap.
"""

from __future__ import annotations

import random
from enum import Enum
from typing import Iterable, Tuple

from .game import Action, BoardConfig, GameState, Player, bits


class Symmetry(str, Enum):
    """The eight square-board transformations in D4 order."""

    IDENTITY = "identity"
    ROTATE_90 = "rotate-90"
    ROTATE_180 = "rotate-180"
    ROTATE_270 = "rotate-270"
    FLIP_ROWS = "flip-rows"
    FLIP_COLUMNS = "flip-columns"
    TRANSPOSE = "transpose"
    ANTI_TRANSPOSE = "anti-transpose"


ALL_SYMMETRIES: Tuple[Symmetry, ...] = tuple(Symmetry)

# Rotations by 90 degrees and diagonal reflections exchange the vertical and
# horizontal connection axes.  Swapping players at the same time restores the
# rules: Light's vertical goal becomes Dark's horizontal goal, and vice versa.
PLAYER_SWAPPING_SYMMETRIES = frozenset(
    {
        Symmetry.ROTATE_90,
        Symmetry.ROTATE_270,
        Symmetry.TRANSPOSE,
        Symmetry.ANTI_TRANSPOSE,
    }
)


def symmetry_swaps_players(symmetry: Symmetry) -> bool:
    return symmetry in PLAYER_SWAPPING_SYMMETRIES


def _coordinates(size: int, row: int, column: int, symmetry: Symmetry) -> tuple[int, int]:
    last = size - 1
    if symmetry is Symmetry.IDENTITY:
        return row, column
    if symmetry is Symmetry.ROTATE_90:
        return column, last - row
    if symmetry is Symmetry.ROTATE_180:
        return last - row, last - column
    if symmetry is Symmetry.ROTATE_270:
        return last - column, row
    if symmetry is Symmetry.FLIP_ROWS:
        return last - row, column
    if symmetry is Symmetry.FLIP_COLUMNS:
        return row, last - column
    if symmetry is Symmetry.TRANSPOSE:
        return column, row
    if symmetry is Symmetry.ANTI_TRANSPOSE:
        return last - column, last - row
    raise ValueError(f"unsupported symmetry: {symmetry}")


def transform_square(size: int, square: int, symmetry: Symmetry) -> int:
    """Map a row-major square index through ``symmetry``."""

    if not 0 <= square < size * size:
        raise ValueError(f"square {square} is outside a {size}x{size} board")
    row, column = divmod(square, size)
    new_row, new_column = _coordinates(size, row, column, symmetry)
    return new_row * size + new_column


def transform_mask(mask: int, size: int, symmetry: Symmetry) -> int:
    """Transform a bitboard mask without assuming a fixed board size."""

    transformed = 0
    for square in bits(mask):
        transformed |= 1 << transform_square(size, square, symmetry)
    return transformed


def transform_action(action: Action, config: BoardConfig, symmetry: Symmetry) -> Action:
    """Transform an action while retaining its placement/relocation kind."""

    destination = transform_square(config.size, action.to, symmetry)
    if action.kind == 0:
        return Action.place(destination)
    return Action.relocate(transform_square(config.size, action.from_square, symmetry), destination)


def transform_actions(actions: Iterable[Action], config: BoardConfig, symmetry: Symmetry) -> tuple[Action, ...]:
    return tuple(transform_action(action, config, symmetry) for action in actions)


def transform_state(state: GameState, symmetry: Symmetry) -> GameState:
    """Return a rules-equivalent state in the transformed coordinate system."""

    config = state.config
    light = transform_mask(state.light, config.size, symmetry)
    dark = transform_mask(state.dark, config.size, symmetry)
    forbidden = transform_mask(state.forbidden, config.size, symmetry)
    last_relocated_to = tuple(
        None
        if square is None
        else transform_square(config.size, square, symmetry)
        for square in state.last_relocated_to
    )
    reserves = state.reserves
    turn = state.turn
    last_player = state.last_player
    winner = state.winner

    if symmetry_swaps_players(symmetry):
        light, dark = dark, light
        reserves = (reserves[Player.DARK], reserves[Player.LIGHT])
        last_relocated_to = (last_relocated_to[Player.DARK], last_relocated_to[Player.LIGHT])
        turn = turn.other()
        last_player = None if last_player is None else last_player.other()
        winner = None if winner is None else winner.other()

    return GameState(
        config=config,
        light=light,
        dark=dark,
        reserves=reserves,
        turn=turn,
        forbidden=forbidden,
        last_relocated_to=last_relocated_to,
        last_capture=state.last_capture,
        last_player=last_player,
        winner=winner,
        ply=state.ply,
    )


def sample_symmetry(rng: random.Random, include_identity: bool = True) -> Symmetry:
    """Sample one D4 transform for deterministic, seedable augmentation."""

    choices = ALL_SYMMETRIES if include_identity else ALL_SYMMETRIES[1:]
    return choices[rng.randrange(len(choices))]

