"""Small-board graph and tactical search helpers.

These helpers are deliberately action-aware. ``connection_distance`` is a
useful static estimate, but a move can also block an immediate loss or create
multiple winning replies. The latter require a shallow AND/OR search over the
legal move graph.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Tuple

from .evaluation import connection_distance
from .game import Action, GameState, Player, bits


@lru_cache(maxsize=None)
def goal_path_masks(size: int, player: Player) -> Tuple[int, ...]:
    """Enumerate simple goal-to-goal path masks for a small board.

    Path masks are an optimization for one-away checks, not a replacement for
    graph search. A 4x4 board has a small enough path family to cache. Larger
    boards should use ``connection_distance`` instead of eagerly enumerating
    every simple path.
    """

    if size > 4:
        raise ValueError("goal path masks are limited to boards of size 4 or smaller")
    if player is Player.LIGHT:
        starts = tuple(range((size - 1) * size, size * size))

        def is_goal(square: int) -> bool:
            return square < size
    else:
        starts = tuple(range(0, size * size, size))

        def is_goal(square: int) -> bool:
            return square % size == size - 1

    paths: set[int] = set()

    def visit(square: int, visited: int) -> None:
        if is_goal(square):
            paths.add(visited)
            return
        row, column = divmod(square, size)
        adjacent = []
        if row:
            adjacent.append(square - size)
        if row + 1 < size:
            adjacent.append(square + size)
        if column:
            adjacent.append(square - 1)
        if column + 1 < size:
            adjacent.append(square + 1)
        for neighbor in adjacent:
            bit = 1 << neighbor
            if not visited & bit:
                visit(neighbor, visited | bit)

    for start in starts:
        visit(start, 1 << start)
    return tuple(sorted(paths))


def _view_for_player(state: GameState, player: Player) -> GameState:
    return state if state.turn is player else replace(state, turn=player)


def immediate_winning_actions(state: GameState, player: Player) -> Tuple[Action, ...]:
    """Return legal actions that finish a path for ``player`` immediately."""

    if state.winner is not None:
        return ()
    view = _view_for_player(state, player)
    return tuple(action for action in view.legal_actions() if view.apply_legal(action).winner is player)


def one_away_path_actions(state: GameState, player: Player) -> Tuple[Action, ...]:
    """Fast bit-mask candidates for actions filling a one-missing-cell path.

    The authoritative check remains ``immediate_winning_actions`` because
    captures and relocation constraints are part of the rules. This helper is
    useful for feature extraction and as a cheap prefilter.
    """

    view = _view_for_player(state, player)
    own = view.pieces(player)
    occupied = view.light | view.dark
    full = (1 << view.config.cell_count) - 1
    empty = full & ~(occupied | view.forbidden)
    legal = set(view.legal_actions())
    candidates: set[Action] = set()
    for path in goal_path_masks(view.config.size, player):
        missing = path & ~own
        if missing.bit_count() != 1 or not missing & empty:
            continue
        target = next(bits(missing))
        if view.reserves[player] > 0:
            action = Action.place(target)
            if action in legal:
                candidates.add(action)
            continue
        movable = own
        previous = view.last_relocated_to[player]
        if previous is not None:
            movable &= ~(1 << previous)
        # Moving a piece already on this path would create a second hole.
        for source in bits(movable & ~path):
            action = Action.relocate(source, target)
            if action in legal:
                candidates.add(action)
    return tuple(sorted(candidates))


def connection_distance_delta(state: GameState, action: Action) -> Tuple[int, int, int, int]:
    """Return own/opponent connection distances before and after an action."""

    player = state.turn
    opponent = player.other()
    next_state = state.apply_legal(action)
    return (
        connection_distance(state, player),
        connection_distance(next_state, player),
        connection_distance(state, opponent),
        connection_distance(next_state, opponent),
    )


@dataclass(frozen=True)
class TacticalRoot:
    """Exact shallow tactical classifications for one root position."""

    immediate_wins: Tuple[Action, ...]
    forced_blocks: Tuple[Action, ...]
    forced_forks: Tuple[Action, ...]
    root_action_count: int
    root_reply_edges: int

    @property
    def priority_actions(self) -> Tuple[Action, ...]:
        if self.immediate_wins:
            return self.immediate_wins
        if self.forced_blocks:
            return self.forced_blocks
        return self.forced_forks


def tactical_root(state: GameState) -> TacticalRoot:
    """Exhaust the tactical tree needed for wins, blocks, and forced forks."""

    player = state.turn
    opponent = player.other()
    root_actions = tuple(state.legal_actions())
    reply_edges = sum(len(state.apply_legal(action).legal_actions()) for action in root_actions)
    immediate = tuple(action for action in root_actions if state.apply_legal(action).winner is player)
    if immediate:
        return TacticalRoot(immediate, (), (), len(root_actions), reply_edges)

    opponent_threats = immediate_winning_actions(state, opponent)
    blocks = tuple(
        action
        for action in root_actions
        if opponent_threats and not immediate_winning_actions(state.apply_legal(action), opponent)
    )

    forks = []
    for action in root_actions:
        after = state.apply_legal(action)
        if after.winner is not None or immediate_winning_actions(after, opponent):
            continue
        if len(immediate_winning_actions(after, player)) < 2:
            continue
        replies = after.legal_actions()
        if not replies:
            continue
        # Every opponent reply must leave at least one immediate win for the
        # root player. This makes the fork a forced tactical result rather
        # than merely two threats that can both be neutralized.
        if all(
            reply_state.winner is not opponent and immediate_winning_actions(reply_state, player)
            for reply in replies
            for reply_state in (after.apply_legal(reply),)
        ):
            forks.append(action)
    return TacticalRoot(immediate, blocks, tuple(forks), len(root_actions), reply_edges)


def tactical_priority_actions(state: GameState) -> Tuple[Action, ...]:
    """Return exact tactical actions in priority order for a small board."""

    return tactical_root(state).priority_actions
