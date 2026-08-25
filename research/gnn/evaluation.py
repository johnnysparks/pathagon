"""Cheap successor-state evaluation shared by search and league agents."""

from __future__ import annotations

import heapq
import math
from typing import Iterable, List, Set

from .game import BoardConfig, GameState, Player


def normalize_heuristic(score: float) -> float:
    return math.tanh(score / 3_500.0)


def evaluate_position(state: GameState, player: Player) -> float:
    if state.winner is player:
        return 1_000_000_000 - state.ply
    if state.winner is player.other():
        return -1_000_000_000 + state.ply
    own_distance = connection_distance(state, player)
    opponent_distance = connection_distance(state, player.other())
    own_pieces = count_bits(state.pieces(player))
    opponent_pieces = count_bits(state.pieces(player.other()))
    capture_direction = 1 if state.last_player is player else -1
    structure = largest_component(state, player) - largest_component(state, player.other())
    threats = capture_opportunities(state, player) - capture_opportunities(state, player.other())
    edges = edge_presence(state, player) - edge_presence(state, player.other())
    return (
        (opponent_distance - own_distance) * 240
        + (own_pieces - opponent_pieces) * 110
        + capture_direction * state.last_capture * 700
        + structure * 55
        + threats * 130
        + edges * 80
    )


def connection_distance(state: GameState, player: Player) -> int:
    size = state.config.size
    opponent = player.other()
    distances = [float("inf")] * state.config.cell_count
    frontier: List[tuple[float, int]] = []
    starts = range((size - 1) * size, size * size) if player is Player.LIGHT else range(0, size * size, size)
    for square in starts:
        if state.board_at(square) is opponent:
            continue
        distance = 0 if state.board_at(square) is player else 1
        if distance < distances[square]:
            distances[square] = distance
            heapq.heappush(frontier, (distance, square))
    while frontier:
        distance, square = heapq.heappop(frontier)
        if distance != distances[square]:
            continue
        row, column = divmod(square, size)
        if (player is Player.LIGHT and row == 0) or (player is Player.DARK and column == size - 1):
            return int(distance)
        for neighbor in neighbors(state.config, square):
            if state.board_at(neighbor) is opponent:
                continue
            next_distance = distance + (0 if state.board_at(neighbor) is player else 1)
            if next_distance < distances[neighbor]:
                distances[neighbor] = next_distance
                heapq.heappush(frontier, (next_distance, neighbor))
    return state.config.cell_count


def largest_component(state: GameState, player: Player) -> int:
    remaining = set(squares_from_mask(state.pieces(player)))
    largest = 0
    while remaining:
        stack = [remaining.pop()]
        component = 0
        while stack:
            square = stack.pop()
            component += 1
            for neighbor in neighbors(state.config, square):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        largest = max(largest, component)
    return largest


def capture_opportunities(state: GameState, player: Player) -> int:
    forbidden = state.forbidden
    opponent = player.other()
    victims: Set[int] = set()
    for origin in range(state.config.cell_count):
        if state.board_at(origin) is not None or forbidden & (1 << origin):
            continue
        row, column = divmod(origin, state.config.size)
        for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            far_row = row + row_delta * 2
            far_column = column + column_delta * 2
            if not (0 <= far_row < state.config.size and 0 <= far_column < state.config.size):
                continue
            near = (row + row_delta) * state.config.size + column + column_delta
            far = far_row * state.config.size + far_column
            if state.board_at(near) is opponent and state.board_at(far) is player:
                victims.add(near)
    return len(victims)


def edge_presence(state: GameState, player: Player) -> int:
    size = state.config.size
    if player is Player.LIGHT:
        near = range((size - 1) * size, size * size)
        far = range(size)
    else:
        near = range(0, size * size, size)
        far = range(size - 1, size * size, size)
    return int(any(state.board_at(square) is player for square in near)) + int(any(state.board_at(square) is player for square in far))


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


def squares_from_mask(mask: int) -> Iterable[int]:
    while mask:
        lowest = mask & -mask
        yield lowest.bit_length() - 1
        mask ^= lowest


def count_bits(mask: int) -> int:
    return mask.bit_count()
