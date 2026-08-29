"""Dynamic graph construction for 5x5, 7x7, and future board sizes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .game import BoardConfig, GameState, Player, bits, neighbors


NODE_FEATURES = 21
GLOBAL_FEATURES = 8
BOUNDARY_NODES = 4


@dataclass(frozen=True)
class GraphTensors:
    node_features: torch.Tensor
    edge_index: torch.Tensor
    global_features: torch.Tensor
    board_nodes: int


def build_graph(state: GameState, device: Optional[torch.device] = None) -> GraphTensors:
    """Return a board graph plus four typed virtual goal nodes.

    Board nodes use only local state and normalized coordinates. The virtual
    nodes make the four target edges explicit without encoding a fixed 7x7
    output shape. All edges are undirected, including board-to-goal edges.
    """

    config = state.config
    cell_count = config.cell_count
    total_nodes = cell_count + BOUNDARY_NODES
    features = torch.zeros((total_nodes, NODE_FEATURES), dtype=torch.float32, device=device)
    denominator = float(max(1, config.size - 1))
    size_feature = float(config.size) / 7.0
    for square in range(cell_count):
        row, column = divmod(square, config.size)
        piece = state.board_at(square)
        features[square, 0 if piece is None else int(piece) + 1] = 1.0
        features[square, 3] = float(bool(state.forbidden & (1 << square)))
        features[square, 4] = float(state.last_relocated_to[Player.LIGHT] == square)
        features[square, 5] = float(state.last_relocated_to[Player.DARK] == square)
        features[square, 6] = row / denominator
        features[square, 7] = column / denominator
        features[square, 8] = float(row == 0)
        features[square, 9] = float(row == config.size - 1)
        features[square, 10] = float(column == 0)
        features[square, 11] = float(column == config.size - 1)
        features[square, 12] = 1.0
        features[square, 14] = size_feature
        features[square, 15] = float(state.turn is Player.LIGHT)
        features[square, 16] = float(state.turn is Player.DARK)

    for boundary in range(BOUNDARY_NODES):
        features[cell_count + boundary, 13] = 1.0
        features[cell_count + boundary, 14] = size_feature
        features[cell_count + boundary, 17 + boundary] = 1.0

    edges = []

    def add_edge(left: int, right: int) -> None:
        edges.append((left, right))
        edges.append((right, left))

    for square in range(cell_count):
        add_edge(square, square)
        for neighbor in neighbors(config, square):
            if square < neighbor:
                add_edge(square, neighbor)
    light_top, light_bottom, dark_left, dark_right = range(cell_count, cell_count + 4)
    for square in range(config.size):
        add_edge(square, light_top)
        add_edge((config.size - 1) * config.size + square, light_bottom)
        add_edge(square * config.size, dark_left)
        add_edge(square * config.size + config.size - 1, dark_right)
    for boundary in range(BOUNDARY_NODES):
        add_edge(cell_count + boundary, cell_count + boundary)

    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    global_features = torch.tensor(
        [
            state.reserves[Player.LIGHT] / float(config.reserve_per_player),
            state.reserves[Player.DARK] / float(config.reserve_per_player),
            float(state.turn is Player.LIGHT),
            float(state.turn is Player.DARK),
            state.last_capture / 4.0,
            float(state.last_player is Player.LIGHT),
            float(state.last_player is Player.DARK),
            state.ply / float(max(1, config.max_plies)),
        ],
        dtype=torch.float32,
        device=device,
    )
    return GraphTensors(features, edge_index, global_features, cell_count)

