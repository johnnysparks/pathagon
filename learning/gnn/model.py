"""A dependency-light residual message-passing policy/value network."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .game import Action, GameState
from .graph import GLOBAL_FEATURES, NODE_FEATURES, build_graph


class MessageLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.update = nn.Linear(hidden_size * 2, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, nodes: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source, destination = edge_index
        aggregate = torch.zeros_like(nodes)
        aggregate.index_add_(0, destination, nodes[source])
        degree = torch.zeros((nodes.shape[0], 1), dtype=nodes.dtype, device=nodes.device)
        degree.index_add_(0, destination, torch.ones((destination.shape[0], 1), dtype=nodes.dtype, device=nodes.device))
        aggregate = aggregate / degree.clamp_min(1.0)
        update = F.gelu(self.update(torch.cat((nodes, aggregate), dim=-1)))
        return self.norm(nodes + update)


class PathagonGNN(nn.Module):
    """Graph encoder with node and pairwise dynamic action heads.

    The model never allocates a fixed 25- or 49-way policy head. Placement
    actions read one node embedding; relocation actions read a source and a
    destination embedding. The caller supplies the legal action list, so the
    same network handles both board sizes and both movement phases.
    """

    def __init__(self, hidden_size: int = 64, message_layers: int = 8) -> None:
        super().__init__()
        if message_layers < 1:
            raise ValueError("message_layers must be positive")
        self.hidden_size = hidden_size
        self.message_layer_count = message_layers
        self.input = nn.Linear(NODE_FEATURES, hidden_size)
        self.layers = nn.ModuleList(MessageLayer(hidden_size) for _ in range(message_layers))
        self.context = nn.Sequential(
            nn.Linear(hidden_size * 2 + GLOBAL_FEATURES, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.place_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.relocate_head = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def encode(self, state: GameState) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        graph = build_graph(state, device=device)
        nodes = self.input(graph.node_features)
        for layer in self.layers:
            nodes = layer(nodes, graph.edge_index)
        pooled = torch.cat((nodes.mean(dim=0), nodes.amax(dim=0), graph.global_features), dim=0)
        context = self.context(pooled)
        return nodes[: graph.board_nodes], context

    def policy_value(
        self,
        state: GameState,
        actions: Optional[List[Action]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        legal = actions if actions is not None else list(state.legal_actions())
        board_nodes, context = self.encode(state)
        logits: List[torch.Tensor] = []
        for action in legal:
            if action.kind == 0:
                features = torch.cat((board_nodes[action.to], context), dim=0)
                logits.append(self.place_head(features).squeeze(-1))
            else:
                features = torch.cat((board_nodes[action.from_square], board_nodes[action.to], context), dim=0)
                logits.append(self.relocate_head(features).squeeze(-1))
        if logits:
            policy_logits = torch.stack(logits)
        else:
            policy_logits = torch.empty((0,), dtype=context.dtype, device=context.device)
        value = torch.tanh(self.value_head(context).squeeze(-1))
        return policy_logits, value

    def forward(self, state: GameState, actions: Optional[List[Action]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.policy_value(state, actions)

    def config_dict(self) -> dict:
        return {
            "architecture": "residual-mean-message-passing",
            "hidden_size": self.hidden_size,
            "message_layers": self.message_layer_count,
            "node_features": NODE_FEATURES,
            "global_features": GLOBAL_FEATURES,
            "action_head": "dynamic-place-and-relocate-pair",
        }
