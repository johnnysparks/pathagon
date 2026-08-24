"""A small fixed-7x7 convolutional policy/value network."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .game import Action, GameState
from .graph import GLOBAL_FEATURES, build_graph


BOARD_SIZE = 7
BOARD_FEATURES = 16


class ResidualConvBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.convolution1 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.normalization1 = nn.GroupNorm(4, hidden_size)
        self.convolution2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.normalization2 = nn.GroupNorm(4, hidden_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = features
        features = F.gelu(self.normalization1(self.convolution1(features)))
        features = self.normalization2(self.convolution2(features))
        return F.gelu(residual + features)


class PathagonCNN(nn.Module):
    """A compact 7x7 CNN with the same dynamic action interface as the GNN.

    The convolutional trunk produces one embedding per board square. The
    policy head still scores the caller-provided legal actions, so placement
    and relocation moves remain represented identically to the GNN. Unlike
    ``PathagonGNN``, this model intentionally rejects other board sizes.
    """

    def __init__(self, hidden_size: int = 32, residual_blocks: int = 4, board_size: int = BOARD_SIZE) -> None:
        super().__init__()
        if board_size != BOARD_SIZE:
            raise ValueError("PathagonCNN is fixed to a 7x7 board")
        if hidden_size < 4 or hidden_size % 4:
            raise ValueError("CNN hidden_size must be a positive multiple of 4")
        if residual_blocks < 1:
            raise ValueError("residual_blocks must be positive")
        self.board_size = board_size
        self.hidden_size = hidden_size
        self.residual_block_count = residual_blocks
        self.input = nn.Conv2d(BOARD_FEATURES, hidden_size, kernel_size=3, padding=1)
        self.input_normalization = nn.GroupNorm(4, hidden_size)
        self.blocks = nn.ModuleList(ResidualConvBlock(hidden_size) for _ in range(residual_blocks))
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
        if state.config.size != self.board_size:
            raise ValueError(f"PathagonCNN requires a {self.board_size}x{self.board_size} state")
        device = next(self.parameters()).device
        graph = build_graph(state, device=device)
        board = graph.node_features[: graph.board_nodes]
        # Graph construction already centralizes the rules features. The CNN
        # uses the 13 local channels plus size/turn channels; global reserves,
        # capture, and ply information enter through the pooled context.
        board = torch.cat((board[:, :13], board[:, 14:17]), dim=1)
        features = board.reshape(self.board_size, self.board_size, BOARD_FEATURES).permute(2, 0, 1).unsqueeze(0)
        features = F.gelu(self.input_normalization(self.input(features)))
        for block in self.blocks:
            features = block(features)
        nodes = features[0].permute(1, 2, 0).reshape(graph.board_nodes, self.hidden_size)
        pooled = torch.cat((nodes.mean(dim=0), nodes.amax(dim=0), graph.global_features), dim=0)
        context = self.context(pooled)
        return nodes, context

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
            "architecture": "residual-cnn-7x7",
            "board_size": self.board_size,
            "hidden_size": self.hidden_size,
            "residual_blocks": self.residual_block_count,
            "input_features": BOARD_FEATURES,
            "global_features": GLOBAL_FEATURES,
            "action_head": "dynamic-place-and-relocate-pair",
        }

