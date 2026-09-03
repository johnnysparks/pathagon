"""Compact action-conditioned JEPA over exact Rust afterstates.

This module deliberately reuses the existing Pathagon GNN as the online
encoder. The JSONL target state is emitted by Rust; the mirrored Python rules
adapter is used only to decode and audit the contract, never to generate the
world-model target.
"""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
from torch import nn
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_ROOT = REPO_ROOT / "research/20260824-gnn-cnn-lab"
if str(LEGACY_ROOT) not in sys.path:
    sys.path.insert(0, str(LEGACY_ROOT))

from python.evaluation import evaluate_position, normalize_heuristic  # noqa: E402
from python.game import Action, BoardConfig, GameState, Player  # noqa: E402
from python.model import PathagonGNN  # noqa: E402


@dataclass(frozen=True)
class RustTransition:
    state: GameState
    action: Action
    next_state: GameState
    game: int
    seed: int
    selected_for_rollout: bool


def parse_action(raw: dict) -> Action:
    kind = raw.get("kind")
    if kind == "place":
        return Action.place(int(raw["to"]))
    if kind == "relocate":
        return Action.relocate(int(raw["from"]), int(raw["to"]))
    raise ValueError(f"unknown action kind: {kind!r}")


def parse_player(raw: str | None) -> Player | None:
    if raw is None:
        return None
    if raw == "light":
        return Player.LIGHT
    if raw == "dark":
        return Player.DARK
    raise ValueError(f"unknown player: {raw!r}")


def parse_state(raw: dict) -> GameState:
    config = BoardConfig(
        size=int(raw["boardSize"]),
        reserve_per_player=int(raw["reservePerPlayer"]),
        ply_limit=int(raw["maxPlies"]),
    )
    reserves = tuple(int(value) for value in raw["reserve"])
    if len(reserves) != 2:
        raise ValueError("state reserve must contain two values")
    markers = tuple(
        None if value is None else int(value) for value in raw.get("lastRelocatedTo", [None, None])
    )
    if len(markers) != 2:
        raise ValueError("state lastRelocatedTo must contain two values")
    return GameState(
        config=config,
        light=int(raw["light"]),
        dark=int(raw["dark"]),
        reserves=reserves,
        turn=parse_player(raw["turn"]) or Player.LIGHT,
        forbidden=int(raw.get("forbidden", 0)),
        last_relocated_to=markers,
        last_capture=int(raw.get("lastCapture", 0)),
        last_player=parse_player(raw.get("lastPlayer")),
        winner=parse_player(raw.get("winner")),
        ply=int(raw.get("ply", 0)),
    )


def load_rust_transitions(path: Path, verify_mirror: bool = True) -> list[RustTransition]:
    """Load Rust-emitted rows and optionally audit them against the mirror."""

    rows: list[RustTransition] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("format") != "pathagon-rust-jepa-afterstate-v1":
                raise ValueError(f"{path}:{line_number}: unsupported transition format")
            state = parse_state(raw["state"])
            action = parse_action(raw["action"])
            next_state = parse_state(raw["nextState"])
            if action not in state.legal_actions():
                raise ValueError(f"{path}:{line_number}: Rust row contains an illegal action")
            if verify_mirror and state.apply_legal(action) != next_state:
                raise ValueError(
                    f"{path}:{line_number}: mirrored rules disagree with Rust afterstate"
                )
            rows.append(
                RustTransition(
                    state=state,
                    action=action,
                    next_state=next_state,
                    game=int(raw["game"]),
                    seed=int(raw["seed"]),
                    selected_for_rollout=bool(raw.get("selectedForRollout", False)),
                )
            )
    if not rows:
        raise ValueError(f"{path}: transition corpus is empty")
    return rows


class ActionConditionedJEPA(nn.Module):
    """An EMA-target, action-conditioned predictor in a compact latent space."""

    def __init__(
        self,
        hidden_size: int = 64,
        message_layers: int = 8,
        embedding_size: int | None = None,
    ) -> None:
        super().__init__()
        self.embedding_size = embedding_size or hidden_size
        self.online = PathagonGNN(hidden_size=hidden_size, message_layers=message_layers)
        self.target = copy.deepcopy(self.online)
        self.online_projection = nn.Sequential(
            nn.Linear(hidden_size, self.embedding_size),
            nn.LayerNorm(self.embedding_size),
        )
        self.target_projection = copy.deepcopy(self.online_projection)
        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_size + hidden_size * 2 + 4, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.embedding_size),
        )
        # These heads are the deployable JEPA opponent surface. The original
        # smoke checkpoint only learned an embedding prediction objective; it
        # deliberately cannot be used as an opponent until these action-aware
        # outputs are trained and exported.
        action_input_size = self.embedding_size + hidden_size * 2 + 4
        self.action_rank_head = nn.Sequential(
            nn.Linear(action_input_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )
        self.action_value_head = nn.Sequential(
            nn.Linear(action_input_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1),
        )
        self._freeze_target()

    def _freeze_target(self) -> None:
        self.target.eval()
        self.target_projection.eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        for parameter in self.target_projection.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "ActionConditionedJEPA":
        super().train(mode)
        self._freeze_target()
        return self

    def update_target(self, momentum: float = 0.996) -> None:
        if not 0.0 < momentum < 1.0:
            raise ValueError("target momentum must be between zero and one")
        with torch.no_grad():
            for online, target in zip(self.online.parameters(), self.target.parameters()):
                target.mul_(momentum).add_(online, alpha=1.0 - momentum)
            for online, target in zip(
                self.online_projection.parameters(), self.target_projection.parameters()
            ):
                target.mul_(momentum).add_(online, alpha=1.0 - momentum)
        self._freeze_target()

    def _action_features(
        self,
        state: GameState,
        action: Action,
        board_nodes: torch.Tensor,
    ) -> torch.Tensor:
        denominator = float(max(1, state.config.size - 1))
        to_node = board_nodes[action.to]
        if action.kind == 0:
            from_node = torch.zeros_like(to_node)
            from_square = 0.0
            has_source = 0.0
        else:
            from_node = board_nodes[action.from_square]
            from_square = action.from_square / denominator
            has_source = 1.0
        scalar = torch.tensor(
            [float(action.kind), has_source, from_square, action.to / denominator],
            dtype=board_nodes.dtype,
            device=board_nodes.device,
        )
        return torch.cat((to_node, from_node, scalar), dim=0)

    def forward(self, rows: Sequence[RustTransition]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not rows:
            raise ValueError("JEPA batch is empty")
        predictions = []
        targets = []
        online_embeddings = []
        for row in rows:
            board_nodes, context = self.online.encode(row.state)
            online_z = self.online_projection(context)
            action_features = self._action_features(row.state, row.action, board_nodes)
            predictions.append(self.predictor(torch.cat((online_z, action_features), dim=0)))
            online_embeddings.append(online_z)
            with torch.no_grad():
                _, target_context = self.target.encode(row.next_state)
                targets.append(self.target_projection(target_context))
        return torch.stack(predictions), torch.stack(targets), torch.stack(online_embeddings)

    def action_rank_value(self, state: GameState, actions: Sequence[Action] | None = None):
        """Score legal actions with the trained JEPA afterstate heads.

        The Rust exporter supplies the afterstate training rows and the Rust
        inference ABI later supplies the same state/action tensors. This
        method is intentionally separate from ``policy_value`` so an
        embedding-only checkpoint cannot accidentally masquerade as a player.
        """

        legal = list(actions) if actions is not None else list(state.legal_actions())
        board_nodes, context = self.online.encode(state)
        online_z = self.online_projection(context)
        action_inputs = torch.stack(
            [torch.cat((online_z, self._action_features(state, action, board_nodes)), dim=0) for action in legal]
        ) if legal else torch.empty((0, online_z.shape[-1] + board_nodes.shape[-1] * 2 + 4), device=online_z.device)
        rank_logits = self.action_rank_head(action_inputs).squeeze(-1)
        values = torch.tanh(self.action_value_head(action_inputs).squeeze(-1))
        return rank_logits, values

    def policy_value(self, state: GameState, actions: Sequence[Action] | None = None):
        """Delegate the deployable policy/value interface to the online trunk."""

        return self.online.policy_value(state, list(actions) if actions is not None else None)


def _covariance_loss(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] < 2:
        return values.sum() * 0.0
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / float(values.shape[0] - 1)
    diagonal = torch.diagonal(covariance)
    off_diagonal = covariance.flatten()[:-1].view(covariance.shape[0] - 1, covariance.shape[0] + 1)[:, 1:].flatten()
    # The reshape above drops the diagonal from a row-major square matrix.
    del diagonal
    return off_diagonal.pow(2).mean()


def _variance_loss(values: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    if values.shape[0] < 2:
        return values.sum() * 0.0
    standard_deviation = torch.sqrt(values.var(dim=0, unbiased=False) + 1.0e-4)
    return F.relu(target_std - standard_deviation).mean()


def jepa_loss(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    predicted = F.normalize(predictions, dim=-1)
    target = F.normalize(targets.detach(), dim=-1)
    invariance = F.mse_loss(predicted, target)
    variance = _variance_loss(predicted) + _variance_loss(target)
    covariance = _covariance_loss(predicted) + _covariance_loss(target)
    total = invariance + 25.0 * variance + covariance
    return {
        "loss": total,
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance,
    }


def train_jepa(
    model: ActionConditionedJEPA,
    rows: Sequence[RustTransition],
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    momentum: float = 0.996,
) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot train on an empty transition corpus")
    if steps < 1 or batch_size < 2 or learning_rate <= 0.0:
        raise ValueError("steps, batch_size, and learning_rate are invalid")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=1.0e-4,
    )
    totals = {key: 0.0 for key in ("loss", "invariance", "variance", "covariance")}
    model.train()
    for _ in range(steps):
        indices = torch.randint(len(rows), (batch_size,), generator=generator).tolist()
        batch = [rows[index] for index in indices]
        predictions, targets, _ = model(batch)
        losses = jepa_loss(predictions, targets)
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad], 1.0
        )
        optimizer.step()
        model.update_target(momentum)
        for key in totals:
            totals[key] += float(losses[key].detach().cpu())
    return {key: value / steps for key, value in totals.items()}


def _afterstate_target(row: RustTransition) -> torch.Tensor:
    """Build a bounded preference/value label from a Rust-validated row."""

    return torch.tensor(
        normalize_heuristic(evaluate_position(row.next_state, row.state.turn)),
        dtype=torch.float32,
    )


def train_action_head(
    model: ActionConditionedJEPA,
    rows: Sequence[RustTransition],
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, float]:
    """Train the real JEPA action-ranking/value path on validated afterstates."""

    if not rows:
        raise ValueError("cannot train action heads on an empty transition corpus")
    if steps < 1 or batch_size < 1 or learning_rate <= 0.0:
        raise ValueError("action-head steps, batch size, and learning rate are invalid")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    trainable = list(model.online.parameters()) + list(model.online_projection.parameters())
    trainable += list(model.action_rank_head.parameters()) + list(model.action_value_head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1.0e-4)
    rank_total = 0.0
    value_total = 0.0
    model.train()
    for _ in range(steps):
        indices = torch.randint(len(rows), (batch_size,), generator=generator).tolist()
        batch = [rows[index] for index in indices]
        predictions = []
        values = []
        targets = []
        for row in batch:
            rank, value = model.action_rank_value(row.state, [row.action])
            predictions.append(rank.squeeze(0))
            values.append(value.squeeze(0))
            targets.append(_afterstate_target(row).to(value.device))
        predicted_rank = torch.stack(predictions)
        predicted_value = torch.stack(values)
        target = torch.stack(targets)
        # The ranking head receives the same monotonic teacher signal as the
        # value head, but remains an independent learned output for the UI.
        rank_loss = F.smooth_l1_loss(predicted_rank, target)
        value_loss = F.smooth_l1_loss(predicted_value, target)
        loss = rank_loss + value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        rank_total += float(rank_loss.detach().cpu())
        value_total += float(value_loss.detach().cpu())
    return {
        "rankLoss": rank_total / steps,
        "valueLoss": value_total / steps,
    }


def evaluate_jepa(
    model: ActionConditionedJEPA,
    rows: Sequence[RustTransition],
    batch_size: int = 32,
) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot evaluate an empty transition corpus")
    model.eval()
    totals = {key: 0.0 for key in ("loss", "invariance", "variance", "covariance")}
    count = 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            losses = jepa_loss(*model(rows[start : start + batch_size])[:2])
            size = len(rows[start : start + batch_size])
            count += size
            for key in totals:
                totals[key] += float(losses[key].cpu()) * size
    return {key: value / count for key, value in totals.items()}
