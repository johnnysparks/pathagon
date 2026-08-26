"""Export the fixed 7x7 GNN policy/value path for native Rust inference.

The Q/advantage checkpoint shares this policy/value trunk with the Python
self-play search.  The first native port intentionally exports that trunk;
root-Q targets still come from Rust PUCT visits and child values, while the
Q/A action head is added in a subsequent ABI revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .game import BoardConfig, GameState
from .graph import NODE_FEATURES, build_graph
from .model import PathagonGNN
from .train import load_model, model_state_hash


BOARD_SIZE = 7
BOARD_NODES = BOARD_SIZE * BOARD_SIZE
GRAPH_NODES = BOARD_NODES + 4
GLOBAL_FEATURES = 8
ACTION_FEATURES = 3
MAX_ACTIONS = BOARD_NODES * BOARD_NODES
EXPORT_SCHEMA = "pathagon-gnn-policy-value-v1"


def graph_adjacency() -> torch.Tensor:
    """Build the dense normalized adjacency used by the message layers."""

    state = GameState.initial(BoardConfig(size=BOARD_SIZE, reserve_per_player=14, ply_limit=196))
    graph = build_graph(state)
    adjacency = torch.zeros((GRAPH_NODES, GRAPH_NODES), dtype=torch.float32)
    source, destination = graph.edge_index
    adjacency.index_put_((destination, source), torch.ones_like(source, dtype=torch.float32), accumulate=True)
    return adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)


class ExportableGNN(nn.Module):
    """Tensor-only wrapper around the shared GNN policy/value path."""

    def __init__(self, model: PathagonGNN) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("adjacency", graph_adjacency())

    def encode(self, node_features: torch.Tensor, global_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        nodes = self.model.input(node_features)
        adjacency = self.adjacency.unsqueeze(0).expand(nodes.shape[0], -1, -1)
        for layer in self.model.layers:
            aggregate = torch.bmm(adjacency, nodes)
            update = F.gelu(layer.update(torch.cat((nodes, aggregate), dim=-1)))
            nodes = layer.norm(nodes + update)
        pooled = torch.cat((nodes.mean(dim=1), nodes.amax(dim=1), global_features), dim=-1)
        context = self.model.context(pooled)
        return nodes[:, :BOARD_NODES, :], context

    def forward(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        action_specs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        board_nodes, context = self.encode(node_features, global_features)
        action_indices = action_specs[:, :, 1:].clamp(0, BOARD_NODES - 1).to(torch.int64)
        from_index = action_indices[:, :, 0].unsqueeze(-1).expand(-1, -1, board_nodes.shape[-1])
        to_index = action_indices[:, :, 1].unsqueeze(-1).expand(-1, -1, board_nodes.shape[-1])
        from_nodes = torch.gather(board_nodes, 1, from_index)
        to_nodes = torch.gather(board_nodes, 1, to_index)
        context_actions = context.unsqueeze(1).expand(-1, MAX_ACTIONS, -1)
        place_features = torch.cat((to_nodes, context_actions), dim=-1)
        relocate_features = torch.cat((from_nodes, to_nodes, context_actions), dim=-1)
        place_logits = self.model.place_head(place_features).squeeze(-1)
        relocate_logits = self.model.relocate_head(relocate_features).squeeze(-1)
        policy_logits = torch.where(action_specs[:, :, 0] < 0.5, place_logits, relocate_logits)
        value = torch.tanh(self.model.value_head(context).squeeze(-1))
        # Keep padding numerically harmless for runtimes that inspect all slots.
        return policy_logits * action_mask, value


def tensor_inputs(state: GameState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
    graph = build_graph(state)
    node_features = graph.node_features.unsqueeze(0).contiguous()
    global_features = graph.global_features.unsqueeze(0).contiguous()
    actions = list(state.legal_actions())
    action_specs = torch.zeros((1, MAX_ACTIONS, ACTION_FEATURES), dtype=torch.float32)
    action_mask = torch.zeros((1, MAX_ACTIONS), dtype=torch.float32)
    for index, action in enumerate(actions):
        action_specs[0, index] = torch.tensor(
            [0, 0, action.to] if action.kind == 0 else [1, action.from_square, action.to],
            dtype=torch.float32,
        )
        action_mask[0, index] = 1.0
    return node_features, global_features, action_specs, action_mask, actions


def export_checkpoint(checkpoint: Path, output: Path, device: torch.device) -> Dict[str, object]:
    model = load_model(checkpoint, device, qadv=True)
    if not isinstance(model, PathagonGNN):
        raise ValueError("GNN export requires a GNN checkpoint")
    model.eval()
    wrapper = ExportableGNN(model).to(device).eval()
    state = GameState.initial(BoardConfig(size=BOARD_SIZE, reserve_per_player=14, ply_limit=196))
    raw_inputs = tensor_inputs(state)
    inputs = tuple(value.to(device) for value in raw_inputs[:4])
    actions = raw_inputs[4]
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        inputs,
        str(output),
        input_names=["node_features", "global_features", "action_specs", "action_mask"],
        output_names=["policy_logits", "value"],
        opset_version=18,
        do_constant_folding=True,
        external_data=False,
    )
    with torch.no_grad():
        expected_logits, expected_value = model.policy_value(state, actions)
        actual_logits, actual_value = wrapper(*inputs)
    legal_count = len(actions)
    if not torch.allclose(actual_logits[0, :legal_count], expected_logits, rtol=1e-4, atol=1e-5):
        raise AssertionError("GNN export policy does not match the Python model")
    if not torch.allclose(actual_value.reshape(-1), expected_value.reshape(-1), rtol=1e-4, atol=1e-5):
        raise AssertionError("GNN export value does not match the Python model")
    artifact_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest: Dict[str, object] = {
        "schemaVersion": 1,
        "schema": EXPORT_SCHEMA,
        "format": "onnx",
        "sourceModelHash": model_state_hash(model),
        "artifactHash": f"sha256:{artifact_hash}",
        "architecture": model.config_dict(),
        "boardSize": BOARD_SIZE,
        "reservePerPlayer": 14,
        "graphNodes": GRAPH_NODES,
        "nodeFeatures": NODE_FEATURES,
        "globalFeatures": GLOBAL_FEATURES,
        "actionFeatures": ACTION_FEATURES,
        "maxActions": MAX_ACTIONS,
        "outputs": {"policy": "masked logits in legal-action order", "value": "side-to-move scalar in [-1,1]"},
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    manifest = export_checkpoint(args.checkpoint.resolve(), args.output.resolve(), torch.device(args.device))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
