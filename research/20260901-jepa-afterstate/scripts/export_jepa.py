"""Export the trained JEPA afterstate action-ranking/value path for Rust."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT_DIR / "research/20260901-jepa-afterstate/python"
LEGACY_DIR = ROOT_DIR / "research/20260824-gnn-cnn-lab"
import sys

sys.path.insert(0, str(MODULE_DIR))
sys.path.insert(0, str(LEGACY_DIR))

from jepa_afterstate import ActionConditionedJEPA  # noqa: E402
from python.game import BoardConfig, GameState  # noqa: E402
from python.export_gnn import (  # noqa: E402
    ACTION_FEATURES,
    BOARD_NODES,
    GLOBAL_FEATURES,
    MAX_ACTIONS,
    ExportableGNN,
    tensor_inputs,
)
from python.train import model_state_hash  # noqa: E402


class ExportableJEPA(nn.Module):
    """Tensor-only wrapper for the trained JEPA action heads.

    Rust owns the legal action list and supplies the same fixed graph/action
    ABI used by the GNN exporter. The exported outputs remain aligned to that
    action order: rank logits first, then bounded afterstate values.
    """

    def __init__(self, model: ActionConditionedJEPA) -> None:
        super().__init__()
        self.encoder = ExportableGNN(model.online)
        self.projection = model.online_projection
        self.action_rank_head = model.action_rank_head
        self.action_value_head = model.action_value_head

    def forward(
        self,
        node_features: torch.Tensor,
        global_features: torch.Tensor,
        action_specs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        board_nodes, context = self.encoder.encode(node_features, global_features)
        indices = action_specs[:, :, 1:].clamp(0, BOARD_NODES - 1).to(torch.int64)
        from_index = indices[:, :, 0].unsqueeze(-1).expand(-1, -1, board_nodes.shape[-1])
        to_index = indices[:, :, 1].unsqueeze(-1).expand(-1, -1, board_nodes.shape[-1])
        from_nodes = torch.gather(board_nodes, 1, from_index)
        to_nodes = torch.gather(board_nodes, 1, to_index)
        is_relocate = (action_specs[:, :, 0] > 0.5).unsqueeze(-1)
        from_nodes = torch.where(is_relocate, from_nodes, torch.zeros_like(from_nodes))
        denominator = float(BOARD_NODES ** 0.5 - 1.0)
        kind = action_specs[:, :, 0:1]
        has_source = is_relocate.to(node_features.dtype)
        from_square = action_specs[:, :, 1:2] / denominator
        to_square = action_specs[:, :, 2:3] / denominator
        scalars = torch.cat((kind, has_source, from_square, to_square), dim=-1)
        action_features = torch.cat((to_nodes, from_nodes, scalars), dim=-1)
        online_z = self.projection(context).unsqueeze(1).expand(-1, MAX_ACTIONS, -1)
        conditioned = torch.cat((online_z, action_features), dim=-1)
        rank_logits = self.action_rank_head(conditioned).squeeze(-1) * action_mask
        values = torch.tanh(self.action_value_head(conditioned).squeeze(-1)) * action_mask
        return rank_logits, values


def load_checkpoint(path: Path, device: torch.device) -> ActionConditionedJEPA:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint.get("model_config", {})
    if config.get("action_head") != "afterstate-rank-and-value-v1":
        raise ValueError("checkpoint does not contain the trained JEPA action-ranking/value head")
    model = ActionConditionedJEPA(
        hidden_size=int(config.get("hidden_size", 64)),
        message_layers=int(config.get("message_layers", 8)),
        embedding_size=int(config.get("embedding_size", 64)),
    ).to(device)
    model.load_state_dict(checkpoint["jepa_state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    device = torch.device(args.device)
    model = load_checkpoint(checkpoint, device)
    wrapper = ExportableJEPA(model).to(device).eval()
    state = GameState.initial(BoardConfig(size=7, reserve_per_player=14, ply_limit=196))
    initial = tensor_inputs(state)
    inputs = tuple(value.to(device) for value in initial[:4])
    actions = initial[5]
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        inputs,
        str(output),
        input_names=["node_features", "global_features", "action_specs", "action_mask"],
        output_names=["rank_logits", "action_values"],
        opset_version=18,
        do_constant_folding=True,
        external_data=False,
    )
    with torch.no_grad():
        expected_rank, expected_value = model.action_rank_value(state, actions)
        actual_rank, actual_value = wrapper(*inputs)
    legal_count = len(actions)
    if not torch.allclose(actual_rank[0, :legal_count], expected_rank, rtol=1e-4, atol=1e-5):
        raise AssertionError("JEPA export rank head does not match the Python model")
    if not torch.allclose(actual_value[0, :legal_count], expected_value, rtol=1e-4, atol=1e-5):
        raise AssertionError("JEPA export value head does not match the Python model")
    artifact_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "schemaVersion": 1,
        "schema": "pathagon-jepa-afterstate-v1",
        "format": "onnx",
        "sourceModelHash": model_state_hash(model.online),
        "artifactHash": f"sha256:{artifact_hash}",
        "checkpointHash": f"sha256:{hashlib.sha256(checkpoint.read_bytes()).hexdigest()}",
        "boardSize": 7,
        "reservePerPlayer": 14,
        "graphNodes": 53,
        "nodeFeatures": 21,
        "globalFeatures": GLOBAL_FEATURES,
        "actionFeatures": ACTION_FEATURES,
        "maxActions": MAX_ACTIONS,
        "outputs": {
            "rankLogits": "action-ranking logits in legal-action order",
            "actionValues": "bounded afterstate values in [-1,1] in legal-action order",
        },
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
