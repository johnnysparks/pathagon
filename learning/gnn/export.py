"""Export the deployable 7x7 policy/value model for Rust and WASM inference."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch

from .cnn_model import BOARD_FEATURES, BOARD_SIZE, PathagonCNN
from .game import Action, BoardConfig, GameState
from .graph import GLOBAL_FEATURES, build_graph
from .train import choose_device, load_model, model_state_hash


EXPORT_SCHEMA = "pathagon-policy-value-v1"
MAX_ACTIONS = BOARD_SIZE ** 4
DEPLOYED_MAX_PLIES = 180


class ExportableCNN(torch.nn.Module):
    """Tensor-only wrapper around the training model's dynamic action heads."""

    def __init__(self, model: PathagonCNN) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        board_features: torch.Tensor,
        global_features: torch.Tensor,
        action_specs: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.policy_value_tensors(board_features, global_features, action_specs, action_mask)


def tensor_inputs(state: GameState) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[Action]]:
    if state.config.size != BOARD_SIZE:
        raise ValueError(f"export model requires a {BOARD_SIZE}x{BOARD_SIZE} state")
    graph = build_graph(state)
    board = torch.cat((graph.node_features[: graph.board_nodes, :13], graph.node_features[: graph.board_nodes, 14:17]), dim=1)
    board_features = board.reshape(BOARD_SIZE, BOARD_SIZE, BOARD_FEATURES).permute(2, 0, 1).unsqueeze(0).contiguous()
    global_features = graph.global_features.unsqueeze(0).contiguous()
    actions = list(state.legal_actions())
    if len(actions) > MAX_ACTIONS:
        raise ValueError(f"state has {len(actions)} actions, model capacity is {MAX_ACTIONS}")
    action_specs = torch.zeros((1, MAX_ACTIONS, 3), dtype=torch.float32)
    action_mask = torch.zeros((1, MAX_ACTIONS), dtype=torch.float32)
    for index, action in enumerate(actions):
        action_specs[0, index] = torch.tensor(
            [0, 0, action.to] if action.kind == 0 else [1, action.from_square, action.to],
            dtype=torch.float32,
        )
        action_mask[0, index] = 1.0
    return board_features, global_features, action_specs, action_mask, actions


def export_checkpoint(checkpoint: Path, output: Path, device: torch.device) -> Dict[str, object]:
    model = load_model(checkpoint, device)
    if not isinstance(model, PathagonCNN):
        raise ValueError("Rust deployment currently supports only the fixed 7x7 CNN")
    model.eval()
    wrapper = ExportableCNN(model).to(device).eval()
    state = GameState.initial(BoardConfig(size=BOARD_SIZE, reserve_per_player=14, ply_limit=DEPLOYED_MAX_PLIES))
    raw_inputs = tensor_inputs(state)
    inputs = tuple(value.to(device) for value in raw_inputs[:4])
    actions = raw_inputs[4]
    output.parent.mkdir(parents=True, exist_ok=True)
    export_kwargs = {
        "input_names": ["board_features", "global_features", "action_specs", "action_mask"],
        "output_names": ["policy_logits", "value"],
        "opset_version": 18,
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = True
    torch.onnx.export(wrapper, inputs, str(output), **export_kwargs)

    with torch.no_grad():
        expected_logits, expected_value = model.policy_value(state, actions)
        actual_logits, actual_value = wrapper(*inputs)
    legal_count = len(actions)
    if not torch.allclose(actual_logits[0, :legal_count], expected_logits, rtol=1e-4, atol=1e-5):
        raise AssertionError("export wrapper policy does not match the training model")
    if not torch.allclose(actual_value.reshape(-1), expected_value.reshape(-1), rtol=1e-4, atol=1e-5):
        raise AssertionError("export wrapper value does not match the training model")

    artifact_hash = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest: Dict[str, object] = {
        "schemaVersion": 1,
        "schema": EXPORT_SCHEMA,
        "format": "onnx",
        "architecture": model.config_dict(),
        "boardSize": BOARD_SIZE,
        "maxPlies": DEPLOYED_MAX_PLIES,
        "maxActions": MAX_ACTIONS,
        "boardFeatures": BOARD_FEATURES,
        "globalFeatures": GLOBAL_FEATURES,
        "actionFeatures": 3,
        "outputs": {"policy": "masked logits in action_specs order", "value": "side-to-move scalar in [-1,1]"},
        "sourceModelHash": model_state_hash(model),
        "artifactHash": f"sha256:{artifact_hash}",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    try:
        manifest = export_checkpoint(args.checkpoint.resolve(), args.output.resolve(), device)
    except ModuleNotFoundError as error:
        if error.name == "onnx":
            raise SystemExit("ONNX export requires the optional 'onnx' package in the learner environment") from error
        raise
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()
