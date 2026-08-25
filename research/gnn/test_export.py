from __future__ import annotations

import unittest

import torch

from .cnn_model import PathagonCNN
from .export import ExportableCNN, tensor_inputs
from .game import BoardConfig, GameState


class ExportTest(unittest.TestCase):
    def test_export_wrapper_matches_training_model_on_initial_position(self) -> None:
        model = PathagonCNN(hidden_size=8, residual_blocks=1)
        model.eval()
        state = GameState.initial(BoardConfig(size=7, reserve_per_player=14, ply_limit=180))
        board, global_features, action_specs, action_mask, actions = tensor_inputs(state)
        wrapper = ExportableCNN(model).eval()
        with torch.no_grad():
            expected_logits, expected_value = model.policy_value(state, actions)
            actual_logits, actual_value = wrapper(board, global_features, action_specs, action_mask)
        self.assertTrue(torch.allclose(actual_logits[0, : len(actions)], expected_logits, rtol=1e-4, atol=1e-5))
        self.assertTrue(torch.allclose(actual_value.reshape(-1), expected_value.reshape(-1), rtol=1e-4, atol=1e-5))
        self.assertTrue(torch.all(actual_logits[0, len(actions):] < -1e8))


if __name__ == "__main__":
    unittest.main()
