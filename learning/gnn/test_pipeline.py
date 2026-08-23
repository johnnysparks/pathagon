"""Fast regression tests for the dynamic graph and rules adapter."""

from __future__ import annotations

import unittest

from .game import BoardConfig, GameState, Player
from .graph import build_graph
from .model import PathagonGNN


class PipelineTest(unittest.TestCase):
    def test_graph_and_policy_change_board_size_without_rebuilding_weights(self) -> None:
        model = PathagonGNN(hidden_size=16, message_layers=2)
        for size, expected_actions in ((5, 25), (7, 49)):
            state = GameState.initial(BoardConfig(size))
            graph = build_graph(state)
            logits, value = model(state)
            self.assertEqual(graph.board_nodes, size * size)
            self.assertEqual(len(state.legal_actions()), expected_actions)
            self.assertEqual(tuple(logits.shape), (expected_actions,))
            self.assertEqual(tuple(value.shape), ())

    def test_capture_and_connection_rules_are_size_parameterized(self) -> None:
        config = BoardConfig(5)
        state = GameState(config, 1 << 24, 1 << 23, (10, 10), Player.LIGHT)
        next_state = state.apply_legal(next(action for action in state.legal_actions() if action.to == 22))
        self.assertEqual(next_state.last_capture, 1)
        self.assertEqual(next_state.dark, 0)
        self.assertEqual(next_state.reserves[Player.DARK], 11)


if __name__ == "__main__":
    unittest.main()

