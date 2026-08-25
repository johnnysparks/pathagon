"""Fast regression tests for the dynamic graph and rules adapter."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from .game import Action, BoardConfig, GameState, Player, repetition_key
from .graph import build_graph
from .cnn_model import PathagonCNN
from .data import load_replay_examples
from .model import PathagonGNN
from .selfplay import SearchExample, game_record


class PipelineTest(unittest.TestCase):
    def test_cnn_scores_7x7_actions_and_rejects_other_sizes(self) -> None:
        model = PathagonCNN(hidden_size=16, residual_blocks=1)
        state = GameState.initial(BoardConfig(7, 14))
        logits, value = model(state)
        self.assertEqual(tuple(logits.shape), (49,))
        self.assertEqual(tuple(value.shape), ())
        self.assertEqual(model.config_dict()["architecture"], "residual-cnn-7x7")
        relocation_state = GameState.initial(BoardConfig(7, 1))
        relocation_state = relocation_state.apply_legal(Action.place(0)).apply_legal(Action.place(1))
        relocation_logits, _ = model(relocation_state)
        self.assertEqual(tuple(relocation_logits.shape), (47,))
        with self.assertRaisesRegex(ValueError, "requires a 7x7"):
            model(GameState.initial(BoardConfig(5)))

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

    def test_repetition_identity_ignores_only_ply(self) -> None:
        config = BoardConfig(5)
        first = GameState.initial(config)
        later = GameState(
            config,
            first.light,
            first.dark,
            first.reserves,
            first.turn,
            first.forbidden,
            first.last_relocated_to,
            first.last_capture + 1,
            Player.DARK,
            first.winner,
            first.ply + 10,
        )
        self.assertEqual(repetition_key(first), repetition_key(later))

    def test_selfplay_ply_limit_can_override_the_size_default(self) -> None:
        self.assertEqual(BoardConfig(7, 14).max_plies, 196)
        self.assertEqual(BoardConfig(7, 14, 100).max_plies, 100)
        with self.assertRaisesRegex(ValueError, "ply_limit"):
            BoardConfig(7, 14, -1)

    def test_search_policy_survives_game_archive_and_replay_loading(self) -> None:
        config = BoardConfig(3, 3, 12)
        state = GameState.initial(config)
        actions = tuple(state.legal_actions())
        policy = tuple(0.2 if index == 0 else 0.1 for index in range(len(actions)))
        action_values = tuple((index - 4) / 10 for index in range(len(actions)))
        action_visits = tuple(index for index in range(len(actions)))
        example = SearchExample(state, actions, policy, actions[0], 0.0, action_values, action_visits)
        record = game_record([example], state.apply_legal(actions[0]), seed=7, simulations=8)
        self.assertEqual(record["moves"][0]["policy"], list(policy))
        self.assertEqual(record["moves"][0]["actionValues"], list(action_values))
        self.assertEqual(record["moves"][0]["actionVisits"], list(action_visits))
        self.assertEqual(record["moves"][0]["actionValueSource"], "mcts-root-q-v1")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "policy.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            examples = load_replay_examples(path)
        self.assertEqual(examples[0].policy, policy)
        self.assertEqual(examples[0].policy_actions, actions)
        self.assertEqual(examples[0].action_values, action_values)
        self.assertEqual(examples[0].action_visits, action_visits)
        self.assertEqual(examples[0].action_value_actions, actions)


if __name__ == "__main__":
    unittest.main()
