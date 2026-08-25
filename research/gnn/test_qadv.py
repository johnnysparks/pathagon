from __future__ import annotations

import unittest
import random

import torch

from .data import ReplayExample
from .game import BoardConfig, GameState
from .league import QAdvAgent
from .model import PathagonGNN
from .train import train_qadv_replay
from .transition import TRANSITION_FEATURES, transition_features


class QAdvTest(unittest.TestCase):
    def test_transition_features_are_action_aligned(self) -> None:
        state = GameState.initial(BoardConfig(5, 8))
        actions = list(state.legal_actions())
        features = transition_features(state, actions)
        self.assertEqual(tuple(features.shape), (len(actions), TRANSITION_FEATURES))
        self.assertTrue(torch.isfinite(features).all())
        self.assertEqual(features[:, 0].tolist(), [1.0] * len(actions))

    def test_qadv_head_scores_each_legal_action(self) -> None:
        state = GameState.initial(BoardConfig(7, 14))
        actions = list(state.legal_actions())
        model = PathagonGNN(hidden_size=8, message_layers=1, qadv=True)
        logits, value, q_values, advantages = model.policy_value_q(state, actions)
        self.assertEqual(tuple(logits.shape), (len(actions),))
        self.assertEqual(tuple(q_values.shape), (len(actions),))
        self.assertEqual(tuple(advantages.shape), (len(actions),))
        self.assertEqual(tuple(value.shape), ())
        self.assertTrue(torch.all(q_values <= 1.0))
        self.assertTrue(torch.all(q_values >= -1.0))
        self.assertAlmostEqual(float(advantages.mean().detach()), 0.0, places=5)

    def test_qadv_training_uses_visit_weighted_targets_and_agent_selects_legal_move(self) -> None:
        state = GameState.initial(BoardConfig(5, 8))
        actions = tuple(state.legal_actions())
        example = ReplayExample(
            state=state,
            action=actions[0],
            value=1.0,
            seed=1,
            policy=tuple(1.0 / len(actions) for _ in actions),
            policy_actions=actions,
            action_values=tuple((index % 5) / 5.0 for index in range(len(actions))),
            action_visits=tuple(1 if index < 8 else 0 for index in range(len(actions))),
            action_value_actions=actions,
        )
        model = PathagonGNN(hidden_size=8, message_layers=1, qadv=True)
        metrics = train_qadv_replay(model, [example], steps=1, learning_rate=1.0, seed=3, symmetry_augmentation=False)
        self.assertGreater(metrics["q_loss"], 0.0)
        action = QAdvAgent(model).choose_action(state, random.Random(0), set())
        self.assertIn(action, actions)


if __name__ == "__main__":
    unittest.main()
