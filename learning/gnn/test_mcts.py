from __future__ import annotations

import unittest

import torch

from .game import BoardConfig, GameState, Player
from .mcts import PUCTSearch


class ZeroModel:
    def policy_value(self, _state, actions):
        return torch.zeros(len(actions)), torch.tensor(0.0)


class MctsTest(unittest.TestCase):
    def test_root_afterstate_scan_seeds_every_legal_child(self) -> None:
        state = GameState.initial(BoardConfig(3, 3, 12))
        root, actions, probabilities = PUCTSearch(ZeroModel(), simulations=0).run(state)

        self.assertEqual(len(root.children), len(actions))
        self.assertAlmostEqual(sum(probabilities), 1.0)
        for action in actions:
            child = root.children[action]
            self.assertEqual(child.state, state.apply_legal(action))
            self.assertIsNotNone(child.seeded_value)
            self.assertEqual(child.visit_count, 0)

    def test_root_action_values_are_flipped_into_root_perspective(self) -> None:
        state = GameState.initial(BoardConfig(3, 3, 12))
        root, actions, _ = PUCTSearch(ZeroModel(), simulations=0).run(state)
        values, visits = PUCTSearch.root_action_values(root, actions)

        self.assertEqual(len(values), len(actions))
        self.assertEqual(visits, [0] * len(actions))
        for action, value in zip(actions, values):
            self.assertAlmostEqual(value, -root.children[action].seeded_value)

    def test_search_treats_the_ply_cap_as_a_draw_terminal(self) -> None:
        config = BoardConfig(3, 3, 12)
        state = GameState(
            config=config,
            light=0,
            dark=0,
            reserves=(3, 3),
            turn=Player.LIGHT,
            ply=config.max_plies,
        )
        root, actions, probabilities = PUCTSearch(None, simulations=8).run(state)
        self.assertTrue(root.expanded)
        self.assertEqual(actions, [])
        self.assertEqual(probabilities, [])


if __name__ == "__main__":
    unittest.main()
