from __future__ import annotations

import unittest

import torch

from .game import BoardConfig, GameState, Player
from .mcts import PUCTSearch
from .tactics import (
    connection_distance_delta,
    immediate_winning_actions,
    one_away_path_actions,
    tactical_root,
)


def mask(squares: tuple[int, ...]) -> int:
    return sum(1 << square for square in squares)


class ZeroModel:
    def policy_value(self, _state, actions):
        return torch.zeros(len(actions)), torch.tensor(0.0)


class TacticsTest(unittest.TestCase):
    def setUp(self) -> None:
        config = BoardConfig(4, 5, 64)
        self.immediate = GameState(
            config, mask((4, 8, 12, 2, 10)), mask((1, 3, 6, 9, 14)), (0, 0), Player.LIGHT, ply=20
        )
        self.block = GameState(
            config, mask((5, 7, 9, 11, 15)), mask((1, 2, 3, 6, 10)), (0, 0), Player.LIGHT, ply=20
        )
        self.fork = GameState(
            config, mask((4, 5, 8, 10, 15)), mask((2, 3, 6, 9, 14)), (0, 0), Player.LIGHT, ply=20
        )

    def test_bit_masks_prefilter_immediate_relocations(self) -> None:
        authoritative = set(immediate_winning_actions(self.immediate, Player.LIGHT))
        candidates = set(one_away_path_actions(self.immediate, Player.LIGHT))
        self.assertEqual({action.short() for action in authoritative}, {"R2>0", "R10>0"})
        self.assertTrue(authoritative <= candidates)

    def test_tactical_root_finds_block_and_forced_fork(self) -> None:
        block = tactical_root(self.block)
        fork = tactical_root(self.fork)
        self.assertEqual(len(block.immediate_wins), 0)
        self.assertEqual(len(block.forced_blocks), 5)
        self.assertEqual(len(fork.immediate_wins), 0)
        self.assertEqual(len(fork.forced_forks), 2)
        self.assertEqual(block.root_action_count, 30)
        self.assertEqual(fork.root_action_count, 30)
        self.assertGreater(block.root_reply_edges, 0)
        self.assertGreater(fork.root_reply_edges, 0)

    def test_connection_distance_delta_is_action_aware(self) -> None:
        action = next(action for action in self.immediate.legal_actions() if action.short() == "R2>0")
        own_before, own_after, opponent_before, opponent_after = connection_distance_delta(self.immediate, action)
        self.assertGreater(own_before, 0)
        self.assertEqual(own_after, 0)
        self.assertGreaterEqual(opponent_before, 0)
        self.assertGreaterEqual(opponent_after, 0)

    def test_mcts_tactical_guard_emits_only_exact_priority_actions(self) -> None:
        search = PUCTSearch(ZeroModel(), simulations=0, tactical_guard=True)
        _, actions, probabilities = search.run(self.block)
        allowed = set(tactical_root(self.block).priority_actions)
        self.assertTrue(allowed)
        for action, probability in zip(actions, probabilities):
            self.assertEqual(probability > 0.0, action in allowed)


if __name__ == "__main__":
    unittest.main()
