from __future__ import annotations

import unittest

import torch

from .game import BoardConfig, GameState, Player, repetition_key
from .mcts import PUCTSearch
from .solver import ExactSolver
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

    def test_generic_solver_labels_the_fixtures_without_tactic_predicates(self) -> None:
        solver = ExactSolver(horizon=3)
        cases = (
            (self.immediate, 1, {"R2>0", "R10>0"}),
            (self.block, 0, {"R5>0", "R7>0", "R9>0", "R11>0", "R15>0"}),
            (self.fork, 1, {"R10>12", "R15>12"}),
        )
        for state, expected_outcome, expected_actions in cases:
            analysis = solver.analyze(state)
            self.assertEqual(analysis.result.outcome, expected_outcome)
            self.assertEqual({action.short() for action in analysis.optimal_actions}, expected_actions)
        self.assertGreater(solver.stats.cache_hits, 0)

    def test_solver_respects_threefold_history_before_search(self) -> None:
        state = GameState.initial(BoardConfig(3, 3, 12))
        solver = ExactSolver(horizon=3)
        result = solver.solve(state, {repetition_key(state): 2})
        self.assertEqual(result.outcome, 0)
        self.assertEqual(solver.stats.nodes, 1)


if __name__ == "__main__":
    unittest.main()
