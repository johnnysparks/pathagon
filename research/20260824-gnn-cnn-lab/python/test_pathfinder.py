from __future__ import annotations

import unittest
import random

from .league import SorterPathfinderAgent
from .game import BoardConfig, GameState, Player
from .model import PathagonGNN
from .pathfinder import PathfinderGuide
from .tactics import tactical_root


def mask(squares: tuple[int, ...]) -> int:
    return sum(1 << square for square in squares)


class PathfinderGuideTest(unittest.TestCase):
    def test_scores_every_legal_root_action(self) -> None:
        state = GameState.initial(BoardConfig(3, 3, 12))
        actions = state.legal_actions()
        scores = PathfinderGuide(depth=2, beam_width=2, max_nodes=24).score_actions(state, actions)
        self.assertEqual(len(scores), len(actions))
        self.assertTrue(all(isinstance(score, float) for score in scores))

    def test_node_budget_is_respected(self) -> None:
        state = GameState.initial(BoardConfig(3, 3, 12))
        guide = PathfinderGuide(depth=3, beam_width=8, max_nodes=5)
        guide.score_actions(state, state.legal_actions())
        self.assertLessEqual(guide.nodes, 5)

    def test_compact_sorter_keeps_pathfinder_legal_and_budgeted(self) -> None:
        state = GameState.initial(BoardConfig(5, 8))
        model = PathagonGNN(hidden_size=8, message_layers=1)
        agent = SorterPathfinderAgent(
            model,
            depth=3,
            beam_width=6,
            max_nodes=40,
            top_k=3,
        )
        action = agent.choose_action(state, random.Random(0), set())
        self.assertIn(action, state.legal_actions())
        self.assertLessEqual(agent.nodes, 40)

    def test_compact_sorter_tactical_extension_stays_within_budget(self) -> None:
        state = GameState.initial(BoardConfig(4, 6))
        model = PathagonGNN(hidden_size=8, message_layers=1)
        agent = SorterPathfinderAgent(model, depth=2, beam_width=6, max_nodes=8, top_k=3)
        action = agent.choose_action(state, random.Random(1), set())
        self.assertIn(action, state.legal_actions())
        self.assertLessEqual(agent.nodes, 8)

    def test_compact_sorter_keeps_exact_forced_blocks_in_root_beam(self) -> None:
        config = BoardConfig(4, 5, 64)
        state = GameState(
            config,
            mask((5, 7, 9, 11, 15)),
            mask((1, 2, 3, 6, 10)),
            (0, 0),
            Player.LIGHT,
            ply=20,
        )
        model = PathagonGNN(hidden_size=8, message_layers=1)
        agent = SorterPathfinderAgent(model, depth=2, beam_width=6, max_nodes=24, top_k=3)
        root_actions = agent._root_actions(state)
        expected = set(tactical_root(state).forced_blocks)
        self.assertTrue(expected)
        self.assertTrue(expected.issubset(root_actions))


if __name__ == "__main__":
    unittest.main()
