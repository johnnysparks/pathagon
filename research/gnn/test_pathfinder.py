from __future__ import annotations

import unittest

from .game import BoardConfig, GameState
from .pathfinder import PathfinderGuide


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


if __name__ == "__main__":
    unittest.main()
