import random
import unittest

from .game import BoardConfig, GameState
from .league import HeuristicAgent, update_elo


class LeagueTest(unittest.TestCase):
    def test_elo_draw_keeps_ratings_equal(self) -> None:
        ratings = {"light": 1_000.0, "dark": 1_000.0}
        update_elo(ratings, {"agents": {"light": "light", "dark": "dark"}, "winner": None})
        self.assertEqual(ratings["light"], 1_000.0)
        self.assertEqual(ratings["dark"], 1_000.0)

    def test_elo_win_moves_ratings(self) -> None:
        ratings = {"light": 1_000.0, "dark": 1_000.0}
        update_elo(ratings, {"agents": {"light": "light", "dark": "dark"}, "winner": "light"})
        self.assertGreater(ratings["light"], 1_000.0)
        self.assertLess(ratings["dark"], 1_000.0)

    def test_heuristic_returns_legal_action(self) -> None:
        state = GameState.initial(BoardConfig(5, 8))
        action = HeuristicAgent(depth=1, beam_width=4, max_nodes=20).choose_action(state, random.Random(0), set())
        self.assertIn(action, list(state.legal_actions()))


if __name__ == "__main__":
    unittest.main()
