"""Check the variable-size adapter against the shared 7x7 rule fixtures."""

from __future__ import annotations

import csv
import unittest
from pathlib import Path

from .game import Action, BoardConfig, GameState, Player


def square_token(text: str):
    return None if text == "-" else int(text)


def fixture_action(text: str) -> Action:
    if text.startswith("P"):
        return Action.place(int(text[1:]))
    source, destination = text[1:].split(">")
    return Action.relocate(int(source), int(destination))


class ParityFixtureTest(unittest.TestCase):
    def test_shared_fixtures(self) -> None:
        path = Path(__file__).parents[3] / "data" / "fixtures" / "rules-parity.tsv"
        with path.open(encoding="utf-8", newline="") as handle:
            rows = csv.reader((line for line in handle if not line.startswith("#")), delimiter="\t")
            for row in rows:
                name, placements, turn, light_reserve, dark_reserve, forbidden, last_light, last_dark, action_text, legal, winner, captured = row
                light = 0
                dark = 0
                if placements != "-":
                    for placement in placements.split(","):
                        square = int(placement[:-1])
                        if placement[-1] == "L":
                            light |= 1 << square
                        else:
                            dark |= 1 << square
                state = GameState(
                    BoardConfig(7, 14),
                    light,
                    dark,
                    (int(light_reserve), int(dark_reserve)),
                    Player.LIGHT if turn == "light" else Player.DARK,
                    0 if forbidden == "-" else 1 << int(forbidden),
                    (square_token(last_light), square_token(last_dark)),
                )
                action = fixture_action(action_text)
                is_legal = action in state.legal_actions()
                self.assertEqual(is_legal, legal == "true", name)
                if not is_legal:
                    continue
                next_state = state.apply_legal(action)
                expected_winner = None if winner == "-" else (Player.LIGHT if winner == "light" else Player.DARK)
                expected_captured = 0 if captured == "-" else sum(1 << int(square) for square in captured.split(","))
                self.assertEqual(next_state.winner, expected_winner, name)
                self.assertEqual(next_state.forbidden, expected_captured, name)


if __name__ == "__main__":
    unittest.main()
