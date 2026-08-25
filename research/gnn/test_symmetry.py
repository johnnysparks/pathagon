"""Rules and action tests for D4 symmetry augmentation."""

from __future__ import annotations

import unittest

from .game import Action, BoardConfig, GameState, Player, has_winning_path, repetition_key
from .selfplay import SearchExample
from .symmetry import (
    ALL_SYMMETRIES,
    Symmetry,
    symmetry_swaps_players,
    transform_action,
    transform_square,
    transform_state,
)
from .train import transform_search_example


class SymmetryTest(unittest.TestCase):
    def setUp(self) -> None:
        config = BoardConfig(5, reserve_per_player=4, ply_limit=100)
        self.state = GameState(
            config=config,
            light=(1 << 0) | (1 << 6),
            dark=(1 << 1) | (1 << 7),
            reserves=(2, 3),
            turn=Player.LIGHT,
            forbidden=1 << 12,
            last_relocated_to=(6, 7),
            last_capture=1,
            last_player=Player.DARK,
            ply=17,
        )

    def test_every_transform_is_a_board_permutation(self) -> None:
        for symmetry in ALL_SYMMETRIES:
            mapped = {transform_square(self.state.config.size, square, symmetry) for square in range(25)}
            self.assertEqual(mapped, set(range(25)), symmetry)

    def test_transformed_legal_actions_match_original_actions(self) -> None:
        original_actions = set(self.state.legal_actions())
        for symmetry in ALL_SYMMETRIES:
            transformed = transform_state(self.state, symmetry)
            transformed_actions = {
                transform_action(action, self.state.config, symmetry) for action in original_actions
            }
            self.assertEqual(set(transformed.legal_actions()), transformed_actions, symmetry)

    def test_transforming_before_or_after_a_move_is_equivalent(self) -> None:
        for symmetry in ALL_SYMMETRIES:
            transformed = transform_state(self.state, symmetry)
            for action in self.state.legal_actions():
                transformed_action = transform_action(action, self.state.config, symmetry)
                moved_then_transformed = transform_state(self.state.apply_legal(action), symmetry)
                transformed_then_moved = transformed.apply_legal(transformed_action)
                self.assertEqual(moved_then_transformed, transformed_then_moved, symmetry)

    def test_player_swapping_transforms_preserve_player_relative_value(self) -> None:
        winning_state = GameState(
            config=self.state.config,
            light=sum(1 << (row * 5) for row in range(5)),
            dark=1 << 24,
            reserves=(0, 0),
            turn=Player.DARK,
            winner=Player.LIGHT,
            ply=5,
        )
        self.assertTrue(has_winning_path(winning_state, Player.LIGHT))
        for symmetry in ALL_SYMMETRIES:
            transformed = transform_state(winning_state, symmetry)
            expected_winner = Player.LIGHT.other() if symmetry_swaps_players(symmetry) else Player.LIGHT
            self.assertEqual(transformed.winner, expected_winner, symmetry)
            self.assertTrue(has_winning_path(transformed, expected_winner), symmetry)

    def test_repetition_identity_is_preserved(self) -> None:
        for symmetry in ALL_SYMMETRIES:
            transformed = transform_state(self.state, symmetry)
            for action in self.state.legal_actions():
                original_next = self.state.apply_legal(action)
                transformed_next = transformed.apply_legal(transform_action(action, self.state.config, symmetry))
                self.assertEqual(
                    repetition_key(transform_state(original_next, symmetry)),
                    repetition_key(transformed_next),
                    symmetry,
                )

    def test_search_targets_keep_policy_alignment_after_transform(self) -> None:
        actions = tuple(self.state.legal_actions())
        example = SearchExample(
            state=self.state,
            actions=actions,
            policy=tuple(index + 1 for index in range(len(actions))),
            selected_action=actions[0],
            value=-1.0,
        )
        for symmetry in ALL_SYMMETRIES:
            transformed = transform_search_example(example, symmetry)
            self.assertEqual(transformed.policy, example.policy, symmetry)
            self.assertEqual(set(transformed.actions), set(transformed.state.legal_actions()), symmetry)
            self.assertEqual(
                transformed.selected_action,
                transform_action(example.selected_action, self.state.config, symmetry),
                symmetry,
            )


if __name__ == "__main__":
    unittest.main()
