"""Tests for the persistent historyless golden-position format."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .golden import (
    FlatGoldenTable,
    GoldenConflictError,
    GoldenTable,
    KEY_BYTES,
    LOSS,
    ROW_BYTES,
    WIN,
    canonical_position_key,
    key_bytes_for_board_size,
    pack_position_key,
    unpack_position_key,
)

from .game import BoardConfig, GameState, Player
from .symmetry import Symmetry, transform_state


class GoldenFormatTests(unittest.TestCase):
    def setUp(self) -> None:
        config = BoardConfig(size=7, reserve_per_player=14)
        self.state = GameState.seeded(
            config=config,
            light=(1 << 0) | (1 << 8),
            dark=(1 << 40) | (1 << 48),
            reserves=(12, 12),
            turn=Player.DARK,
            forbidden=1 << 24,
            last_relocated_to=(3, 45),
        )

    def test_dense_key_round_trips_rule_relevant_fields(self) -> None:
        key = pack_position_key(self.state)
        self.assertEqual(len(key), KEY_BYTES)
        decoded = unpack_position_key(key)
        self.assertEqual(decoded.light, self.state.light)
        self.assertEqual(decoded.dark, self.state.dark)
        self.assertEqual(decoded.forbidden, self.state.forbidden)
        self.assertEqual(decoded.turn, self.state.turn)
        self.assertEqual(decoded.last_relocated_to, self.state.last_relocated_to)

    def test_all_rule_preserving_symmetries_share_one_key(self) -> None:
        key = canonical_position_key(self.state)
        for symmetry in Symmetry:
            self.assertEqual(key, canonical_position_key(transform_state(self.state, symmetry)))

    def test_conflicting_exact_values_are_rejected(self) -> None:
        table = GoldenTable()
        key = table.put(self.state, LOSS)
        self.assertEqual(table.lookup(self.state), LOSS)
        with self.assertRaises(GoldenConflictError):
            table.put_key(key, WIN)

    def test_sorted_shard_can_be_looked_up_without_loading_it(self) -> None:
        other = GameState.seeded(
            config=self.state.config,
            light=1 << 5,
            dark=1 << 43,
            reserves=(13, 13),
            turn=Player.LIGHT,
        )
        table = GoldenTable()
        table.put(self.state, LOSS)
        table.put(other, WIN)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.bin"
            self.assertEqual(table.write(path), 2)
            self.assertEqual(path.stat().st_size, 2 * ROW_BYTES)
            with FlatGoldenTable(path) as shard:
                self.assertEqual(shard.lookup(self.state), LOSS)
                self.assertEqual(shard.lookup(other), WIN)
                self.assertIsNone(shard.lookup_key(b"\0" * KEY_BYTES))

    def test_five_by_five_key_round_trips_and_uses_its_own_width(self) -> None:
        config = BoardConfig(size=5, reserve_per_player=8)
        state = GameState.seeded(
            config=config,
            light=1 << 0,
            dark=1 << 24,
            reserves=(7, 7),
            turn=Player.LIGHT,
            forbidden=1 << 12,
            last_relocated_to=(3, 21),
        )
        key = pack_position_key(state)
        self.assertEqual(len(key), key_bytes_for_board_size(5))
        decoded = unpack_position_key(key, board_size=5)
        self.assertEqual(decoded.light, state.light)
        self.assertEqual(decoded.dark, state.dark)
        self.assertEqual(decoded.forbidden, state.forbidden)
        self.assertEqual(decoded.turn, state.turn)
        self.assertEqual(decoded.last_relocated_to, state.last_relocated_to)

    def test_five_by_five_shard_lookup_is_namespace_bound(self) -> None:
        config = BoardConfig(size=5, reserve_per_player=8)
        state = GameState.seeded(
            config=config,
            light=1 << 0,
            dark=1 << 24,
            reserves=(7, 7),
            turn=Player.DARK,
        )
        table = GoldenTable(board_size=5, reserve_per_player=8)
        table.put(state, WIN)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.bin"
            self.assertEqual(table.write(path), 1)
            with FlatGoldenTable(path, board_size=5, reserve_per_player=8) as shard:
                self.assertEqual(shard.lookup(state), WIN)


if __name__ == "__main__":
    unittest.main()
