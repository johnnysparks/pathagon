#!/usr/bin/env python3
"""Focused invariants for the random phase dataset generator."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("scripts") / "build-random-phase-datasets.py"
SPEC = importlib.util.spec_from_file_location("random_phase_datasets", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RandomPhaseDatasetTest(unittest.TestCase):
    def test_midgame_default_preserves_full_board_inventory(self) -> None:
        config = MODULE.make_config()
        rng = MODULE.random.Random(17)
        for turn in (MODULE.Player.LIGHT, MODULE.Player.DARK):
            state, metadata = MODULE.build_midgame(rng, config, turn)
            self.assertEqual(MODULE.count_bits(state.light), 14)
            self.assertEqual(MODULE.count_bits(state.dark), 14)
            self.assertEqual(state.reserves, (0, 0))
            self.assertIsNone(state.winner)
            self.assertEqual(metadata["requestedNoneCount"], 0)

    def test_midgame_none_count_is_rule_valid_capture_like_imbalance(self) -> None:
        config = MODULE.make_config()
        state, metadata = MODULE.build_midgame(
            MODULE.random.Random(18), config, MODULE.Player.LIGHT, none_count=14
        )
        self.assertEqual(sum(state.reserves), 14)
        self.assertEqual(MODULE.count_bits(state.light) + state.reserves[0], 14)
        self.assertEqual(MODULE.count_bits(state.dark) + state.reserves[1], 14)
        self.assertEqual(
            metadata["missingInventory"]["light"] + metadata["missingInventory"]["dark"],
            14,
        )
        self.assertIsNone(state.winner)

    def test_lategame_has_two_gaps_and_an_immediate_win(self) -> None:
        config = MODULE.make_config()
        state, metadata, winning_actions = MODULE.build_lategame(
            MODULE.random.Random(19), config, MODULE.Player.DARK
        )
        self.assertEqual(len(metadata["completedPath"]), 7)
        self.assertEqual(len(metadata["missingPath"]), 2)
        self.assertEqual(len(set(metadata["completedPath"])), 7)
        self.assertTrue(set(metadata["missingPath"]).isdisjoint(set(MODULE.bits(state.pieces(MODULE.Player.DARK)))))
        self.assertTrue(winning_actions)
        self.assertTrue(
            all(state.apply_legal(action).winner is MODULE.Player.DARK for action in winning_actions)
        )
        self.assertIsNone(state.winner)

    def test_lategame_replay_matches_existing_contract(self) -> None:
        config = MODULE.make_config()
        state, _metadata, winning_actions = MODULE.build_lategame(
            MODULE.random.Random(20), config, MODULE.Player.LIGHT
        )
        record = MODULE.late_replay_record(state, winning_actions[0], 20, "lategame-test")
        self.assertEqual(record["winner"], "light")
        self.assertEqual(record["plies"], 1)
        self.assertEqual(record["moves"][0]["action"], MODULE.action_json(winning_actions[0]))

    def test_batch_is_reproducible_and_emits_three_streams(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_summary = MODULE.build_datasets(
                output_dir=Path(first),
                mid_count=4,
                late_count=4,
                seed=21,
                mid_none_count=0,
                late_none_count=0,
            )
            second_summary = MODULE.build_datasets(
                output_dir=Path(second),
                mid_count=4,
                late_count=4,
                seed=21,
                mid_none_count=0,
                late_none_count=0,
            )
            self.assertEqual(first_summary, {**second_summary, "outputDir": first_summary["outputDir"]})
            for name in ("midgame.jsonl", "lategame.jsonl", "lategame-replays.jsonl", "report.json"):
                self.assertEqual(
                    (Path(first) / name).read_text(encoding="utf-8"),
                    (Path(second) / name).read_text(encoding="utf-8").replace(str(second), str(first)),
                )
            header = json.loads((Path(first) / "lategame-replays.jsonl").read_text().splitlines()[0])
            self.assertEqual(header["termination"], "one-move-path-win")


if __name__ == "__main__":
    unittest.main()
