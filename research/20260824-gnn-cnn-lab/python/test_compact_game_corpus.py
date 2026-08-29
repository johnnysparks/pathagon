"""Regression tests for the unified compact game corpus."""

from __future__ import annotations

import gzip
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.compact_game_corpus import (
    candidate_files,
    normalize_p1,
    normalize_record,
    records_from_value,
)


def record(seed: int = 7) -> dict:
    return {
        "contractVersion": 1,
        "seed": seed,
        "config": {
            "rulesVersion": "pathagon-rules-v1",
            "boardSize": 7,
            "reservePerPlayer": 14,
            "maxPlies": 120,
            "repetitionLimit": 3,
        },
        "engine": {"id": "test-engine"},
        "agents": {"light": "alpha", "dark": "beta"},
        "winner": "light",
        "reason": "path",
        "moves": [
            {"action": {"kind": "place", "to": 0}, "nodes": 10},
            {"action": {"kind": "place", "to": 48}, "policy": [1.0]},
            {
                "action": {"kind": "relocate", "from": 0, "to": 1},
                "actionValues": [0.5],
                "actionVisits": [2],
            },
        ],
    }


class CompactGameCorpusTest(unittest.TestCase):
    def test_game_identity_ignores_run_metadata(self) -> None:
        first = record(7)
        second = record(99)
        second["agents"] = {"light": "other", "dark": "agents"}
        second["winner"] = "dark"
        first_game, first_observation = normalize_record(first, Path("first.jsonl"), "source-a")
        second_game, second_observation = normalize_record(second, Path("second.jsonl"), "source-b")
        self.assertEqual(first_game.key, second_game.key)
        self.assertEqual(first_game.actions, "000m0o")
        self.assertNotEqual(first_observation.seed, second_observation.seed)
        self.assertEqual(first_observation.light_model, "-")

    def test_configuration_changes_game_identity(self) -> None:
        first = record()
        second = record()
        second["config"]["reservePerPlayer"] = 13
        first_game, _ = normalize_record(first, Path("first.jsonl"), "source")
        second_game, _ = normalize_record(second, Path("second.jsonl"), "source")
        self.assertNotEqual(first_game.key, second_game.key)

    def test_model_hash_is_preserved_as_portable_observation_metadata(self) -> None:
        value = record()
        value["agentSpecifications"] = {
            "light": {"manifest": {"modelHash": "sha256:light"}},
            "dark": {"manifest": {"checkpointHash": "sha256:dark"}},
        }
        _, observation = normalize_record(value, Path("game.jsonl"), "source")
        self.assertEqual(observation.light_model, "sha256:light")
        self.assertEqual(observation.dark_model, "sha256:dark")

    def test_p1_and_json_share_identity(self) -> None:
        value = record()
        value["moves"] = [
            {"action": {"kind": "place", "to": 0}},
            {"action": {"kind": "place", "to": 48}},
            {"action": {"kind": "relocate", "from": 0, "to": 1}},
        ]
        json_game, _ = normalize_record(value, Path("game.jsonl"), "json")
        p1_game, _ = normalize_p1("p1\t7\talpha\tbeta\tL\tP\t000m0o\n", Path("games.tsv"), "p1")
        self.assertEqual(json_game.key, p1_game.key)

    def test_nested_reports_and_string_records_are_discovered(self) -> None:
        value = {"pairings": [{"records": [record()]}, {"record": json.dumps(record(8))}]}
        self.assertEqual(len(list(records_from_value(value))), 2)

    def test_candidate_scan_includes_supported_archives_only(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.json").write_text("{}", encoding="utf-8")
            (root / "two.jsonl").write_text("{}\n", encoding="utf-8")
            with gzip.open(root / "three.jsonl.gz", "wt", encoding="utf-8") as handle:
                handle.write("{}\n")
            (root / "games.tsv").write_text("# header\n", encoding="utf-8")
            (root / "positions.tsv").write_text("ignored\n", encoding="utf-8")
            (root / "model.pt").write_bytes(b"ignored")
            names = {path.name for path in candidate_files([root], root / "output")}
        self.assertEqual(names, {"one.json", "two.jsonl", "three.jsonl.gz", "games.tsv"})


if __name__ == "__main__":
    unittest.main()
