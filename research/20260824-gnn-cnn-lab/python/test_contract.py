"""Contract v1 parity tests for the Python runtime."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from .contract import agent_manifest, agent_specification, validate_position, validate_replay_record


class ContractTest(unittest.TestCase):
    def test_shared_replay_fixture(self) -> None:
        path = Path(__file__).parents[3] / "pathagon" / "contracts" / "fixtures" / "replay-v1.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        schema = json.loads((path.parents[1] / "pathagon-contract-v1.schema.json").read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(record)
        self.assertEqual(validate_replay_record(record)["contractVersion"], 1)

    def test_position_contains_rule_relevant_state(self) -> None:
        position = {
            "contractVersion": 1,
            "config": {"rulesVersion": "pathagon-rules-v1", "boardSize": 3, "reservePerPlayer": 6, "maxPlies": 36, "repetitionLimit": 3},
            "board": ["light", None, None, None, "dark", None, None, None, None],
            "reserve": {"light": 5, "dark": 6},
            "turn": "dark",
            "forbidden": [],
            "lastRelocatedTo": {"light": None, "dark": None},
            "winner": None,
            "ply": 1,
        }
        self.assertEqual(validate_position(position)["board"].count(None), 7)

    def test_agent_manifest_carries_search_and_model_identity(self) -> None:
        manifest = agent_manifest(runtime="python", depth=3, node_budget=64, beam=12, model_hash="sha256:" + "a" * 64)
        specification = agent_specification("gnn-v1", "GNN", "1.0.0", "puct", "python-gnn", manifest=manifest)
        self.assertEqual(validate_replay_record({
            "contractVersion": 1,
            "seed": 1,
            "config": {"rulesVersion": "pathagon-rules-v1", "boardSize": 3, "reservePerPlayer": 6, "maxPlies": 36, "repetitionLimit": 3},
            "engine": {"id": "python-gnn", "runtime": "python", "version": "1.0.0", "rulesVersion": "pathagon-rules-v1"},
            "agents": {"light": "gnn-v1", "dark": "gnn-v1"},
            "agentSpecifications": {"light": specification, "dark": specification},
            "winner": None,
            "result": "draw",
            "reason": "max-plies",
            "plies": 0,
            "moves": [],
        })["agentSpecifications"]["light"]["manifest"]["nodeBudget"], 64)

    def test_optional_search_policy_is_validated(self) -> None:
        record = {
            "contractVersion": 1,
            "seed": 1,
            "config": {"rulesVersion": "pathagon-rules-v1", "boardSize": 3, "reservePerPlayer": 6, "maxPlies": 36, "repetitionLimit": 3},
            "engine": {"id": "python-gnn", "runtime": "python", "version": "1.0.0", "rulesVersion": "pathagon-rules-v1"},
            "agents": {"light": "gnn-v1", "dark": "gnn-v1"},
            "agentSpecifications": {
                "light": agent_specification("gnn-v1", "GNN", "1.0.0", "puct", "python-gnn"),
                "dark": agent_specification("gnn-v1", "GNN", "1.0.0", "puct", "python-gnn"),
            },
            "winner": None,
            "result": "draw",
            "reason": "max-plies",
            "plies": 1,
            "moves": [{
                "ply": 1,
                "player": "light",
                "action": {"kind": "place", "to": 0},
                "captured": [],
                "nodes": 4,
                "completedDepth": 0,
                "tableHits": 0,
                "policy": [0.75, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "actionValues": [0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "actionVisits": [4, 2, 0, 0, 0, 0, 0, 0, 0],
                "actionValueSource": "mcts-root-q-v1",
            }],
        }
        self.assertEqual(validate_replay_record(record)["moves"][0]["policy"][0], 0.75)


if __name__ == "__main__":
    unittest.main()
