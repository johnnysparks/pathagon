"""Focused tests for the Rust transition contract and JEPA objective."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from jepa_afterstate import (
    ActionConditionedJEPA,
    evaluate_jepa,
    jepa_loss,
    load_rust_transitions,
)


def synthetic_row() -> dict:
    state = {
        "boardSize": 7,
        "reservePerPlayer": 14,
        "maxPlies": 40,
        "light": 0,
        "dark": 0,
        "reserve": [14, 14],
        "turn": "light",
        "forbidden": 0,
        "lastRelocatedTo": [None, None],
        "lastCapture": 0,
        "lastPlayer": None,
        "winner": None,
        "ply": 0,
    }
    next_state = dict(state)
    next_state["light"] = 1 << 24
    next_state["reserve"] = [13, 14]
    next_state["turn"] = "dark"
    next_state["lastPlayer"] = "light"
    next_state["ply"] = 1
    return {
        "schemaVersion": 1,
        "format": "pathagon-rust-jepa-afterstate-v1",
        "game": 0,
        "seed": 1,
        "state": state,
        "action": {"kind": "place", "to": 24},
        "nextState": next_state,
        "selectedForRollout": True,
    }


class JepaAfterstateTests(unittest.TestCase):
    def test_rust_row_decodes_and_mirror_audits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(json.dumps(synthetic_row()) + "\n", encoding="utf-8")
            rows = load_rust_transitions(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].action.to, 24)
        self.assertTrue(rows[0].selected_for_rollout)

    def test_jepa_loss_is_finite_and_nontrivial(self) -> None:
        predictions = torch.randn(8, 16)
        targets = torch.randn(8, 16)
        losses = jepa_loss(predictions, targets)
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertGreater(float(losses["loss"]), 0.0)

    def test_model_returns_compact_afterstate_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            payload = "\n".join(json.dumps(synthetic_row()) for _ in range(2)) + "\n"
            path.write_text(payload, encoding="utf-8")
            rows = load_rust_transitions(path)
        model = ActionConditionedJEPA(hidden_size=8, message_layers=1, embedding_size=8)
        predictions, targets, online = model(rows)
        self.assertEqual(tuple(predictions.shape), (2, 8))
        self.assertEqual(tuple(targets.shape), (2, 8))
        self.assertEqual(tuple(online.shape), (2, 8))
        metrics = evaluate_jepa(model, rows, batch_size=2)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
