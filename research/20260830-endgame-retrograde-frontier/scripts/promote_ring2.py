#!/usr/bin/env python3
"""Independently cross-check Rust replay-ring tablebase promotion.

This script is intentionally conservative.  An exported Ring-2 node is not
gold merely because it has a replay witness: every legal child must have an
exact value in the solved inner tablebase.  A large incomplete graph therefore
produces a report with zero promotions instead of silently turning unknowns
into losses or draws.

The authoritative promotion path is the Rust
``pathagon-endgame-promote`` executable. This Python implementation remains
for independent agreement checks and historical reproducibility only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LAB_ROOT = PROJECT_ROOT / "research/20260824-gnn-cnn-lab"
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from python.game import Action, BoardConfig, GameState, Player  # noqa: E402
from python.golden import FlatGoldenTable, GoldenTable, rows_sha256  # noqa: E402
from python.symmetry import ALL_SYMMETRIES, transform_state  # noqa: E402


ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
ACTION_BOOK_MAGIC = b"PGACT02\0"
ACTION_BOOK_NONE_DISTANCE = 0xFFFF
COMPACT_VALUE_MAGIC = b"PGTBV01\0"
COMPACT_VALUE_HEADER_BYTES = 20
COMPACT_NONE_DISTANCE = 65535
BOARD_SIZE = 7
RESERVE = 14


def decode_action(token: str) -> Action:
    if len(token) != 2:
        raise ValueError(f"invalid action token {token!r}")
    code = (ALPHABET.index(token[0]) << 6) | ALPHABET.index(token[1])
    cells = BOARD_SIZE * BOARD_SIZE
    if code < cells:
        return Action.place(code)
    return Action.relocate(*divmod(code - cells, cells))


def action_code(action: Action) -> int:
    cells = BOARD_SIZE * BOARD_SIZE
    return action.to if action.kind == 0 else cells + action.from_square * cells + action.to


def encode_action(action: Action) -> str:
    code = action_code(action)
    return ALPHABET[code >> 6] + ALPHABET[code & 63]


def state_from_json(raw: dict[str, Any]) -> GameState:
    config = BoardConfig(size=int(raw["boardSize"]), reserve_per_player=int(raw["reservePerPlayer"]))
    return GameState.seeded(
        config=config,
        light=int(raw["light"]),
        dark=int(raw["dark"]),
        reserves=tuple(int(value) for value in raw["reserve"]),
        turn=Player.LIGHT if raw["turn"] == "light" else Player.DARK,
        forbidden=int(raw.get("forbidden", 0)),
        last_relocated_to=tuple(
            None if value is None else int(value) for value in raw.get("lastRelocatedTo", [None, None])
        ),
        ply=int(raw.get("ply", 0)),
    )


def canonical_key_hex(state: GameState) -> str:
    from python.golden import canonical_position_key

    return canonical_position_key(state).hex()


def load_shard_values(directory: Path) -> tuple[dict[str, dict[str, Any]], int]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    shard_count = int(manifest["shardCount"])
    values: dict[str, dict[str, Any]] = {}
    shard_paths = manifest.get("shards") or [f"shard-{index:05}.json" for index in range(shard_count)]
    if len(shard_paths) != shard_count:
        raise ValueError("shard manifest path count does not match shardCount")
    for index in range(shard_count):
        path = directory / str(shard_paths[index])
        if path.suffix == ".bin":
            shard = read_compact_values(path)
        else:
            shard = json.loads(path.read_text(encoding="utf-8"))
        for key, value in shard.items():
            if value.get("outcome") not in {"loss", "draw", "win"}:
                raise ValueError(f"inner value for {key} is not exact W/D/L")
            if key in values and values[key] != value:
                raise ValueError(f"contradictory inner value for {key}")
            values[key] = value
    return values, shard_count


def read_compact_values(path: Path) -> dict[str, dict[str, Any]]:
    source = path.read_bytes()
    if len(source) < COMPACT_VALUE_HEADER_BYTES or source[:8] != COMPACT_VALUE_MAGIC:
        raise ValueError(f"{path}: invalid compact value header")
    key_bytes = source[8]
    if source[9:12] != b"\0\0\0":
        raise ValueError(f"{path}: unsupported compact value flags")
    rows = struct.unpack_from("<Q", source, 12)[0]
    row_bytes = key_bytes + 3
    expected = COMPACT_VALUE_HEADER_BYTES + rows * row_bytes
    if len(source) != expected:
        raise ValueError(f"{path}: compact value size does not match header")
    values: dict[str, dict[str, Any]] = {}
    previous = None
    offset = COMPACT_VALUE_HEADER_BYTES
    for _ in range(rows):
        key_bytes_value = source[offset : offset + key_bytes]
        offset += key_bytes
        if previous is not None and key_bytes_value <= previous:
            raise ValueError(f"{path}: compact value keys are not sorted")
        outcome = source[offset]
        distance = struct.unpack_from("<H", source, offset + 1)[0]
        offset += 3
        if outcome not in {0, 1, 2}:
            raise ValueError(f"{path}: compact value has an invalid outcome")
        values[key_bytes_value.hex()] = {
            "outcome": ("loss", "draw", "win")[outcome],
            "distance": None if distance == COMPACT_NONE_DISTANCE else distance,
        }
        previous = key_bytes_value
    return values


def write_action_book(path: Path, rows: OrderedDict[str, dict[str, Any]]) -> None:
    with path.open("wb") as output:
        output.write(ACTION_BOOK_MAGIC)
        output.write(bytes((BOARD_SIZE, RESERVE, 14, 0)))
        output.write(struct.pack("<I", len(rows)))
        for key in sorted(rows):
            row = rows[key]
            actions = sorted(
                row["provenActions"],
                key=lambda action: action_code(decode_action(action["token"])),
            )
            output.write(bytes.fromhex(key))
            root_outcome = {"loss": 0, "draw": 1, "win": 2}.get(row.get("outcome"))
            if root_outcome is None:
                raise ValueError(f"unsupported root outcome for {key}")
            output.write(bytes((int(bool(row.get("optimalActionsKnown", False))), root_outcome)))
            distance = row.get("distance")
            output.write(struct.pack("<H", ACTION_BOOK_NONE_DISTANCE if distance is None else int(distance)))
            output.write(struct.pack("<H", len(actions)))
            for action in actions:
                output.write(struct.pack("<H", action_code(decode_action(action["token"]))))
                outcome = {"loss": 0, "draw": 1, "win": 2}.get(action.get("outcome"))
                output.write(bytes((3 if outcome is None else outcome,)))
                action_distance = action.get("distance")
                output.write(struct.pack(
                    "<H",
                    ACTION_BOOK_NONE_DISTANCE if action_distance is None else int(action_distance),
                ))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--existing-table", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--ring", type=int, default=2)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.ring < 2:
        raise ValueError("--ring must be at least 2")

    values, shard_count = load_shard_values(args.shards)
    existing = FlatGoldenTable(args.existing_table, board_size=BOARD_SIZE, reserve_per_player=RESERVE)
    promoted: OrderedDict[bytes, int] = OrderedDict()
    actions: OrderedDict[str, dict[str, Any]] = OrderedDict()
    stats = {
        "graphRecords": 0,
        "ringRecords": 0,
        "exactValueRecords": 0,
        "closedRingRows": 0,
        "promotedRows": 0,
        "unknownRingRows": 0,
        "contradictions": 0,
        "invalidRecords": 0,
        "symmetryChecks": 0,
        "seededInnerRows": len(values),
        "valueShards": shard_count,
    }
    sample_by_shard: dict[int, dict[str, Any]] = {}

    try:
        with args.graph.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                stats["graphRecords"] += 1
                record = json.loads(line)
                if record.get("ring") != args.ring:
                    continue
                stats["ringRecords"] += 1
                key = str(record["key"])
                value = values.get(key)
                if value is None:
                    stats["unknownRingRows"] += 1
                    continue
                stats["exactValueRecords"] += 1
                if record.get("complete") is not True or record.get("proof", {}).get("lineage") != "full-corpus-replay-plus-verified-terminal-suffix":
                    raise ValueError(
                        f"{args.graph}:{line_number}: exact Ring-{args.ring} row lacks complete replay proof"
                    )
                state = state_from_json(record["position"])
                if canonical_key_hex(state) != key:
                    raise ValueError(f"{args.graph}:{line_number}: position key is not canonical")
                legal = state.legal_actions()
                edge_rows = record.get("actions", [])
                edge_actions = {decode_action(edge["action"]): edge["child"] for edge in edge_rows}
                if len(edge_actions) != len(edge_rows):
                    raise ValueError(f"{args.graph}:{line_number}: duplicate action edge")
                if set(edge_actions) != set(legal) or set(edge_actions.values()) != set(record.get("children", [])):
                    raise ValueError(f"{args.graph}:{line_number}: edge graph does not cover the legal action set")
                for action, child_key in edge_actions.items():
                    if canonical_key_hex(state.apply_legal(action)) != child_key:
                        raise ValueError(f"{args.graph}:{line_number}: child key does not replay from the parent")
                # A closed value is promotable only when every child has a
                # known inner value, not merely when this parent has a seed.
                child_values = [values.get(child) for child in edge_actions.values()]
                if any(child is None for child in child_values):
                    stats["unknownRingRows"] += 1
                    continue
                stats["closedRingRows"] += 1
                outcome = {"loss": 0, "draw": 1, "win": 2}[value["outcome"]]
                action_outcomes = [
                    {"loss": "win", "draw": "draw", "win": "loss"}[child["outcome"]]
                    for child in child_values
                ]
                expected_outcome = (
                    "win"
                    if "win" in action_outcomes
                    else "loss"
                    if all(item == "loss" for item in action_outcomes)
                    else "draw"
                )
                if expected_outcome == "win":
                    winning_distances = [
                        int(child["distance"]) + 1
                        for child in child_values
                        if child["outcome"] == "loss" and child.get("distance") is not None
                    ]
                    if not winning_distances:
                        raise ValueError(
                            f"{args.graph}:{line_number}: winning child lacks a proven distance"
                        )
                    expected_distance = min(winning_distances)
                elif expected_outcome == "loss":
                    losing_distances = [
                        int(child["distance"]) + 1
                        for child in child_values
                        if child["outcome"] == "win" and child.get("distance") is not None
                    ]
                    if not losing_distances:
                        raise ValueError(
                            f"{args.graph}:{line_number}: losing child lacks a proven distance"
                        )
                    expected_distance = max(losing_distances)
                else:
                    expected_distance = None
                if value["outcome"] != expected_outcome or value.get("distance") != expected_distance:
                    raise ValueError(
                        f"{args.graph}:{line_number}: solved Ring-{args.ring} value disagrees with child minimax"
                    )
                old = existing.lookup(state)
                if old is not None and old != outcome:
                    stats["contradictions"] += 1
                    raise ValueError(f"{args.graph}:{line_number}: conflicts with existing golden value")
                canonical = bytes.fromhex(key)
                previous = promoted.get(canonical)
                if previous is not None and previous != outcome:
                    stats["contradictions"] += 1
                    raise ValueError(f"{args.graph}:{line_number}: contradictory promoted value")
                promoted[canonical] = outcome
                action_values = []
                for action, child_key in edge_actions.items():
                    child = values[child_key]
                    child_outcome = {"loss": "win", "draw": "draw", "win": "loss"}[child["outcome"]]
                    action_values.append(
                        {
                            "token": encode_action(action),
                            "outcome": child_outcome,
                            "distance": (
                                None
                                if child_outcome == "draw"
                                else int(child["distance"]) + 1
                            ),
                        }
                    )
                actions[key] = {
                    "optimalActionsKnown": True,
                    "outcome": value["outcome"],
                    "distance": value.get("distance"),
                    "provenActions": action_values,
                }
                for symmetry in ALL_SYMMETRIES:
                    transformed = transform_state(state, symmetry)
                    if canonical_key_hex(transformed) != key:
                        raise ValueError(f"{args.graph}:{line_number}: symmetry canonicalization changed")
                    stats["symmetryChecks"] += 1
    finally:
        existing.close()

    if promoted and not (args.table and args.sidecar and args.manifest):
        raise ValueError("--table, --sidecar, and --manifest are required when rows are promotable")
    if promoted:
        table = GoldenTable(board_size=BOARD_SIZE, reserve_per_player=RESERVE)
        for key, outcome in promoted.items():
            table.put_key(key, outcome)
        args.table.parent.mkdir(parents=True, exist_ok=True)
        rows = table.write(args.table)
        args.sidecar.parent.mkdir(parents=True, exist_ok=True)
        write_action_book(args.sidecar, actions)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "tableFamily": "fresh-frontier-wdl-v2",
                    "rulesVersion": "pathagon-rules-v1",
                    "ring": args.ring,
                    "provenance": {
                        "solverVersion": "pathagon-endgame-tablebase-v1",
                        "rulesVersion": "pathagon-rules-v1",
                        "proofLineage": "complete-forward-legal-edges-plus-exact-inner-seeds",
                    },
                    "rows": rows,
                    "shard": {"path": str(args.table), "sha256": rows_sha256(args.table)},
                    "sidecar": {"path": str(args.sidecar), "sha256": sha256(args.sidecar)},
                    "source": str(args.graph),
                    "promotion": f"closed-ring-{args.ring}-only",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        stats["promotedRows"] = rows

    report = {
        "schemaVersion": 1,
        "experiment": f"ring-{args.ring}-golden-promotion",
        "graph": str(args.graph),
        "innerShards": str(args.shards),
        "stats": stats,
        "gates": {
            "inventoryAndSeededValidation": "pass",
            "forwardTransitionWitness": "pass",
            "symmetryInvariant": "pass" if stats["symmetryChecks"] else "not-run",
            "contradictoryExistingGold": "pass" if stats["contradictions"] == 0 else "fail",
            "completeActionSets": "pass" if stats["closedRingRows"] else "not-run",
            "promotionDecision": "promote" if stats["promotedRows"] else "retain-unknown-and-do-not-promote",
        },
        "shardSamples": sample_by_shard,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
