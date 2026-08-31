#!/usr/bin/env python3
"""Validate and promote Rust Ring-1 candidates into a golden shard."""

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
from python.golden import FlatGoldenTable, GoldenTable, WIN, pack_position_key, rows_sha256  # noqa: E402
from python.symmetry import ALL_SYMMETRIES, transform_action, transform_state  # noqa: E402


ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
BOARD_SIZE = 7
RESERVE = 14
ACTION_BOOK_MAGIC = b"PGACT01\0"


def decode_action(token: str) -> Action:
    if len(token) != 2:
        raise ValueError("action token must contain exactly two characters")
    try:
        code = (ALPHABET.index(token[0]) << 6) | ALPHABET.index(token[1])
    except ValueError as error:
        raise ValueError(f"invalid action token {token!r}") from error
    cells = BOARD_SIZE * BOARD_SIZE
    if code < cells:
        return Action.place(code)
    relocation = code - cells
    from_square, to = divmod(relocation, cells)
    if from_square >= cells or to >= cells:
        raise ValueError(f"action token {token!r} is outside the board")
    return Action.relocate(from_square, to)


def encode_action(action: Action) -> str:
    cells = BOARD_SIZE * BOARD_SIZE
    code = action.to if action.kind == 0 else cells + action.from_square * cells + action.to
    return ALPHABET[code >> 6] + ALPHABET[code & 63]


def state_from_json(raw: dict[str, Any]) -> GameState:
    config = BoardConfig(
        size=int(raw["boardSize"]),
        reserve_per_player=int(raw["reservePerPlayer"]),
    )
    turn = Player.LIGHT if raw["turn"] == "light" else Player.DARK
    markers = tuple(None if marker is None else int(marker) for marker in raw["lastRelocatedTo"])
    return GameState.seeded(
        config=config,
        light=int(raw["light"]),
        dark=int(raw["dark"]),
        reserves=tuple(int(value) for value in raw["reserve"]),
        turn=turn,
        forbidden=int(raw["forbidden"]),
        last_relocated_to=markers,
        ply=int(raw["ply"]),
    )


def canonical_action(state: GameState, action: Action) -> tuple[bytes, Action]:
    choices = []
    for symmetry in ALL_SYMMETRIES:
        transformed = transform_state(state, symmetry)
        choices.append((pack_position_key(transformed), transform_action(action, state.config, symmetry)))
    return min(choices, key=lambda choice: choice[0])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_code(action: Action) -> int:
    cells = BOARD_SIZE * BOARD_SIZE
    return action.to if action.kind == 0 else cells + action.from_square * cells + action.to


def write_action_book(path: Path, rows: OrderedDict[bytes, dict[str, Any]]) -> None:
    """Write compact sorted key -> proven-action labels for runtime lookup."""

    with path.open("wb") as output:
        output.write(ACTION_BOOK_MAGIC)
        output.write(bytes((BOARD_SIZE, RESERVE, 14, 0)))
        output.write(struct.pack("<I", len(rows)))
        for key in sorted(rows):
            actions = rows[key]["provenActions"]
            output.write(key)
            output.write(struct.pack("<H", len(actions)))
            for action in actions:
                output.write(struct.pack("<H", action_code(decode_action(action["token"]))))


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--held-out",
        type=Path,
        default=PROJECT_ROOT / "data/golden/partitions/fresh-frontier-wdl-v1/ring-01-heldout.txt",
    )
    parser.add_argument(
        "--existing-table",
        type=Path,
        default=PROJECT_ROOT / "data/golden/tables/historyless-wdl-v1/7x7-r14/shard-00.bin",
    )
    args = parser.parse_args()

    table = GoldenTable(board_size=BOARD_SIZE, reserve_per_player=RESERVE)
    existing_table = FlatGoldenTable(
        args.existing_table,
        board_size=BOARD_SIZE,
        reserve_per_player=RESERVE,
    )
    rows: OrderedDict[bytes, dict[str, Any]] = OrderedDict()
    input_records = 0
    raw_positions = 0
    proven_actions = 0
    existing_overlaps = 0

    try:
        with args.input.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                input_records += 1
                record = json.loads(line)
                if record.get("schemaVersion") != 1 or record.get("ring") != 1:
                    raise ValueError(f"{args.input}:{line_number}: unsupported frontier record")
                state = state_from_json(record["position"])
                if state.config.size != BOARD_SIZE or state.config.reserve_per_player != RESERVE:
                    raise ValueError(f"{args.input}:{line_number}: Ring 1 promotion requires 7x7/14")
                if state.winner is not None or not state.legal_actions():
                    raise ValueError(f"{args.input}:{line_number}: candidate parent is not playable")
                raw_positions += 1
                old_value = existing_table.lookup(state)
                if old_value is not None:
                    existing_overlaps += 1
                    if old_value != WIN:
                        raise ValueError(
                            f"{args.input}:{line_number}: candidate contradicts existing golden value {old_value}"
                        )
                canonical_key, _ = canonical_action(state, decode_action(record["actions"][0]["token"]))
                existing = rows.setdefault(
                    canonical_key,
                    {
                        "schemaVersion": 1,
                        "tableFamily": "fresh-frontier-wdl-v1",
                        "ring": 1,
                        "key": canonical_key.hex(),
                        "outcome": "win",
                        "distance": 1,
                        "optimalActionsKnown": False,
                        "legalActionCount": int(record["legalActionCount"]),
                        "provenActions": {},
                        "witnesses": [],
                        "proof": record.get("proof", {}),
                    },
                )
                table.put(state, WIN)
                for action_record in record["actions"]:
                    action = decode_action(action_record["token"])
                    if action not in state.legal_actions():
                        raise ValueError(f"{args.input}:{line_number}: action is not legal in parent")
                    child = state.apply_legal(action)
                    if child.winner != state.turn:
                        raise ValueError(f"{args.input}:{line_number}: witness action does not win")
                    action_key, canonical = canonical_action(state, action)
                    token = encode_action(canonical)
                    existing["provenActions"].setdefault(
                        token,
                        {"token": token, "outcome": "win", "distance": 1, "known": True, "witnessCount": 0},
                    )["witnessCount"] += int(action_record.get("witnessCount", 1))
                    proven_actions += 1
                for witness in record.get("witnesses", []):
                    existing["witnesses"].append(witness)
    finally:
        existing_table.close()

    args.table.parent.mkdir(parents=True, exist_ok=True)
    table_rows = table.write(args.table)
    args.sidecar.parent.mkdir(parents=True, exist_ok=True)
    for row in rows.values():
        row["provenActions"] = sorted(row["provenActions"].values(), key=lambda action: action["token"])
        row["witnesses"] = sorted(row["witnesses"], key=lambda witness: (witness.get("gameKey", ""), witness.get("action", "")))
    write_action_book(args.sidecar, rows)
    held_out = sorted(
        key.hex()
        for key in rows
        if hashlib.sha256(key).digest()[0] % 10 == 0
    )
    args.held_out.parent.mkdir(parents=True, exist_ok=True)
    args.held_out.write_text("".join(f"{key}\n" for key in held_out), encoding="ascii")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schemaVersion": 1,
        "tableFamily": "fresh-frontier-wdl-v1",
        "rulesVersion": "pathagon-rules-v1",
        "semantics": {
            "outcomes": {"loss": 0, "draw": 1, "win": 2, "unknown": "absent"},
            "history": "fresh-root-with-repetition-aware-proof",
            "horizon": "ring-1-terminal-witness",
            "optimalActions": "partial-proven-actions-until-complete",
        },
        "key": {
            "encoding": "packed-2bit-cells-turn-relocation-v1",
            "canonicalization": "d4-eight-rules-preserving-symmetries",
            "bytes": 14,
        },
        "shards": [{
            "path": project_relative(args.table),
            "boardSize": BOARD_SIZE,
            "reservePerPlayer": RESERVE,
            "rows": table_rows,
            "bytes": args.table.stat().st_size,
            "sha256": rows_sha256(args.table),
        }],
        "sidecars": [{
            "path": project_relative(args.sidecar),
            "bytes": args.sidecar.stat().st_size,
            "sha256": sha256(args.sidecar),
        }],
        "counts": {
            "inputRecords": input_records,
            "rawPositions": raw_positions,
            "canonicalRows": table_rows,
            "provenActions": proven_actions,
            "completeOptimalActionSets": 0,
            "existingGoldenOverlaps": existing_overlaps,
            "heldOutRows": len(held_out),
            "trainingRows": table_rows - len(held_out),
        },
        "source": {
            "kind": "replayed-corpus-penultimate-positions",
            "input": project_relative(args.input),
            "existingTable": project_relative(args.existing_table),
            "heldOutPartition": project_relative(args.held_out),
            "heldOutPolicy": "sha256(canonical-key)[0] modulo 10 equals 0",
            "proof": "each action was re-applied and had to produce the parent player's path terminal",
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, separators=(",", ":")))


if __name__ == "__main__":
    main()
