#!/usr/bin/env python3
"""Verify the promoted Ring-1 shard and compact action book deterministically."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ACTION_BOOK_MAGIC = b"PGACT01\0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(value: str) -> Path:
    return PROJECT_ROOT / value


def verify_table(path: Path, expected_rows: int, key_bytes: int) -> int:
    row_bytes = key_bytes + 1
    source = path.read_bytes()
    if len(source) != expected_rows * row_bytes:
        raise ValueError(f"{path}: expected {expected_rows * row_bytes} bytes, got {len(source)}")
    previous = None
    for offset in range(0, len(source), row_bytes):
        key = source[offset : offset + key_bytes]
        value = source[offset + key_bytes]
        if previous is not None and key <= previous:
            raise ValueError(f"{path}: keys are not strictly sorted at byte {offset}")
        if value not in {0, 1, 2}:
            raise ValueError(f"{path}: invalid WDL value {value} at byte {offset}")
        previous = key
    return expected_rows


def verify_heldout_partition(
    path: Path,
    table_path: Path,
    expected_rows: int,
    key_bytes: int,
) -> int:
    lines = [line.strip() for line in path.read_text(encoding="ascii").splitlines() if line.strip()]
    if len(lines) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} keys, got {len(lines)}")
    if lines != sorted(set(lines)):
        raise ValueError(f"{path}: keys must be unique and sorted")
    if any(len(key) != key_bytes * 2 for key in lines):
        raise ValueError(f"{path}: partition key has the wrong width")
    try:
        keys = [bytes.fromhex(key) for key in lines]
    except ValueError as error:
        raise ValueError(f"{path}: partition contains non-hex key") from error
    table_keys = {
        table_path.read_bytes()[offset : offset + key_bytes]
        for offset in range(0, table_path.stat().st_size, key_bytes + 1)
    }
    if any(key not in table_keys for key in keys):
        raise ValueError(f"{path}: partition contains a key absent from the promoted table")
    if any(hashlib.sha256(key).digest()[0] % 10 != 0 for key in keys):
        raise ValueError(f"{path}: key violates deterministic held-out policy")
    return len(keys)


def verify_action_book(path: Path, expected_rows: int, key_bytes: int, board_size: int, reserve: int) -> int:
    source = path.read_bytes()
    if len(source) < 16 or source[:8] != ACTION_BOOK_MAGIC:
        raise ValueError(f"{path}: invalid PGACT01 header")
    if source[8:12] != bytes((board_size, reserve, key_bytes, 0)):
        raise ValueError(f"{path}: namespace/header mismatch")
    rows = struct.unpack_from("<I", source, 12)[0]
    if rows != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} action rows, got {rows}")
    cells = board_size * board_size
    offset = 16
    previous = None
    for row_index in range(rows):
        if offset + key_bytes + 2 > len(source):
            raise ValueError(f"{path}: truncated row {row_index}")
        key = source[offset : offset + key_bytes]
        offset += key_bytes
        if previous is not None and key <= previous:
            raise ValueError(f"{path}: keys are not strictly sorted at row {row_index}")
        previous = key
        count = struct.unpack_from("<H", source, offset)[0]
        offset += 2
        if offset + count * 2 > len(source):
            raise ValueError(f"{path}: truncated actions at row {row_index}")
        actions = struct.unpack_from(f"<{count}H", source, offset) if count else ()
        offset += count * 2
        if len(set(actions)) != len(actions):
            raise ValueError(f"{path}: duplicate action at row {row_index}")
        if any(action >= cells + cells * cells for action in actions):
            raise ValueError(f"{path}: action outside board at row {row_index}")
    if offset != len(source):
        raise ValueError(f"{path}: trailing bytes after action rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    shard = manifest["shards"][0]
    sidecar = manifest["sidecars"][0]
    shard_path = project_path(shard["path"])
    sidecar_path = project_path(sidecar["path"])
    if shard_path.stat().st_size != shard["bytes"] or sha256(shard_path) != shard["sha256"]:
        raise ValueError("WDL shard size or SHA-256 does not match manifest")
    if sidecar_path.stat().st_size != sidecar["bytes"] or sha256(sidecar_path) != sidecar["sha256"]:
        raise ValueError("action sidecar size or SHA-256 does not match manifest")
    rows = verify_table(shard_path, shard["rows"], manifest["key"]["bytes"])
    held_out_path = project_path(manifest["source"]["heldOutPartition"])
    held_out = verify_heldout_partition(
        held_out_path,
        shard_path,
        manifest["counts"]["heldOutRows"],
        manifest["key"]["bytes"],
    )
    action_rows = verify_action_book(
        sidecar_path,
        manifest["counts"]["canonicalRows"],
        manifest["key"]["bytes"],
        shard["boardSize"],
        shard["reservePerPlayer"],
    )
    if rows != action_rows or manifest["counts"]["provenActions"] < action_rows:
        raise ValueError("table/action row counts are inconsistent")
    if held_out + manifest["counts"]["trainingRows"] != rows:
        raise ValueError("held-out and training partition counts do not cover the table")
    print(json.dumps({"status": "pass", "rows": rows, "actionRows": action_rows, "heldOutRows": held_out}, sort_keys=True))


if __name__ == "__main__":
    main()
