#!/usr/bin/env python3
"""Build a root-aware content-addressed sidecar for seeded game records."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
from typing import Any

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"


def action_code(action: dict[str, Any], size: int) -> int:
    kind = action.get("kind")
    if kind == "place":
        return int(action["to"])
    if kind == "relocate":
        return 49 + int(action["from"]) * 49 + int(action["to"])
    raise ValueError(f"unsupported action kind: {kind!r}")


def encode_actions(moves: list[dict[str, Any]], size: int) -> str:
    result = []
    for move in moves:
        code = action_code(move["action"], size)
        if not 0 <= code < 4096:
            raise ValueError("action exceeds compact 12-bit encoding")
        result.append(ALPHABET[code >> 6] + ALPHABET[code & 63])
    return "".join(result)


def record_iter(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        record = value.get("record", value) if isinstance(value, dict) else value
        if isinstance(record, dict) and isinstance(record.get("moves"), list):
            yield record


def key_for(root: dict[str, Any], actions: str) -> str:
    payload = "seeded-v1\0" + json.dumps(root, sort_keys=True, separators=(",", ":")) + "\0" + actions
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload.encode()).digest()).decode().rstrip("=")
    return "sg1_" + digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    games: dict[str, dict[str, Any]] = {}
    for path in args.input:
        for record in record_iter(path):
            root = record.get("initialPosition")
            if not isinstance(root, dict):
                continue
            config = record.get("config", {})
            size = int(config.get("boardSize", 7))
            actions = encode_actions(record["moves"], size)
            key = key_for(root, actions)
            observation = {
                "source": str(path),
                "seed": record.get("seed"),
                "winner": record.get("winner"),
                "reason": record.get("reason"),
                "rootClass": (record.get("provenance") or {}).get("rootClass"),
            }
            entry = games.setdefault(key, {
                "key": key,
                "rulesVersion": config.get("rulesVersion", "pathagon-rules-v1"),
                "boardSize": size,
                "reservePerPlayer": int(config.get("reservePerPlayer", 14)),
                "repetitionLimit": int(config.get("repetitionLimit", 3)),
                "plies": len(record["moves"]),
                "initialPosition": root,
                "actions": actions,
                "observations": [],
            })
            entry["observations"].append(observation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for key in sorted(games):
            value = games[key]
            value["observations"].sort(key=lambda item: (str(item["source"]), item["seed"] if item["seed"] is not None else -1))
            handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(args.output), "games": len(games), "observations": sum(len(value["observations"]) for value in games.values())}, sort_keys=True))


if __name__ == "__main__":
    main()
