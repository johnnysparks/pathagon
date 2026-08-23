#!/usr/bin/env python3
"""Convert archived schema-v2 JSONL records into the Rust compact corpus format."""

import argparse
import json
from pathlib import Path

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"


def encode_action(action):
    if action["kind"] == "place":
        code = action["to"]
    elif action["kind"] == "relocate":
        code = 49 + action["from"] * 49 + action["to"]
    else:
        raise ValueError(f"unknown action kind: {action['kind']}")
    return ALPHABET[code >> 6] + ALPHABET[code & 63]


def compact_line(record):
    winner = {None: "-", "light": "L", "dark": "D"}[record["winner"]]
    reason = {
        "path": "P",
        "threefold-repetition": "R",
        "max-plies": "M",
        "no-legal-action": "N",
    }[record["reason"]]
    actions = "".join(encode_action(move["action"]) for move in record["moves"])
    return "\t".join([
        "p1",
        encode_radix(record["seed"]),
        record["agents"]["light"],
        record["agents"]["dark"],
        winner,
        reason,
        actions,
    ])


def encode_radix(value):
    if value == 0:
        return "0"
    digits = []
    while value:
        digits.append(ALPHABET[value & 63])
        value >>= 6
    return "".join(reversed(digits))


def records(source):
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        record = value.get("record", value) if isinstance(value, dict) else value
        if isinstance(record, dict) and isinstance(record.get("moves"), list):
            yield record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    lines = sorted({compact_line(record) for record in records(args.input)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("# p1\tseed64\tlight\tdark\twinner\treason\t2-char-actions\n" + "\n".join(lines) + "\n")
    print(json.dumps({"games": len(lines), "output": str(args.output)}))


if __name__ == "__main__":
    main()
