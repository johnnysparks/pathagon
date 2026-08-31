#!/usr/bin/env python3
"""Select an exactly color-balanced heldout depth-8 ablation set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def turn(root: dict) -> str:
    return root["state"].split(".")[4]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--existing", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-turn", type=int, default=128)
    args = parser.parse_args()

    base = rows(args.base)
    existing = rows(args.existing)
    existing_ids = {root["id"] for root in existing}
    if len(existing_ids) != len(existing):
        raise SystemExit("existing roots contain duplicate IDs")
    kept = {
        "L": [root for root in existing if turn(root) == "L"][: args.per_turn],
        "D": [root for root in existing if turn(root) == "D"][: args.per_turn],
    }
    if len(kept["L"]) > args.per_turn or len(kept["D"]) > args.per_turn:
        raise SystemExit("too many existing roots for requested balance")
    for color in ("L", "D"):
        if len(kept[color]) < args.per_turn:
            needed = args.per_turn - len(kept[color])
            additions = [
                root
                for root in base
                if root["partition"] == "heldout"
                and turn(root) == color
                and root["id"] not in existing_ids
            ][:needed]
            if len(additions) != needed:
                raise SystemExit(f"not enough heldout {color} roots")
            kept[color].extend(additions)
    selected = {root["id"] for color in kept for root in kept[color]}
    ordered = [root for root in base if root["id"] in selected]
    if len(ordered) != args.per_turn * 2:
        raise SystemExit("balanced selection has the wrong size")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(root, sort_keys=True) + "\n" for root in ordered),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "roots": len(ordered),
                "turns": {color: len(kept[color]) for color in ("L", "D")},
                "newRoots": len(selected - existing_ids),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
