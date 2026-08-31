#!/usr/bin/env python3
"""Filter target rows by the frozen root IDs in a calibration slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--roots", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    targets = {row["id"]: row for row in read(args.targets)}
    roots = read(args.roots)
    if len(targets) != len(read(args.targets)):
        raise SystemExit("duplicate target IDs")
    missing = [root["id"] for root in roots if root["id"] not in targets]
    if missing:
        raise SystemExit(f"missing targets for {len(missing)} roots")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(targets[root["id"]], sort_keys=True) + "\n" for root in roots))
    print(json.dumps({"targets": len(roots), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
