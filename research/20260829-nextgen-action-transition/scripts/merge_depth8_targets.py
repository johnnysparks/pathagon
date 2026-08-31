#!/usr/bin/env python3
"""Replace the depth-7 labels for a targeted subset with deeper labels."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def read_rows(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        rows.extend(
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="depth-7 target shard glob")
    parser.add_argument("--depth8", required=True, help="depth-8 target shard glob")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    base = read_rows(args.base)
    deeper = read_rows(args.depth8)
    if not base or not deeper:
        raise SystemExit("both base and depth8 targets are required")
    base_by_id = {row["id"]: row for row in base}
    deeper_by_id = {row["id"]: row for row in deeper}
    if len(base_by_id) != len(base) or len(deeper_by_id) != len(deeper):
        raise SystemExit("duplicate target IDs")
    if not set(deeper_by_id) <= set(base_by_id):
        raise SystemExit("depth8 targets must be a subset of the base roots")
    if any(row["teacher"]["depth"] != 8 or row["teacher"]["maxNodes"] != 2_000_000 for row in deeper):
        raise SystemExit("depth8 shard has an unexpected teacher configuration")
    merged = [deeper_by_id.get(row["id"], row) for row in base]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in merged),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "base": len(base),
                "depth8Replacements": len(deeper),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
