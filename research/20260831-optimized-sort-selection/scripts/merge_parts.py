#!/usr/bin/env python3
"""Merge deterministic Rust target shards and reject duplicate root IDs."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts", required=True, help="glob for target JSONL parts")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    paths = sorted(
        glob.glob(args.parts),
        key=lambda path: int(Path(path).stem.rsplit("-", 1)[1]),
    )
    if not paths:
        raise SystemExit("no target parts matched")
    rows = [
        json.loads(line)
        for path in paths
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("target parts contain duplicate root IDs")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"parts": paths, "rows": len(rows), "out": str(args.out)}))


if __name__ == "__main__":
    main()
