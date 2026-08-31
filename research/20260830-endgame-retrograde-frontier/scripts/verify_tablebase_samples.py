#!/usr/bin/env python3
"""Re-solve one deterministic exact row from every non-empty value shard.

The current Ring-2 frontier has exact inner-ring seed stubs and unknown
parents. A sampled seed stub is therefore a closed one-node graph, which
lets this check exercise the Rust tablebase executable independently of the
full solve without retaining a second copy of the multi-gigabyte frontier.
For later rings, sampled rows must either be terminal/seed rows or have all
of their child records included in the sampled closure.
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any


COMPACT_VALUE_MAGIC = b"PGTBV01\0"
COMPACT_VALUE_HEADER_BYTES = 20
COMPACT_NONE_DISTANCE = 65535


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
        key = source[offset : offset + key_bytes]
        offset += key_bytes
        if previous is not None and key <= previous:
            raise ValueError(f"{path}: compact value keys are not sorted")
        outcome = source[offset]
        distance = struct.unpack_from("<H", source, offset + 1)[0]
        offset += 3
        if outcome not in {0, 1, 2}:
            raise ValueError(f"{path}: compact value has an invalid outcome")
        values[key.hex()] = {
            "outcome": ("loss", "draw", "win")[outcome],
            "distance": None if distance == COMPACT_NONE_DISTANCE else distance,
        }
        previous = key
    return values


def load_samples(shards: Path, per_shard: int) -> dict[str, dict[str, Any]]:
    manifest = json.loads((shards / "manifest.json").read_text(encoding="utf-8"))
    selected: dict[str, dict[str, Any]] = {}
    shard_paths = manifest.get("shards") or [
        f"shard-{index:05}.json" for index in range(int(manifest["shardCount"]))
    ]
    if len(shard_paths) != int(manifest["shardCount"]):
        raise ValueError("shard manifest path count does not match shardCount")
    for index in range(int(manifest["shardCount"])):
        path = shards / str(shard_paths[index])
        shard = read_compact_values(path) if path.suffix == ".bin" else json.loads(path.read_text(encoding="utf-8"))
        for key in sorted(shard)[:per_shard]:
            selected[key] = {"expected": shard[key], "shard": index}
    if not selected:
        raise ValueError("value shards contain no rows")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-shard", type=int, default=1)
    args = parser.parse_args()
    if args.per_shard <= 0:
        raise ValueError("--per-shard must be positive")

    samples = load_samples(args.shards, args.per_shard)
    records: dict[str, dict[str, Any]] = {}
    with args.graph.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            key = str(record.get("key", ""))
            if key in samples:
                records[key] = record
    missing = sorted(set(samples) - set(records))
    if missing:
        raise ValueError(f"sample keys missing from graph: {missing[:5]}")

    for key, record in records.items():
        if record.get("seed") is None and record.get("terminal") is None:
            children = set(record.get("children", []))
            if not children.issubset(records):
                raise ValueError(
                    f"sample {key} is not a closed sample; choose a seed/terminal or include its child closure"
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "graph": str(args.graph),
        "shards": str(args.shards),
        "samples": len(records),
        "perShard": args.per_shard,
        "shardResults": {},
    }
    with tempfile.TemporaryDirectory(prefix="pathagon-tablebase-samples-") as temporary:
        temporary_root = Path(temporary)
        graph_path = temporary_root / "sample.jsonl"
        graph_path.write_text(
            "".join(json.dumps(records[key], sort_keys=True) + "\n" for key in sorted(records)),
            encoding="utf-8",
        )
        output_path = temporary_root / "values.json"
        shards_path = temporary_root / "shards"
        subprocess.run(
            [
                str(args.solver),
                "--input",
                str(graph_path),
                "--out",
                str(output_path),
                "--format",
                "json",
                "--shards",
                str(shards_path),
                "--shard-count",
                "1",
                "--workers",
                "1",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        solved = json.loads(output_path.read_text(encoding="utf-8"))["values"]

    for key, sample in samples.items():
        actual = solved.get(key)
        if actual != sample["expected"]:
            raise ValueError(f"sample {key} changed on deterministic re-solve: {actual!r}")
        shard = str(sample["shard"])
        result = report["shardResults"].setdefault(shard, {"rows": 0, "keys": []})
        result["rows"] += 1
        result["keys"].append(key)
    for result in report["shardResults"].values():
        result["keys"].sort()
    report["status"] = "pass"
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
