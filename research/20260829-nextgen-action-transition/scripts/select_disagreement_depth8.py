#!/usr/bin/env python3
"""Extract only roots whose depth-8 teacher changes the depth-7 action."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def action_key(action: dict) -> tuple:
    return action["kind"], action.get("from"), action.get("to")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path, help="combined depth-7 labels")
    parser.add_argument("--deep", required=True, type=Path, help="balanced depth-8 labels")
    parser.add_argument("--roots", required=True, type=Path, help="balanced roots")
    parser.add_argument("--output-targets", required=True, type=Path)
    parser.add_argument("--output-roots", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    args = parser.parse_args()

    base = {row["id"]: row for row in read_rows(args.base)}
    deep = {row["id"]: row for row in read_rows(args.deep)}
    roots = {row["id"]: row for row in read_rows(args.roots)}
    if len(base) != len(read_rows(args.base)) or len(deep) != len(read_rows(args.deep)):
        raise SystemExit("duplicate labels")
    if not set(deep) <= set(base):
        raise SystemExit("depth-8 labels must be a subset of the depth-7 labels")
    if set(deep) != set(roots):
        raise SystemExit("depth-8 labels and roots must cover the same balanced set")
    disagreements = [
        root_id
        for root_id in sorted(deep)
        if action_key(base[root_id]["teacherAction"])
        != action_key(deep[root_id]["teacherAction"])
    ]
    selected_targets = [deep[root_id] for root_id in disagreements]
    selected_roots = [roots[root_id] for root_id in disagreements]
    if not selected_targets:
        raise SystemExit("no teacher disagreements found")
    if any(row["teacher"]["depth"] != 8 or row["teacher"]["maxNodes"] != 2_000_000 for row in selected_targets):
        raise SystemExit("unexpected depth-8 teacher configuration")

    for path in (args.output_targets, args.output_roots, args.output_summary):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_targets.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected_targets),
        encoding="utf-8",
    )
    args.output_roots.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected_roots),
        encoding="utf-8",
    )
    summary = {
        "baseLabels": len(base),
        "deepLabels": len(deep),
        "balancedRoots": len(roots),
        "disagreementRoots": len(disagreements),
        "agreementRoots": len(deep) - len(disagreements),
        "agreementRate": (len(deep) - len(disagreements)) / len(deep),
        "turns": Counter(root["state"].split(".")[4] for root in selected_roots),
        "partitions": Counter(row["partition"] for row in selected_targets),
        "teacher": selected_targets[0]["teacher"],
        "outputTargets": str(args.output_targets),
        "outputRoots": str(args.output_roots),
    }
    args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True, default=dict))


if __name__ == "__main__":
    main()
