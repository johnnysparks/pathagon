#!/usr/bin/env python3
"""Compare teacher actions on the same roots at two search budgets."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def read(pattern: str) -> dict[str, dict]:
    rows = {}
    for path in sorted(glob.glob(pattern)):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def action_key(action: dict) -> tuple:
    return action["kind"], action.get("from"), action["to"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shallow", required=True)
    parser.add_argument("--deep", required=True)
    args = parser.parse_args()
    shallow = read(args.shallow)
    deep = read(args.deep)
    shared = sorted(set(shallow) & set(deep))
    if not shared:
        raise SystemExit("no shared roots")
    agreement = sum(
        action_key(shallow[root]["teacherAction"]) == action_key(deep[root]["teacherAction"])
        for root in shared
    )
    turns = Counter(shallow[root]["state"].split(".")[4] for root in shared)
    depth_pairs = Counter(
        (shallow[root]["completedDepth"], deep[root]["completedDepth"]) for root in shared
    )
    print(
        json.dumps(
            {
                "sharedRoots": len(shared),
                "actionAgreement": agreement,
                "actionAgreementRate": agreement / len(shared),
                "turns": turns,
                "completedDepthPairs": {f"{left}->{right}": count for (left, right), count in depth_pairs.items()},
                "shallowTeacher": shallow[shared[0]]["teacher"],
                "deepTeacher": deep[shared[0]]["teacher"],
            },
            sort_keys=True,
            default=dict,
        )
    )


if __name__ == "__main__":
    main()
