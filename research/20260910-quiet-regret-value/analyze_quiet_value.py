#!/usr/bin/env python3
"""Validate and summarize the quiet-regret/value experiment artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_report(path: Path) -> dict:
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"{path}: no rows")
    phases = Counter()
    turns = Counter()
    opponents = Counter()
    partitions = Counter()
    games_by_partition: dict[str, set[str]] = {"train": set(), "heldout": set()}
    margins = []
    continuation_differences = 0
    exhausted_rows = 0
    for row in rows:
        if row.get("schemaVersion") != 1:
            raise ValueError(f"{path}: unsupported schema")
        candidates = row["candidateActions"]
        lengths = [len(row[key]) for key in ("teacherScores", "teacherValues", "softPolicy", "continuationOutcomes")]
        if any(length != len(candidates) for length in lengths):
            raise ValueError(f"{row['id']}: action-conditioned vectors are misaligned")
        if not math.isclose(sum(row["softPolicy"]), 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError(f"{row['id']}: soft policy does not sum to one")
        if row["teacherMargin"] < 100:
            raise ValueError(f"{row['id']}: regret margin below configured gate")
        phase = row["phase"]
        partition = row["partition"]
        phases[phase] += 1
        turns[row["turn"]] += 1
        opponents[row["opponentType"]] += 1
        partitions[partition] += 1
        games_by_partition.setdefault(partition, set()).add(row["gameKey"])
        margins.append(row["teacherMargin"])
        continuation_differences += int(bool(row["continuationDifference"]))
        exhausted_rows += int(row["teacherExhausted"] > 0)
    overlap = games_by_partition["train"] & games_by_partition["heldout"]
    if overlap:
        raise ValueError(f"train/heldout game leakage: {sorted(overlap)[:3]}")
    return {
        "rows": len(rows),
        "games": len({row["gameKey"] for row in rows}),
        "phases": dict(sorted(phases.items())),
        "turns": dict(sorted(turns.items())),
        "opponentTypes": dict(sorted(opponents.items())),
        "partitions": dict(sorted(partitions.items())),
        "trainGames": len(games_by_partition["train"]),
        "heldoutGames": len(games_by_partition["heldout"]),
        "minimumTeacherMargin": min(margins),
        "meanTeacherMargin": sum(margins) / len(margins),
        "maximumTeacherMargin": max(margins),
        "continuationDifferenceRows": continuation_differences,
        "rowsWithAnyExhaustion": exhausted_rows,
    }


def arena_report(path: Path) -> dict:
    rows = load_jsonl(path)
    if not rows:
        raise ValueError(f"{path}: no games")
    candidate_id = rows[0]["agents"]["light"]
    wins = losses = draws = 0
    for row in rows:
        champion_is_light = row["agents"]["light"] == candidate_id
        winner = row.get("winner")
        if winner is None:
            draws += 1
        elif (winner == "light") == champion_is_light:
            wins += 1
        else:
            losses += 1
    games = len(rows)
    return {
        "file": str(path),
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "gamePoints": (wins + 0.5 * draws) / games,
        "plies": sum(row["plies"] for row in rows),
        "nodes": sum(move["nodes"] for row in rows for move in row["moves"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--arena-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "mode": "quiet-regret-value",
        "artifacts": {
            "targets": sha256(args.targets),
            "training": sha256(args.training),
            "checkpoint": sha256(args.training.with_name("quiet-value.pt"))
            if args.training.with_name("quiet-value.pt").exists()
            else None,
            "onnx": sha256(args.training.with_name("quiet-value-qadv.onnx"))
            if args.training.with_name("quiet-value-qadv.onnx").exists()
            else None,
        },
        "dataset": dataset_report(args.targets),
        "training": json.loads(args.training.read_text(encoding="utf-8")),
        "arenas": [arena_report(path) for path in sorted(args.arena_dir.glob("arena-*-learned.jsonl"))],
        "controls": [arena_report(path) for path in sorted(args.arena_dir.glob("arena-*-control.jsonl"))],
        "audits": {
            path.stem.removesuffix(".audit"): json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(args.arena_dir.glob("arena-*.audit.json"))
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
