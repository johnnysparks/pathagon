#!/usr/bin/env python3
"""Fit a small phase-conditioned evaluator against super-deep Rust labels."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
from pathlib import Path

BASELINE = (241, 112, 887, 40, 154, 74)
FEATURES = ("path", "material", "capture", "structure", "threat", "edge")
PHASES = ("opening", "placement", "movement", "late-game")


def identity(action: dict) -> tuple:
    value = action["action"]
    if value.get("kind") == "place":
        return ("place", int(value["to"]))
    return ("relocate", int(value["from"]), int(value["to"]))


def action_order(action: dict) -> int:
    value = action["action"]
    return int(value["to"]) if value.get("kind") == "place" else int(value["from"]) * 64 + int(value["to"])


def load_rows(pattern: str) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        rows.extend(json.loads(line) for line in Path(path).read_text().splitlines() if line.strip())
    if not rows:
        raise ValueError(f"no targets matched {pattern}")
    return rows


def score(action: dict, weights: tuple[int, ...]) -> int:
    if action["immediateWin"]:
        return 2_000_000_000
    return int(action["captureCount"]) * 10_000 + sum(
        int(feature) * weight for feature, weight in zip(action["features"], weights)
    )


def choose(row: dict, weights: tuple[int, ...]) -> dict:
    return max(row["actions"], key=lambda action: (score(action, weights), -action_order(action)))


def metrics(rows: list[dict], vectors: dict[str, tuple[int, ...]]) -> dict:
    result = {phase: {"roots": 0, "top1": 0, "immediate": 0, "immediateCorrect": 0, "forced": 0, "forcedCorrect": 0} for phase in PHASES}
    for row in rows:
        phase = row["phase"] if row["phase"] in vectors else "placement"
        bucket = result[phase]
        bucket["roots"] += 1
        selected = choose(row, vectors[phase])
        bucket["top1"] += int(identity(selected) == identity({"action": row["teacherAction"]}))
        immediate = any(action["immediateWin"] for action in row["actions"])
        if immediate:
            bucket["immediate"] += 1
            bucket["immediateCorrect"] += int(selected["immediateWin"])
        safe = [action for action in row["actions"] if action["safe"]]
        if len(safe) < len(row["actions"]):
            bucket["forced"] += 1
            bucket["forcedCorrect"] += int(selected["safe"])
    for bucket in result.values():
        bucket["top1Rate"] = bucket["top1"] / bucket["roots"] if bucket["roots"] else None
        bucket["immediateAccuracy"] = bucket["immediateCorrect"] / bucket["immediate"] if bucket["immediate"] else None
        bucket["forcedAccuracy"] = bucket["forcedCorrect"] / bucket["forced"] if bucket["forced"] else None
    result["all"] = {
        "roots": sum(bucket["roots"] for bucket in result.values()),
        "top1": sum(bucket["top1"] for bucket in result.values()),
    }
    result["all"]["top1Rate"] = result["all"]["top1"] / result["all"]["roots"] if result["all"]["roots"] else None
    return result


def prepare(rows: list[dict]) -> list[tuple[str, int, list[tuple[tuple[int, ...], int, bool, bool]]]]:
    prepared = []
    for row in rows:
        phase = row["phase"] if row["phase"] in PHASES else "placement"
        teacher = identity({"action": row["teacherAction"]})
        actions = []
        teacher_index = 0
        for index, action in enumerate(row["actions"]):
            if identity(action) == teacher:
                teacher_index = index
            actions.append((tuple(int(value) for value in action["features"]), int(action["captureCount"]), bool(action["immediateWin"]), bool(action["safe"])))
        prepared.append((phase, teacher_index, actions))
    return prepared


def objective_prepared(
    rows: list[tuple[str, int, list[tuple[tuple[int, ...], int, bool, bool]]]],
    vectors: dict[str, tuple[int, ...]],
    regularization: float,
) -> float:
    loss = 0.0
    for phase, teacher_index, actions in rows:
        weights = vectors[phase]
        def action_score(action: tuple[tuple[int, ...], int, bool, bool]) -> int:
            features, captures, immediate, _safe = action
            return 2_000_000_000 if immediate else captures * 10_000 + sum(feature * weight for feature, weight in zip(features, weights))
        teacher_score = action_score(actions[teacher_index])
        rival = max((action_score(action) for index, action in enumerate(actions) if index != teacher_index), default=teacher_score)
        loss += max(0, 10_000 + rival - teacher_score)
        if any(action[2] for action in actions):
            loss += 2_000_000_000 * (not actions[teacher_index][2])
        if any(not action[3] for action in actions):
            loss += 500_000 * (not actions[teacher_index][3])
    for vector in vectors.values():
        loss += sum((value - base) ** 2 for value, base in zip(vector, BASELINE)) * regularization
    return loss


def mutate(vectors: dict[str, tuple[int, ...]], rng: random.Random) -> dict[str, tuple[int, ...]]:
    result = dict(vectors)
    phase = rng.choice(PHASES)
    values = list(result[phase])
    index = rng.randrange(len(values))
    scale = max(4, BASELINE[index] // 5)
    values[index] = max(1, int(round(values[index] + rng.gauss(0, scale))))
    result[phase] = tuple(values)
    return result


def optimize(rows: list[dict], seeds: list[int], iterations: int, regularization: float) -> tuple[dict[str, tuple[int, ...]], list[dict]]:
    prepared = prepare(rows)
    baseline = {phase: BASELINE for phase in PHASES}
    best = baseline
    best_loss = objective_prepared(prepared, best, regularization)
    runs = []
    for seed in seeds:
        rng = random.Random(seed)
        current = dict(baseline)
        current_loss = objective_prepared(prepared, current, regularization)
        accepted = 0
        for _ in range(iterations):
            candidate = mutate(current, rng)
            candidate_loss = objective_prepared(prepared, candidate, regularization)
            if candidate_loss < current_loss:
                current, current_loss = candidate, candidate_loss
                accepted += 1
                if candidate_loss < best_loss:
                    best, best_loss = candidate, candidate_loss
        runs.append({"seed": seed, "accepted": accepted, "objective": current_loss, "vectors": {phase: list(current[phase]) for phase in PHASES}})
    return best, runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=5_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260829, 20260830, 20260831])
    parser.add_argument("--min-completed-depth", type=int, default=6)
    parser.add_argument("--regularization", type=float, default=0.2)
    args = parser.parse_args()
    rows = load_rows(args.targets)
    usable = [row for row in rows if int(row.get("completedDepth", 0)) >= args.min_completed_depth]
    if not usable:
        raise ValueError("no rows meet --min-completed-depth")
    train = [row for row in usable if row["partition"] == "train"]
    heldout = [row for row in usable if row["partition"] == "heldout"]
    vectors, runs = optimize(train, args.seeds, args.iterations, args.regularization)
    baseline = {phase: BASELINE for phase in PHASES}
    report = {
        "schemaVersion": 1,
        "teacher": rows[0]["teacher"],
        "regularization": args.regularization,
        "featureOrder": list(FEATURES),
        "phases": list(PHASES),
        "roots": {
            "all": len(rows),
            "usable": len(usable),
            "train": len(train),
            "heldout": len(heldout),
            "minCompletedDepth": args.min_completed_depth,
            "excludedByCompletedDepth": len(rows) - len(usable),
        },
        "baseline": {"weights": list(BASELINE), "train": metrics(train, baseline), "heldout": metrics(heldout, baseline)},
        "candidate": {"vectors": {phase: list(vectors[phase]) for phase in PHASES}, "train": metrics(train, vectors), "heldout": metrics(heldout, vectors)},
        "runs": runs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "contextual-weights.json").write_text(json.dumps({"schemaVersion": 1, "teacher": rows[0]["teacher"], "regularization": args.regularization, "weights": {phase: dict(zip(FEATURES, vectors[phase])) for phase in PHASES}}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"train": len(train), "heldout": len(heldout), "baselineHeldout": report["baseline"]["heldout"]["all"], "candidateHeldout": report["candidate"]["heldout"]["all"]}, sort_keys=True))


if __name__ == "__main__":
    main()
