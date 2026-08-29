#!/usr/bin/env python3
"""Train and audit the explainable Pathfinder evaluator on root-regret labels.

The script deliberately models only Pathfinder's cheap root ordering. The
native alpha-beta search remains the authority in the later arena. Source
groups are split before any weight search, and human tactical roots are held
out from training regardless of hash assignment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


BASELINE = (241, 112, 887, 40, 154, 74)
FEATURE_NAMES = ("path", "material", "capture", "structure", "threat", "edge")
MUTATION_SCALES = (60, 45, 220, 35, 75, 45)


def action_order(action: dict[str, Any]) -> int:
    action = action["action"]
    return int(action["to"]) if action["kind"] == "place" else int(action["from"]) * 64 + int(action["to"])


def action_identity(action: dict[str, Any]) -> tuple[Any, ...]:
    action = action.get("action", action)
    if action["kind"] == "place":
        return ("place", int(action["to"]))
    return ("relocate", int(action["from"]), int(action["to"]))


def load_targets(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("schemaVersion") != 1:
            raise ValueError(f"target line {line_number} has unsupported schema")
        if not row["actions"]:
            raise ValueError(f"target line {line_number} has no actions")
        rows.append(row)
    if not rows:
        raise ValueError("target file is empty")
    return rows


def group_key(row: dict[str, Any]) -> str:
    return f"{row['sourceFamily']}:{row['sourceGameId']}"


def split_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    assignment: dict[str, str] = {}
    for row in rows:
        group = group_key(row)
        if group in assignment:
            continue
        if row["sourceFamily"] == "human-tactical":
            assignment[group] = "heldout"
        else:
            digest = hashlib.sha256(group.encode("utf-8")).digest()
            assignment[group] = "heldout" if digest[0] % 5 == 0 else "train"
    training = [row for row in rows if assignment[group_key(row)] == "train"]
    heldout = [row for row in rows if assignment[group_key(row)] == "heldout"]
    if not training or not heldout:
        raise ValueError("source-disjoint split produced an empty partition")
    return training, heldout, assignment


def predicted_score(action: dict[str, Any], weights: tuple[int, ...]) -> int:
    if action["immediateWin"]:
        return 2_000_000_000
    return int(action["captureCount"]) * 10_000 + sum(
        int(feature) * int(weight) for feature, weight in zip(action["features"], weights)
    )


def select_action(row: dict[str, Any], weights: tuple[int, ...]) -> dict[str, Any]:
    return max(
        row["actions"],
        key=lambda action: (predicted_score(action, weights), -action_order(action)),
    )


def metrics(rows: list[dict[str, Any]], weights: tuple[int, ...]) -> dict[str, Any]:
    regrets = []
    nonterminal_regrets = []
    selected_teacher_scores = []
    top1 = 0
    immediate_total = 0
    immediate_correct = 0
    forced_total = 0
    forced_correct = 0
    selected_capture = 0
    for row in rows:
        selected = select_action(row, weights)
        best_score = max(int(action["teacherScore"]) for action in row["actions"])
        regret = best_score - int(selected["teacherScore"])
        regrets.append(regret)
        selected_teacher_scores.append(int(selected["teacherScore"]))
        best_actions = {action_identity(action) for action in row["teacherBestActions"]}
        top1 += int(action_identity(selected) in best_actions)
        has_immediate = any(action["immediateWin"] for action in row["actions"])
        if has_immediate:
            immediate_total += 1
            immediate_correct += int(selected["immediateWin"])
        if row["rootHasForcedBlock"]:
            forced_total += 1
            forced_correct += int(selected["safe"])
        if not has_immediate:
            nonterminal_regrets.append(regret)
        selected_capture += int(selected["captureCount"] > 0)
    teacher_actions = sum(len(row["actions"]) for row in rows)
    exhausted_actions = sum(
        int(action["teacherExhausted"])
        for row in rows
        for action in row["actions"]
    )
    return {
        "roots": len(rows),
        "actions": teacher_actions,
        "meanRegret": mean(regrets),
        "medianRegret": median(regrets),
        "meanNonterminalRegret": mean(nonterminal_regrets) if nonterminal_regrets else None,
        "top1Agreement": top1 / len(rows),
        "immediateWinRoots": immediate_total,
        "immediateWinAccuracy": immediate_correct / immediate_total if immediate_total else None,
        "forcedBlockRoots": forced_total,
        "forcedBlockAccuracy": forced_correct / forced_total if forced_total else None,
        "selectedCaptureRate": selected_capture / len(rows),
        "teacherExhaustedActionRate": exhausted_actions / teacher_actions,
        "selectedTeacherScoreMean": mean(selected_teacher_scores),
    }


def objective(rows: list[dict[str, Any]], weights: tuple[int, ...]) -> float:
    result = metrics(rows, weights)
    # Raw regret is primary. Tactical penalties keep the optimizer from
    # trading away the two safety properties that are not negotiable.
    loss = float(result["meanRegret"])
    if result["immediateWinAccuracy"] is not None:
        loss += (1.0 - result["immediateWinAccuracy"]) * 2_000_000_000.0
    if result["forcedBlockAccuracy"] is not None:
        loss += (1.0 - result["forcedBlockAccuracy"]) * 500_000.0
    return loss


def mutate(weights: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
    values = list(weights)
    touched = 1 if rng.random() < 0.72 else 2
    for index in rng.sample(range(len(values)), touched):
        scale = MUTATION_SCALES[index]
        if rng.random() < 0.14:
            values[index] = max(0, int(round(rng.uniform(0.0, BASELINE[index] * 2.5))))
        else:
            values[index] = max(0, int(round(values[index] + rng.gauss(0.0, scale))))
    return tuple(values)


def optimize(rows: list[dict[str, Any]], seeds: list[int], iterations: int) -> dict[str, Any]:
    baseline_loss = objective(rows, BASELINE)
    runs = []
    best_weights = BASELINE
    best_loss = baseline_loss
    for seed in seeds:
        rng = random.Random(seed)
        current = BASELINE
        current_loss = baseline_loss
        accepted = 0
        for _ in range(iterations):
            candidate = mutate(current, rng)
            candidate_loss = objective(rows, candidate)
            if candidate_loss < current_loss:
                current = candidate
                current_loss = candidate_loss
                accepted += 1
                if candidate_loss < best_loss:
                    best_weights = candidate
                    best_loss = candidate_loss
        runs.append({
            "seed": seed,
            "iterations": iterations,
            "accepted": accepted,
            "weights": list(current),
            "objective": current_loss,
            "metrics": metrics(rows, current),
        })
    return {
        "baseline": {"weights": list(BASELINE), "objective": baseline_loss, "metrics": metrics(rows, BASELINE)},
        "best": {"weights": list(best_weights), "objective": best_loss, "metrics": metrics(rows, best_weights)},
        "runs": runs,
    }


def family_metrics(rows: list[dict[str, Any]], weights: tuple[int, ...]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["sourceFamily"]].append(row)
    return {family: metrics(group, weights) for family, group in sorted(grouped.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260829, 20260830, 20260831])
    args = parser.parse_args()

    rows = load_targets(args.targets)
    training, heldout, assignment = split_rows(rows)
    result = optimize(training, args.seeds, args.iterations)
    candidate = tuple(result["best"]["weights"])
    report = {
        "schemaVersion": 1,
        "targetSchemaVersion": 1,
        "teacher": rows[0]["teacher"],
        "baselineWeights": list(BASELINE),
        "candidateWeights": list(candidate),
        "featureOrder": list(FEATURE_NAMES),
        "split": {
            "policy": "sha256(sourceFamily:sourceGameId) first byte mod 5; human-tactical always heldout",
            "trainingRoots": len(training),
            "heldoutRoots": len(heldout),
            "trainingGroups": sorted(group for group, part in assignment.items() if part == "train"),
            "heldoutGroups": sorted(group for group, part in assignment.items() if part == "heldout"),
        },
        "optimization": result,
        "heldout": {
            "baseline": metrics(heldout, BASELINE),
            "candidate": metrics(heldout, candidate),
            "byFamily": {
                "baseline": family_metrics(heldout, BASELINE),
                "candidate": family_metrics(heldout, candidate),
            },
        },
        "all": {
            "baseline": metrics(rows, BASELINE),
            "candidate": metrics(rows, candidate),
        },
        "byFamily": {
            "baseline": family_metrics(rows, BASELINE),
            "candidate": family_metrics(rows, candidate),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "candidate-weights.json").write_text(
        json.dumps({"schemaVersion": 1, "weights": dict(zip(FEATURE_NAMES, candidate)), "source": "root-regret-train-report"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "split-manifest.json").write_text(json.dumps(report["split"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"training": len(training), "heldout": len(heldout), "baseline": report["heldout"]["baseline"], "candidate": report["heldout"]["candidate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
