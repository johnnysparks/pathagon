#!/usr/bin/env python3
"""Mine a small, reproducible adversarial/regression registry from 7x7 games.

The registry stores seed-level references and measured properties rather than
copying replay payloads into the training tree.  This keeps hard examples
available for capped training and regression review without silently doubling
their weight in corpus scans.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAMES = REPO_ROOT / "research/runs/gnn/benchmark-7x7/generated/batch-20260826-rust-lambda-20k-20260826/games"
DEFAULT_STAGING = REPO_ROOT / "research/runs/gnn/benchmark-7x7/generated/batch-20260826-rust-lambda-20k-20260826/local-backup-20260826/games"
DEFAULT_OUTPUT = REPO_ROOT / "research/adversarial/cases.json"
SEED_BOUNDARY_REGRESSION = 2026201789


def seed_from_path(path: Path) -> int | None:
    try:
        return int(path.stem.removeprefix("game-"))
    except ValueError:
        return None


def load_record(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    record = payload.get("record", payload)
    return record if isinstance(record, dict) and isinstance(record.get("moves"), list) else None


def action_token(move: dict[str, Any]) -> str:
    action = move.get("action") or {}
    kind = action.get("kind", "?")
    if kind == "place":
        return f"P{action.get('to', '?')}"
    if kind == "relocate":
        return f"R{action.get('from', '?')}>{action.get('to', '?')}"
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def metrics(record: dict[str, Any]) -> dict[str, Any]:
    moves = record["moves"]
    kinds = Counter((move.get("action") or {}).get("kind", "unknown") for move in moves)
    captures = [len(move.get("captured", [])) for move in moves]
    q_gaps: list[float] = []
    q_zero_gap_moves = 0
    q_covered = 0
    for move in moves:
        values = move.get("actionValues")
        visits = move.get("actionVisits")
        if not isinstance(values, list) or not isinstance(visits, list) or not values or not visits:
            continue
        q_covered += 1
        ordered = sorted(
            float(value)
            for value, visit in zip(values, visits)
            if int(visit) > 0 and math.isfinite(float(value))
        )
        if len(ordered) >= 2:
            gap = ordered[-1] - ordered[-2]
            q_gaps.append(gap)
            if abs(gap) <= 1.0e-8:
                q_zero_gap_moves += 1
    first_relocation = next(
        (int(move.get("ply", index + 1)) for index, move in enumerate(moves) if (move.get("action") or {}).get("kind") == "relocate"),
        None,
    )
    plies = len(moves)
    return {
        "plies": plies,
        "result": record.get("result"),
        "reason": record.get("reason"),
        "winner": record.get("winner"),
        "captures": sum(captures),
        "maxCapture": max(captures, default=0),
        "actionKinds": dict(sorted(kinds.items())),
        "uniqueActions": len({action_token(move) for move in moves}),
        "firstRelocationPly": first_relocation,
        "openingPlaceMoves": sum(
            1 for move in moves[:16] if (move.get("action") or {}).get("kind") == "place"
        ),
        "qCoveredPositions": q_covered,
        "qCoverage": q_covered / max(1, plies - 1),
        "minimumVisitedQTopGap": min(q_gaps, default=0.0),
        "minimumPositiveVisitedQTopGap": min((gap for gap in q_gaps if gap > 1.0e-8), default=0.0),
        "meanVisitedQTopGap": sum(q_gaps) / len(q_gaps) if q_gaps else 0.0,
        "zeroGapMoves": q_zero_gap_moves,
    }


def collect(games_dirs: list[Path], seed_start: int, seed_end: int) -> dict[int, tuple[Path, dict[str, Any]]]:
    records: dict[int, tuple[Path, dict[str, Any]]] = {}
    for directory in games_dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("game-*.json")):
            seed = seed_from_path(path)
            if seed is None or not seed_start <= seed <= seed_end or seed in records:
                continue
            record = load_record(path)
            if record is not None:
                records[seed] = (path, record)
    return records


def relative_source(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-dir", type=Path, action="append", default=[])
    parser.add_argument("--seed-start", type=int, default=2026200000)
    parser.add_argument("--seed-end", type=int, default=2026217499)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    directories = args.games_dir or [DEFAULT_GAMES, DEFAULT_STAGING]
    records = collect(directories, args.seed_start, args.seed_end)
    if not records:
        raise SystemExit("no valid games found in the requested seed range")

    measured = {
        seed: {"seed": seed, "source": relative_source(path), **metrics(record)}
        for seed, (path, record) in records.items()
    }
    used: set[int] = set()
    cases: list[dict[str, Any]] = []

    def add_case(case_id: str, category: str, seed: int, priority: str, rationale: str, expected: list[str]) -> None:
        if seed in used or seed not in measured:
            return
        used.add(seed)
        item = measured[seed]
        cases.append(
            {
                "id": case_id,
                "category": category,
                "priority": priority,
                "seed": seed,
                "source": item["source"],
                "rationale": rationale,
                "expected": expected,
                "observed": {key: value for key, value in item.items() if key not in {"seed", "source"}},
                "trainingUse": "hard-example; cap at one trajectory weight",
                "evaluationUse": "regression/manual review; exclude this seed from held-out metrics",
            }
        )

    add_case(
        "engine-max-ply-boundary-2026201789",
        "engine-regression",
        SEED_BOUNDARY_REGRESSION,
        "critical",
        "Previously crossed the stale internal 180-ply limit while the match recipe allowed 196 plies; retain as a native/Rust boundary regression.",
        ["completes without panic", "internal and match max-plies agree", "all decision plies retain Q/A targets"],
    )
    for index, item in enumerate(sorted(measured.values(), key=lambda value: (-value["plies"], value["seed"]))[:6], start=1):
        add_case(
            f"long-horizon-{index:02d}-{item['seed']}",
            "long-horizon",
            item["seed"],
            "high",
            "Long trajectory stresses late-game search, terminal handling, and reserve exhaustion.",
            ["replays legally", "does not lose late-ply Q coverage", "termination reason matches final state"],
        )
    for index, item in enumerate(sorted(measured.values(), key=lambda value: (-value["captures"], -value["maxCapture"], value["seed"]))[:6], start=1):
        add_case(
            f"capture-density-{index:02d}-{item['seed']}",
            "capture-density",
            item["seed"],
            "high",
            "Capture-heavy trajectory supplies transition-rich action/value targets and tactical edge cases.",
            ["captured squares replay exactly", "capture transitions retain aligned action values", "no illegal relocation follows a capture"],
        )
    for index, item in enumerate(
        sorted(
            (value for value in measured.values() if value["plies"] >= 12),
            key=lambda value: (value["uniqueActions"] / value["plies"], value["uniqueActions"], value["seed"]),
        )[:6],
        start=1,
    ):
        add_case(
            f"low-action-variety-{index:02d}-{item['seed']}",
            "action-variety",
            item["seed"],
            "medium",
            "Low action variety can expose shortcut learning and over-concentration on a narrow move motif.",
            ["is not duplicated by exact trajectory key", "retains legal alternatives in ranked targets", "does not dominate the training pool"],
        )
    for index, item in enumerate(
        sorted(
            (value for value in measured.values() if value["qCoveredPositions"] >= 2),
            key=lambda value: (value["minimumPositiveVisitedQTopGap"], value["meanVisitedQTopGap"], value["seed"]),
        )[:6],
        start=1,
    ):
        add_case(
            f"q-margin-ambiguity-{index:02d}-{item['seed']}",
            "q-margin-ambiguity",
            item["seed"],
            "high",
            "Smallest observed Q top-gap creates a hard ranking example where nearby actions should not be treated as interchangeable noise.",
            ["Q/action alignment remains exact", "ranking loss sees multiple visited actions", "manual review records the selected move and runner-up"],
        )
    for index, item in enumerate(
        sorted(measured.values(), key=lambda value: (value["firstRelocationPly"] is None, value["firstRelocationPly"] or 0, value["seed"]))[:6],
        start=1,
    ):
        add_case(
            f"placement-transition-{index:02d}-{item['seed']}",
            "placement-transition",
            item["seed"],
            "medium",
            "Early placement-to-relocation transition stresses the phase boundary where the model has historically been weakest.",
            ["placement actions remain legal", "phase transition is represented in metadata", "Q coverage includes the transition position"],
        )

    cases.sort(key=lambda item: (item["priority"] != "critical", item["category"], item["seed"]))
    payload = {
        "schemaVersion": 1,
        "boardSize": 7,
        "rulesVersion": "pathagon-rules-v1",
        "purpose": "Capped hard-example and regression cases for Q/Advantage training; never bulk-ingest this registry as a second corpus.",
        "sourceSeedRange": {"start": args.seed_start, "end": args.seed_end},
        "sourceDirectories": [relative_source(directory) for directory in directories],
        "selection": {
            "method": "seed-level metrics mined from valid JSON records",
            "categories": ["engine-regression", "long-horizon", "capture-density", "action-variety", "q-margin-ambiguity", "placement-transition"],
            "maxCasesPerCategoryBeforeDeduplication": 6,
        },
        "challengeProfiles": {
            "placement-exploration": {
                "purpose": "broaden early placement decisions while retaining Pathfinder as a light prior",
                "openingMoves": 20,
                "openingTemperature": 2.4,
                "openingRandomness": 0.55,
                "placementGuidance": 0.20,
                "pathfinderGuidance": 0.45,
                "temperatureMoves": 64,
                "simulations": 128,
                "tacticalSimulations": 512,
            },
            "ranking-ambiguity": {
                "purpose": "collect nearby-action Q/A rankings instead of only decisive argmax moves",
                "openingMoves": 16,
                "openingTemperature": 1.8,
                "openingRandomness": 0.35,
                "policyTemperature": 1.35,
                "pathfinderGuidance": 0.35,
                "placementGuidance": 0.25,
                "temperatureMoves": 64,
                "simulations": 128,
                "tacticalSimulations": 512,
            },
            "capture-pressure": {
                "purpose": "increase transition and capture diversity without dropping the Q/A head",
                "openingMoves": 18,
                "openingTemperature": 2.0,
                "openingRandomness": 0.45,
                "policyTemperature": 1.50,
                "temperatureMoves": 72,
                "pathfinderGuidance": 0.25,
                "placementGuidance": 0.10,
                "tacticalSimulations": 512,
            },
            "long-horizon": {
                "purpose": "stress late-game Q/A targets and the 196-ply contract boundary",
                "maxPlies": 196,
                "openingMoves": 16,
                "openingTemperature": 1.8,
                "openingRandomness": 0.30,
                "pathfinderGuidance": 0.50,
                "placementGuidance": 0.30,
                "temperatureMoves": 48,
                "simulations": 128,
                "tacticalSimulations": 512,
            },
        },
        "cases": cases,
    }
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sourceGames": len(measured), "cases": len(cases), "categories": Counter(item["category"] for item in cases)}, sort_keys=True))


if __name__ == "__main__":
    main()
