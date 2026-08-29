#!/usr/bin/env python3
"""Audit self-play JSONL archives and emit a compact manual-review queue.

The audit uses the archive's serialized actions and targets, then replays each
record through the shared Python rules adapter to detect actual repeated
positions and capture mismatches. It can inspect Python and Rust archives and
is designed for large files, retaining only summaries and a bounded review
sample.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_GAME_SPEC = importlib.util.spec_from_file_location("pathagon_game_rules", REPO_ROOT / "research/20260824-gnn-cnn-lab/python/game.py")
if _GAME_SPEC is None or _GAME_SPEC.loader is None:
    raise RuntimeError("could not load the shared Python rules adapter")
_GAME_MODULE = importlib.util.module_from_spec(_GAME_SPEC)
sys.modules[_GAME_SPEC.name] = _GAME_MODULE
_GAME_SPEC.loader.exec_module(_GAME_MODULE)
Action = _GAME_MODULE.Action
BoardConfig = _GAME_MODULE.BoardConfig
GameState = _GAME_MODULE.GameState
bits = _GAME_MODULE.bits
repetition_key = _GAME_MODULE.repetition_key


def action_key(action: dict[str, Any]) -> str:
    kind = action.get("kind", "?")
    if kind == "place":
        return f"P{action.get('to', '?')}"
    if kind == "relocate":
        return f"R{action.get('from', '?')}>{action.get('to', '?')}"
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def action_kind(action: dict[str, Any]) -> str:
    return str(action.get("kind", "unknown"))


def action_signature(record: dict[str, Any]) -> str:
    return " ".join(action_key(move.get("action", {})) for move in record.get("moves", []))


def opening_signature(record: dict[str, Any], plies: int) -> str:
    return " ".join(action_key(move.get("action", {})) for move in record.get("moves", [])[:plies])


def entropy(values: Iterable[float]) -> float:
    probabilities = [float(value) for value in values if float(value) > 0]
    total = sum(probabilities)
    if total <= 0:
        return 0.0
    return -sum((value / total) * math.log2(value / total) for value in probabilities)


def quantiles(values: list[int | float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    ordered = sorted(float(value) for value in values)
    return {
        "min": ordered[0],
        "p50": ordered[(len(ordered) - 1) // 2],
        "p95": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def iter_records(paths: list[Path]) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                yield path, line_number, record


def inverse_relocations(moves: list[dict[str, Any]]) -> int:
    count = 0
    for before, after in zip(moves, moves[1:]):
        first = before.get("action", {})
        second = after.get("action", {})
        if (
            first.get("kind") == "relocate"
            and second.get("kind") == "relocate"
            and first.get("from") == second.get("to")
            and first.get("to") == second.get("from")
        ):
            count += 1
    return count


def repeated_blocks(tokens: list[str], max_block: int = 6) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for size in range(2, min(max_block, len(tokens) // 2) + 1):
        for start in range(0, len(tokens) - (2 * size) + 1):
            if tokens[start : start + size] == tokens[start + size : start + 2 * size]:
                blocks.append({"startPly": start + 1, "blockPlies": size, "repeats": 2})
                break
    return blocks


def replay_state_metrics(record: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct rule-relevant positions and detect real cycles."""

    config_value = record.get("config") or {}
    config = BoardConfig(
        int(config_value.get("boardSize", 7)),
        int(config_value.get("reservePerPlayer", 14)),
        int(config_value.get("maxPlies", 196)),
    )
    state = GameState.initial(config)
    seen: Counter[tuple] = Counter()
    capture_mismatches = 0
    illegal_ply: int | None = None
    for index, move in enumerate(record.get("moves", []), start=1):
        seen[repetition_key(state)] += 1
        raw_action = move.get("action", {})
        if raw_action.get("kind") == "place":
            action = Action.place(int(raw_action["to"]))
        elif raw_action.get("kind") == "relocate":
            action = Action.relocate(int(raw_action["from"]), int(raw_action["to"]))
        else:
            illegal_ply = index
            break
        if action not in state.legal_actions():
            illegal_ply = index
            break
        next_state = state.apply_legal(action)
        expected_captured = list(bits(next_state.forbidden))
        actual_captured = sorted(int(square) for square in move.get("captured", []))
        if expected_captured != actual_captured:
            capture_mismatches += 1
        state = next_state
    seen[repetition_key(state)] += 1
    repeat_counts = [count for count in seen.values() if count > 1]
    return {
        "ruleReplay": illegal_ply is None,
        "illegalPly": illegal_ply,
        "captureMismatches": capture_mismatches,
        "repeatedPositions": sum(count - 1 for count in repeat_counts),
        "threefoldPositions": sum(1 for count in seen.values() if count >= 3),
        "maxPositionVisits": max(seen.values(), default=0),
    }


def summarize_record(path: Path, line_number: int, record: dict[str, Any], opening_plies: int) -> dict[str, Any]:
    moves = record.get("moves", [])
    tokens = [action_key(move.get("action", {})) for move in moves]
    kinds = Counter(action_kind(move.get("action", {})) for move in moves)
    captures = sum(len(move.get("captured", [])) for move in moves)
    q_moves = [move for move in moves if move.get("actionValues") and move.get("actionVisits")]
    terminal_closures = [
        move
        for index, move in enumerate(moves)
        if index == len(moves) - 1
        and not move.get("actionValues")
        and not move.get("actionVisits")
        and int(move.get("nodes", 0)) == 0
    ]
    decision_plies = len(moves) - len(terminal_closures)
    q_spreads = []
    q_top_gaps = []
    q_visit_entropies = []
    for move in q_moves:
        values = [float(value) for value in move["actionValues"]]
        if values:
            ordered = sorted(values, reverse=True)
            q_spreads.append(max(values) - min(values))
            q_top_gaps.append(ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0])
        q_visit_entropies.append(entropy(move.get("actionVisits", [])))

    flags: list[str] = []
    inverse_count = inverse_relocations(moves)
    blocks = repeated_blocks(tokens)
    state_metrics = replay_state_metrics(record)
    reason = record.get("reason")
    if reason == "max-plies":
        flags.append("max-plies")
    if inverse_count >= 2:
        flags.append("back-and-forth")
    if blocks:
        flags.append("repeated-block")
    if state_metrics["repeatedPositions"]:
        flags.append("state-repeat")
    if state_metrics["threefoldPositions"]:
        flags.append("threefold-state")
    if not state_metrics["ruleReplay"]:
        flags.append("illegal-record")
    if state_metrics["captureMismatches"]:
        flags.append("capture-mismatch")
    if len(set(tokens)) <= max(3, len(tokens) // 8) and len(tokens) >= 20:
        flags.append("low-action-variety")
    if len(q_moves) < decision_plies:
        flags.append("partial-q-coverage")

    return {
        "source": str(path),
        "line": line_number,
        "seed": record.get("seed"),
        "result": record.get("result"),
        "reason": reason,
        "plies": len(moves),
        "decisionPlies": decision_plies,
        "terminalClosurePositions": len(terminal_closures),
        "captures": captures,
        "actionKinds": dict(sorted(kinds.items())),
        "uniqueActions": len(set(tokens)),
        "opening": opening_signature(record, opening_plies),
        "openingHash": hashlib.sha256(opening_signature(record, opening_plies).encode()).hexdigest()[:16],
        "inverseRelocations": inverse_count,
        "repeatedBlocks": blocks,
        "stateMetrics": state_metrics,
        "qCoveredPositions": len(q_moves),
        "qCoverage": len(q_moves) / decision_plies if decision_plies else 1.0,
        "qSpreadMean": statistics.fmean(q_spreads) if q_spreads else 0.0,
        "qTopGapMean": statistics.fmean(q_top_gaps) if q_top_gaps else 0.0,
        "qVisitEntropyMean": statistics.fmean(q_visit_entropies) if q_visit_entropies else 0.0,
        "flags": flags,
        "actions": tokens,
    }


def expand_inputs(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.jsonl")))
        else:
            paths.append(path)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing input archive(s): " + ", ".join(map(str, missing)))
    if not paths:
        raise ValueError("no JSONL archives found")
    return paths


def choose_review_sample(items: list[dict[str, Any]], sample_size: int, seed: int) -> list[dict[str, Any]]:
    if len(items) <= sample_size:
        return sorted(items, key=lambda item: (item.get("seed") is None, item.get("seed"), item["line"]))
    rng = random.Random(seed)
    sample = rng.sample(items, sample_size)
    return sorted(sample, key=lambda item: (item.get("seed") is None, item.get("seed"), item["line"]))


def write_text_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"Self-play audit: {report['inputs']}",
        f"Games: {report['games']} | positions: {report['positions']} | exact duplicate games: {report['exactDuplicateGames']}",
        f"Length: {report['lengths']}",
        f"Results: {json.dumps(report['results'], sort_keys=True)}",
        f"Reasons: {json.dumps(report['reasons'], sort_keys=True)}",
        f"Rule replay: {report['ruleReplayGames']} valid / {report['games']} games; state-repeat games: {report['stateRepeatGames']}",
        "",
        "Most common opening prefixes:",
    ]
    for item in report["commonOpenings"][:10]:
        lines.append(f"  {item['count']:>4}x  {item['opening']}")
    lines.extend(["", "Suspicious candidates:"])
    for item in report["suspicious"][:20]:
        lines.append(
            f"  seed={item.get('seed')} plies={item['plies']} captures={item['captures']} "
            f"flags={','.join(item['flags']) or '-'} maxStateVisits={item['stateMetrics']['maxPositionVisits']} opening={item['opening']}"
        )
    lines.extend(["", "Interesting candidates:"])
    for item in report["interesting"][:20]:
        lines.append(
            f"  seed={item.get('seed')} plies={item['plies']} captures={item['captures']} "
            f"uniqueActions={item['uniqueActions']} qSpread={item['qSpreadMean']:.3f} "
            f"opening={item['opening']}"
        )
    lines.extend(["", "Manual-review sample:"])
    for item in report["reviewSample"]:
        lines.append(
            f"  seed={item.get('seed')} plies={item['plies']} captures={item['captures']} "
            f"flags={','.join(item['flags']) or '-'} actions={' '.join(item['actions'])}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="JSONL archives or directories containing JSONL archives")
    parser.add_argument("--opening-plies", type=int, default=12)
    parser.add_argument("--sample-games", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=20260826)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--text-report", type=Path)
    parser.add_argument("--sample-output", type=Path, help="write the selected records as a review JSONL")
    args = parser.parse_args()
    if args.opening_plies < 1 or args.sample_games < 1:
        raise SystemExit("--opening-plies and --sample-games must be positive")

    paths = expand_inputs(args.inputs)
    records: list[dict[str, Any]] = []
    exact_counts: Counter[str] = Counter()
    opening_counts: Counter[str] = Counter()
    results: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    action_kinds: Counter[str] = Counter()
    positions = 0
    q_positions = 0
    q_eligible_positions = 0
    terminal_closure_positions = 0
    for path, line_number, record in iter_records(paths):
        summary = summarize_record(path, line_number, record, args.opening_plies)
        records.append(summary)
        exact_counts[" ".join(summary["actions"])] += 1
        opening_counts[summary["opening"]] += 1
        results[str(record.get("result"))] += 1
        reasons[str(record.get("reason"))] += 1
        action_kinds.update(summary["actionKinds"])
        positions += summary["plies"]
        q_positions += summary["qCoveredPositions"]
        q_eligible_positions += summary["decisionPlies"]
        terminal_closure_positions += summary["terminalClosurePositions"]

    duplicate_groups = sorted((count for count in exact_counts.values() if count > 1), reverse=True)
    common_openings = [
        {"count": count, "opening": opening}
        for opening, count in opening_counts.most_common(20)
    ]
    suspicious = sorted(
        records,
        key=lambda item: (
            len(item["flags"]),
            item["inverseRelocations"],
            len(item["repeatedBlocks"]),
            item["qCoverage"] < 1.0,
            item["plies"],
        ),
        reverse=True,
    )
    common_opening_counts = dict(opening_counts)
    for item in records:
        if common_opening_counts[item["opening"]] >= 3 and "common-opening" not in item["flags"]:
            item["flags"].append("common-opening")
    suspicious = sorted(
        records,
        key=lambda item: (len(item["flags"]), item["inverseRelocations"], len(item["repeatedBlocks"]), item["plies"]),
        reverse=True,
    )
    interesting = sorted(
        records,
        key=lambda item: (
            item["captures"],
            item["uniqueActions"],
            item["qSpreadMean"],
            item["plies"],
        ),
        reverse=True,
    )
    report = {
        "schemaVersion": 1,
        "inputs": [str(path) for path in paths],
        "games": len(records),
        "positions": positions,
        "qCoveredPositions": q_positions,
        "qEligiblePositions": q_eligible_positions,
        "terminalClosurePositions": terminal_closure_positions,
        "results": dict(results),
        "reasons": dict(reasons),
        "actionKinds": dict(action_kinds),
        "lengths": quantiles([item["plies"] for item in records]),
        "captures": quantiles([item["captures"] for item in records]),
        "qCoverage": q_positions / q_eligible_positions if q_eligible_positions else 1.0,
        "ruleReplayGames": sum(1 for item in records if item["stateMetrics"]["ruleReplay"]),
        "stateRepeatGames": sum(1 for item in records if item["stateMetrics"]["repeatedPositions"]),
        "threefoldStateGames": sum(1 for item in records if item["stateMetrics"]["threefoldPositions"]),
        "captureMismatchGames": sum(1 for item in records if item["stateMetrics"]["captureMismatches"]),
        "exactDuplicateGroups": duplicate_groups,
        "exactDuplicateGames": sum(count - 1 for count in exact_counts.values() if count > 1),
        "commonOpenings": common_openings,
        "suspicious": suspicious[:20],
        "interesting": interesting[:20],
        "reviewSample": choose_review_sample(records, args.sample_games, args.sample_seed),
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if args.text_report:
        args.text_report.parent.mkdir(parents=True, exist_ok=True)
        write_text_report(args.text_report, report)
    if args.sample_output:
        args.sample_output.parent.mkdir(parents=True, exist_ok=True)
        by_location = {(item["source"], item["line"]): item for item in report["reviewSample"]}
        with args.sample_output.open("w", encoding="utf-8") as output:
            for path, line_number, record in iter_records(paths):
                if (str(path), line_number) in by_location:
                    output.write(json.dumps(record, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
