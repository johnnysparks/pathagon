#!/usr/bin/env python3
"""Audit a directory of Pathagon self-play game artifacts.

This is the corpus-wide companion to ``analyze-selfplay-batch.py``.  It is
streaming over individual game files so it can audit large campaigns without
loading every trajectory into memory, while retaining bounded candidate and
review queues for human inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_ANALYZER_SPEC = importlib.util.spec_from_file_location(
    "pathagon_batch_analyzer", REPO_ROOT / "scripts/analyze-selfplay-batch.py"
)
if _ANALYZER_SPEC is None or _ANALYZER_SPEC.loader is None:
    raise RuntimeError("could not load the batch analyzer")
_ANALYZER = importlib.util.module_from_spec(_ANALYZER_SPEC)
sys.modules[_ANALYZER_SPEC.name] = _ANALYZER
_ANALYZER_SPEC.loader.exec_module(_ANALYZER)
action_key = _ANALYZER.action_key
inverse_relocations = _ANALYZER.inverse_relocations
repeated_blocks = _ANALYZER.repeated_blocks
replay_state_metrics = _ANALYZER.replay_state_metrics


def quantiles(values: Iterable[int | float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": ordered[0],
        "p50": ordered[(len(ordered) - 1) // 2],
        "p95": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def entropy(values: Iterable[float]) -> float:
    positive = [float(value) for value in values if float(value) > 0]
    total = sum(positive)
    if total <= 0:
        return 0.0
    return -sum((value / total) * math.log2(value / total) for value in positive)


def read_record(path: Path) -> tuple[int | None, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    if isinstance(payload.get("record"), dict):
        wrapper_seed = payload.get("seed")
        return (int(wrapper_seed) if wrapper_seed is not None else None), payload["record"]
    seed = payload.get("seed")
    return (int(seed) if seed is not None else None), payload


def iter_games(games_dir: Path) -> Iterator[tuple[Path, int | None, dict[str, Any]]]:
    for path in sorted(games_dir.glob("game-*.json")):
        wrapper_seed, record = read_record(path)
        yield path, wrapper_seed, record


def action_signature(record: dict[str, Any]) -> str:
    return " ".join(action_key(move.get("action", {})) for move in record.get("moves", []))


def opening_signature(record: dict[str, Any], plies: int) -> str:
    return " ".join(action_key(move.get("action", {})) for move in record.get("moves", [])[:plies])


def q_metrics(record: dict[str, Any]) -> tuple[int, int, int, float, float, float]:
    moves = record.get("moves", [])
    terminal_closures = sum(
        1
        for index, move in enumerate(moves)
        if index == len(moves) - 1
        and not move.get("actionValues")
        and not move.get("actionVisits")
        and int(move.get("nodes", 0)) == 0
    )
    eligible = len(moves) - terminal_closures
    covered = [move for move in moves if move.get("actionValues") and move.get("actionVisits")]
    spreads: list[float] = []
    gaps: list[float] = []
    visit_entropies: list[float] = []
    for move in covered:
        values = [float(value) for value in move.get("actionValues", [])]
        if values:
            ordered = sorted(values, reverse=True)
            spreads.append(max(values) - min(values))
            gaps.append(ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0])
        visit_entropies.append(entropy(move.get("actionVisits", [])))
    return (
        eligible,
        terminal_closures,
        len(covered),
        statistics.fmean(spreads) if spreads else 0.0,
        statistics.fmean(gaps) if gaps else 0.0,
        statistics.fmean(visit_entropies) if visit_entropies else 0.0,
    )


def candidate_payload(path: Path, wrapper_seed: int | None, record: dict[str, Any], opening_plies: int) -> dict[str, Any]:
    moves = record.get("moves", [])
    tokens = [action_key(move.get("action", {})) for move in moves]
    eligible, terminal, covered, spread, gap, visit_entropy = q_metrics(record)
    replay = replay_state_metrics(record)
    flags: list[str] = []
    if record.get("reason") == "max-plies":
        flags.append("max-plies")
    if inverse_relocations(moves) >= 2:
        flags.append("back-and-forth")
    if repeated_blocks(tokens):
        flags.append("repeated-block")
    if replay["repeatedPositions"]:
        flags.append("state-repeat")
    if replay["threefoldPositions"]:
        flags.append("threefold-state")
    if not replay["ruleReplay"]:
        flags.append("illegal-record")
    if replay["captureMismatches"]:
        flags.append("capture-mismatch")
    if len(set(tokens)) <= max(3, len(tokens) // 8) and len(tokens) >= 20:
        flags.append("low-action-variety")
    if covered < eligible:
        flags.append("partial-q-coverage")
    seed = wrapper_seed if wrapper_seed is not None else record.get("seed")
    return {
        "path": str(path),
        "seed": int(seed) if seed is not None else None,
        "result": record.get("result"),
        "reason": record.get("reason"),
        "plies": len(moves),
        "captures": sum(len(move.get("captured", [])) for move in moves),
        "uniqueActions": len(set(tokens)),
        "opening": opening_signature(record, opening_plies),
        "openingHash": hashlib.sha256(opening_signature(record, opening_plies).encode()).hexdigest()[:16],
        "inverseRelocations": inverse_relocations(moves),
        "repeatedBlocks": repeated_blocks(tokens),
        "stateMetrics": replay,
        "qCoveredPositions": covered,
        "qEligiblePositions": eligible,
        "terminalClosurePositions": terminal,
        "qCoverage": covered / eligible if eligible else 1.0,
        "qSpreadMean": spread,
        "qTopGapMean": gap,
        "qVisitEntropyMean": visit_entropy,
        "flags": flags,
    }


def rank_interesting(item: dict[str, Any]) -> tuple[float, ...]:
    return (item["captures"], item["uniqueActions"], item["qSpreadMean"], item["plies"])


def rank_suspicious(item: dict[str, Any]) -> tuple[float, ...]:
    return (
        len(item["flags"]),
        item["inverseRelocations"],
        len(item["repeatedBlocks"]),
        int(item["qCoverage"] < 1.0),
        item["plies"],
    )


def keep_top(items: list[dict[str, Any]], item: dict[str, Any], ranker, limit: int) -> None:
    items.append(item)
    items.sort(key=ranker, reverse=True)
    del items[limit:]


def write_text_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "Pathagon self-play corpus audit",
        f"Games: {report['games']} / {report['gamesRequested']} requested",
        f"Seed coverage: {report['seedCoverage']['present']} present, {len(report['seedCoverage']['missing'])} missing, {len(report['seedCoverage']['unexpected'])} unexpected",
        f"Rule replay: {report['ruleReplayGames']} valid; capture-mismatch games: {report['captureMismatchGames']}",
        f"Repeated-state games: {report['stateRepeatGames']}; threefold-state games: {report['threefoldStateGames']}",
        f"Exact duplicate games: {report['exactDuplicateGames']} in {len(report['exactDuplicateGroups'])} groups",
        f"Q coverage: {report['qCoveredPositions']} / {report['qEligiblePositions']} eligible positions ({report['qCoverage']:.2%}); terminal closures: {report['terminalClosurePositions']}",
        f"Openings: {report['uniqueOpenings']} unique {report['openingPlies']}-ply prefixes; repeated-prefix excess: {report['repeatedOpeningExcess']}",
        "",
        "Most repeated openings:",
    ]
    for item in report["commonOpenings"][:10]:
        lines.append(f"  {item['count']:>4}x  {item['opening']}")
    lines.extend(["", "Suspicious candidates:"])
    for item in report["suspicious"][:20]:
        lines.append(
            f"  seed={item['seed']} plies={item['plies']} captures={item['captures']} "
            f"flags={','.join(item['flags']) or '-'} opening={item['opening']}"
        )
    lines.extend(["", "Interesting candidates:"])
    for item in report["interesting"][:20]:
        lines.append(
            f"  seed={item['seed']} plies={item['plies']} captures={item['captures']} "
            f"uniqueActions={item['uniqueActions']} qSpread={item['qSpreadMean']:.3f} opening={item['opening']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--games-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opening-plies", type=int, default=12)
    parser.add_argument("--review-sample", type=int, default=20)
    parser.add_argument("--expected-seed-start", type=int)
    parser.add_argument("--expected-seed-end", type=int)
    parser.add_argument("--limit", type=int, help="audit only the first N game files (for smoke tests)")
    args = parser.parse_args()
    if args.opening_plies < 1 or args.review_sample < 1:
        raise SystemExit("--opening-plies and --review-sample must be positive")
    if (args.expected_seed_start is None) != (args.expected_seed_end is None):
        raise SystemExit("--expected-seed-start and --expected-seed-end must be provided together")
    if args.expected_seed_start is not None and args.expected_seed_end < args.expected_seed_start:
        raise SystemExit("expected seed end must not precede expected seed start")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    games_dir = args.games_dir or args.manifest.parent / "games"
    expected_start = args.expected_seed_start if args.expected_seed_start is not None else int(manifest["seedStart"])
    expected_end = args.expected_seed_end if args.expected_seed_end is not None else int(manifest["seedEnd"])
    expected = set(range(expected_start, expected_end + 1))
    actual_seeds: set[int] = set()
    seed_files: dict[int, str] = {}
    exact_counts: Counter[str] = Counter()
    opening_counts: Counter[str] = Counter()
    results: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    action_kinds: Counter[str] = Counter()
    lengths: list[int] = []
    captures: list[int] = []
    interesting: list[dict[str, Any]] = []
    suspicious: list[dict[str, Any]] = []
    review_sample: list[tuple[str, Path]] = []
    duplicate_seed_files: Counter[int] = Counter()
    excluded_seed_files: list[int] = []
    positions = 0
    q_covered = 0
    q_eligible = 0
    terminal_closures = 0
    rule_replay_games = 0
    state_repeat_games = 0
    threefold_games = 0
    capture_mismatch_games = 0
    malformed_files: list[dict[str, Any]] = []

    for index, (path, wrapper_seed, record) in enumerate(iter_games(games_dir)):
        if args.limit is not None and index >= args.limit:
            break
        try:
            item = candidate_payload(path, wrapper_seed, record, args.opening_plies)
        except (KeyError, TypeError, ValueError, IndexError) as error:
            malformed_files.append({"path": str(path), "error": str(error)})
            continue
        seed_value = item["seed"]
        if seed_value is not None and seed_value not in expected:
            excluded_seed_files.append(int(seed_value))
            continue
        if seed_value is None:
            malformed_files.append({"path": str(path), "error": "missing seed"})
        else:
            actual_seeds.add(seed_value)
            if seed_value in seed_files:
                duplicate_seed_files[seed_value] += 1
            else:
                seed_files[seed_value] = str(path)
            if record.get("seed") is not None and int(record["seed"]) != seed_value:
                malformed_files.append({"path": str(path), "error": "wrapper/record seed mismatch"})
        moves = record.get("moves", [])
        for move in moves:
            action_kinds[str((move.get("action") or {}).get("kind", "unknown"))] += 1
        signature = action_signature(record)
        exact_counts[signature] += 1
        opening_counts[item["opening"]] += 1
        results[str(record.get("result"))] += 1
        reasons[str(record.get("reason"))] += 1
        lengths.append(item["plies"])
        captures.append(item["captures"])
        positions += item["plies"]
        q_covered += item["qCoveredPositions"]
        q_eligible += item["qEligiblePositions"]
        terminal_closures += item["terminalClosurePositions"]
        replay = item["stateMetrics"]
        rule_replay_games += int(replay["ruleReplay"])
        state_repeat_games += int(bool(replay["repeatedPositions"]))
        threefold_games += int(bool(replay["threefoldPositions"]))
        capture_mismatch_games += int(bool(replay["captureMismatches"]))
        keep_top(interesting, item, rank_interesting, 20)
        keep_top(suspicious, item, rank_suspicious, 20)
        if item["seed"] is not None:
            review_score = hashlib.sha256(f"review:{item['seed']}".encode()).hexdigest()
            if len(review_sample) < args.review_sample:
                review_sample.append((review_score, path))
                review_sample.sort(key=lambda pair: pair[0])
            elif review_score < review_sample[-1][0]:
                review_sample[-1] = (review_score, path)
                review_sample.sort(key=lambda pair: pair[0])

    duplicate_groups = [
        {"count": count, "trajectoryHash": hashlib.sha256(signature.encode()).hexdigest()}
        for signature, count in exact_counts.items()
        if count > 1
    ]
    duplicate_groups.sort(key=lambda item: item["count"], reverse=True)
    repeated_openings = [
        {"count": count, "opening": opening}
        for opening, count in opening_counts.most_common(20)
        if count > 1
    ]
    missing = sorted(expected - actual_seeds)
    unexpected = sorted(actual_seeds - expected)
    manifest_failed = sorted({int(item["seed"]) for item in manifest.get("failedSeeds", []) if "seed" in item})
    report = {
        "schemaVersion": 1,
        "manifest": str(args.manifest),
        "gamesDir": str(games_dir),
        "gamesRequested": len(expected),
        "games": len(lengths),
        "positions": positions,
        "results": dict(results),
        "reasons": dict(reasons),
        "actionKinds": dict(action_kinds),
        "lengths": quantiles(lengths),
        "captures": quantiles(captures),
        "qCoveredPositions": q_covered,
        "qEligiblePositions": q_eligible,
        "qCoverage": q_covered / q_eligible if q_eligible else 1.0,
        "terminalClosurePositions": terminal_closures,
        "ruleReplayGames": rule_replay_games,
        "stateRepeatGames": state_repeat_games,
        "threefoldStateGames": threefold_games,
        "captureMismatchGames": capture_mismatch_games,
        "malformedFiles": malformed_files,
        "duplicateSeedFiles": {str(seed): count for seed, count in sorted(duplicate_seed_files.items())},
        "excludedOutOfRangeSeedFiles": sorted(excluded_seed_files),
        "exactDuplicateGroups": duplicate_groups,
        "exactDuplicateGames": sum(item["count"] - 1 for item in duplicate_groups),
        "openingPlies": args.opening_plies,
        "uniqueOpenings": len(opening_counts),
        "repeatedOpeningExcess": sum(count - 1 for count in opening_counts.values() if count > 1),
        "commonOpenings": repeated_openings,
        "seedCoverage": {
            "expectedStart": expected_start,
            "expectedEnd": expected_end,
            "present": len(actual_seeds),
            "missing": missing,
            "unexpected": unexpected,
            "manifestFailedSeeds": manifest_failed,
        },
        "suspicious": suspicious,
        "interesting": interesting,
        "reviewSamplePaths": [str(path) for _, path in sorted(review_sample)],
        "manifestStatus": manifest.get("status"),
        "manifestEstimatedComputeUsd": manifest.get("estimatedComputeUsd"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "final-corpus-audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "final-seed-inventory.json").write_text(
        json.dumps({"expected": sorted(expected), "present": sorted(actual_seeds), "missing": missing, "unexpected": unexpected, "files": seed_files}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "final-review-sample.jsonl").open("w", encoding="utf-8") as output:
        for _, path in sorted(review_sample):
            text = path.read_text(encoding="utf-8")
            output.write(text)
            if not text.endswith("\n"):
                output.write("\n")
    write_text_report(args.output_dir / "final-corpus-audit.txt", report)
    print(json.dumps({
        "games": report["games"],
        "gamesRequested": report["gamesRequested"],
        "missingSeeds": len(missing),
        "unexpectedSeeds": len(unexpected),
        "duplicates": report["exactDuplicateGames"],
        "ruleReplayGames": report["ruleReplayGames"],
        "qCoverage": report["qCoverage"],
        "stateRepeatGames": report["stateRepeatGames"],
        "malformedFiles": len(malformed_files),
        "excludedOutOfRangeSeedFiles": len(excluded_seed_files),
    }, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
