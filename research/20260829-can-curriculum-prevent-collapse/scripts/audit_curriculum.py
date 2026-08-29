#!/usr/bin/env python3
"""Audit the frozen curriculum portfolio and its held-out evidence.

This is intentionally a research-path tool. It validates the source boundaries,
replays JSONL game records with the existing Python rules adapter, measures
trajectory/root/phase concentration, and summarizes any final native arenas.
It does not promote data or alter the canonical corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
LAB_ROOT = REPO_ROOT / "research/20260824-gnn-cnn-lab"
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from python.data import load_replay_examples  # type: ignore  # noqa: E402
from python.game import BoardConfig  # type: ignore  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def game_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and isinstance(value.get("games"), list):
        return [item for item in value["games"] if isinstance(item, dict) and isinstance(item.get("moves"), list)]
    if isinstance(value, list):
        records: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("record"), dict):
                item = item["record"]
            if isinstance(item.get("games"), list):
                records.extend(game_records(item))
            elif isinstance(item.get("moves"), list):
                records.append(item)
        return records
    if isinstance(value, dict) and isinstance(value.get("record"), dict):
        return game_records(value["record"])
    if isinstance(value, dict) and isinstance(value.get("moves"), list):
        return [value]
    return []


def root_provenance(record: dict[str, Any]) -> tuple[str, str]:
    provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
    family = provenance.get("rootFamilyId") or record.get("rootFamilyId")
    root_class = provenance.get("rootClass") or record.get("rootClass")
    if family is None:
        root = record.get("initialPosition")
        family = "root-" + hashlib.sha256(json.dumps(root, sort_keys=True).encode()).hexdigest()[:16]
    return str(family), str(root_class or "ordinary")


def action_signature(record: dict[str, Any]) -> str:
    return json.dumps([move.get("action") for move in record.get("moves", [])], sort_keys=True, separators=(",", ":"))


def phase_metrics(record: dict[str, Any]) -> Counter[str]:
    initial = record.get("initialPosition") if isinstance(record.get("initialPosition"), dict) else {}
    raw_reserve = initial.get("reserve") if isinstance(initial.get("reserve"), dict) else {}
    reserves = {"light": int(raw_reserve.get("light", 14)), "dark": int(raw_reserve.get("dark", 14))}
    counts: Counter[str] = Counter()
    for move in record.get("moves", []):
        player = str(move.get("player", "light"))
        counts["placement" if reserves[player] > 0 else "movement"] += 1
        action = move.get("action") or {}
        if action.get("kind") == "place":
            reserves[player] -= 1
        elif action.get("kind") == "relocate":
            counts["relocation"] += 1
        captured = len(move.get("captured", []))
        reserves["dark" if player == "light" else "light"] += captured
    return counts


def audit_source(source: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    path = REPO_ROOT / source["path"]
    result: dict[str, Any] = {
        "id": source["id"],
        "path": source["path"],
        "partition": source.get("partition"),
        "role": source.get("role"),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
    }
    if not path.is_file():
        result["error"] = "missing source"
        return result
    result["sha256"] = sha256_file(path)
    value = read_json(path)
    records = game_records(value)
    result["records"] = len(records)
    seeds = [int(record["seed"]) for record in records if record.get("seed") is not None]
    families = [root_provenance(record)[0] for record in records]
    classes = Counter(root_provenance(record)[1] for record in records)
    signatures = Counter(action_signature(record) for record in records)
    phase_counts: Counter[str] = Counter()
    for record in records:
        phase_counts.update(phase_metrics(record))
    result.update(
        {
            "seeds": sorted(set(seeds)),
            "seedCount": len(set(seeds)),
            "rootFamilies": len(set(families)),
            "rootFamilyIds": sorted(set(families)),
            "rootClasses": dict(sorted(classes.items())),
            "positions": sum(len(record.get("moves", [])) for record in records),
            "phaseCounts": dict(sorted(phase_counts.items())),
            "uniqueTrajectories": len(signatures),
            "duplicateTrajectoryGroups": sum(count > 1 for count in signatures.values()),
            "duplicateTrajectoryRecords": sum(count for count in signatures.values() if count > 1),
            "duplicateFraction": ((len(records) - len(signatures)) / len(records)) if records else 0.0,
        }
    )
    if source.get("validateReplay") and records:
        try:
            examples = load_replay_examples(path, BoardConfig(7, 14, 196))
            result["replayValid"] = True
            result["replayExamples"] = len(examples)
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["replayValid"] = False
            result["replayError"] = str(error)
    return result


def overlap_report(audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for left, right in itertools.combinations(audits, 2):
        left_partition = left.get("partition")
        right_partition = right.get("partition")
        if left_partition is None or right_partition is None or left_partition == right_partition:
            continue
        seeds = sorted(set(left.get("seeds", [])) & set(right.get("seeds", [])))
        families = sorted(set(left.get("rootFamilyIds", [])) & set(right.get("rootFamilyIds", [])))
        if seeds or families:
            report.append({"left": left["id"], "right": right["id"], "seedOverlap": seeds, "familyOverlap": families})
    return report


def root_score_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    classes = Counter(str(row.get("rootClass", "unknown")) for row in rows)
    turns = Counter(str(row.get("turn", "unknown")) for row in rows)
    near = sum(bool(row.get("nearTerminal")) for row in rows)
    return {
        "exists": True,
        "rows": len(rows),
        "sha256": sha256_file(path),
        "rootClasses": dict(sorted(classes.items())),
        "turns": dict(sorted(turns.items())),
        "nearTerminalRoots": near,
        "nearTerminalFraction": near / len(rows) if rows else 0.0,
        "uniqueRootFamilies": len({str(row.get("rootFamilyId")) for row in rows}),
    }


def arena_metrics(path: Path, candidate: str, opponent: str) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "path": str(path.relative_to(REPO_ROOT))}
    records = game_records(read_json(path))
    wins = losses = draws = 0
    by_color = {"light": Counter(), "dark": Counter()}
    for record in records:
        light = record.get("light_agent") or (record.get("agents") or {}).get("light")
        dark = record.get("dark_agent") or (record.get("agents") or {}).get("dark")
        winner = record.get("winner")
        candidate_color = "light" if light == candidate else "dark" if dark == candidate else None
        if winner is None:
            draws += 1
            if candidate_color:
                by_color[candidate_color]["draws"] += 1
        elif (winner == "light" and light == candidate) or (winner == "dark" and dark == candidate):
            wins += 1
            if candidate_color:
                by_color[candidate_color]["wins"] += 1
        else:
            losses += 1
            if candidate_color:
                by_color[candidate_color]["losses"] += 1
    games = wins + losses + draws
    return {
        "exists": True,
        "path": str(path.relative_to(REPO_ROOT)),
        "candidate": candidate,
        "opponent": opponent,
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "gamePoints": wins + draws * 0.5,
        "gamePointRate": (wins + draws * 0.5) / games if games else 0.0,
        "byCandidateColor": {key: dict(value) for key, value in by_color.items()},
        "seeds": sorted({int(record["seed"]) for record in records if record.get("seed") is not None}),
    }


def common_arena_metrics(path: Path) -> dict[str, Any]:
    """Summarize the Python common-seed learner-vs-learner arena format."""
    if not path.is_file():
        return {"exists": False, "path": str(path.relative_to(REPO_ROOT))}
    value = read_json(path)
    records = game_records(value)
    standings = value.get("standings", []) if isinstance(value, dict) else []
    by_agent_color: dict[str, dict[str, Counter[str]]] = {}
    for record in records:
        agents = record.get("agents") if isinstance(record.get("agents"), dict) else {}
        winner = record.get("winner")
        for color in ("light", "dark"):
            agent = agents.get(color)
            if agent is None:
                continue
            stats = by_agent_color.setdefault(str(agent), {"light": Counter(), "dark": Counter()})[color]
            outcome = "draws" if winner is None else "wins" if winner == color else "losses"
            stats[outcome] += 1
    return {
        "exists": True,
        "path": str(path.relative_to(REPO_ROOT)),
        "games": len(records),
        "standings": standings,
        "byAgentColor": {
            agent: {color: dict(stats) for color, stats in colors.items()}
            for agent, colors in sorted(by_agent_color.items())
        },
        "seeds": sorted({int(record["seed"]) for record in records if record.get("seed") is not None}),
        "recordsPresent": bool(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "portfolio-v1.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arena", action="append", default=[], help="optional native JSONL arena path, repeated as needed")
    parser.add_argument("--common-arena", action="append", default=[], help="optional Python common-seed arena JSON path")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    audits = [audit_source(source, manifest) for source in manifest["sources"]]
    by_id = {audit["id"]: audit for audit in audits}
    proposal = by_id["portfolio-games"]
    family_counts = Counter(proposal.get("rootClasses", {}))
    family_total = sum(family_counts.values())
    family_share = {key: value / family_total for key, value in family_counts.items()} if family_total else {}
    score_source = next(source for source in manifest["sources"] if source["id"] == "portfolio-root-scores")
    score_metrics = root_score_metrics(REPO_ROOT / score_source["path"])
    arena_reports = [arena_metrics(REPO_ROOT / path, "qadv-arbiter-7x7-rust-policy-v0.1.0", "pathfinder-v0.4.0-tactical-filter (v0.5 weights; historical-2k)") for path in args.arena]
    common_arena_reports = [common_arena_metrics(REPO_ROOT / path) for path in args.common_arena]
    overlaps = overlap_report([audit for audit in audits if audit.get("partition") is not None])
    replay_ok = all(audit.get("replayValid", True) for audit in audits if audit.get("partition") is not None and audit.get("role") in {"training-games", "selection-games"})
    movement_positions = proposal.get("phaseCounts", {}).get("movement", 0)
    position_total = proposal.get("positions", 0)
    gates = {
        "partitionLeakageZero": not overlaps,
        "replayValid": replay_ok,
        "duplicateTrajectoryFractionBelow5Percent": proposal.get("duplicateFraction", 1.0) < manifest["decisionRules"]["maxDuplicateFraction"],
        "fourSourceFamiliesAtLeast15Percent": sum(share >= 0.15 for share in family_share.values()) >= manifest["decisionRules"]["minimumFamiliesAt15Percent"],
        "movementPhaseMateriallyRepresented": bool(position_total) and movement_positions / position_total >= manifest["decisionRules"]["minimumMovementPositionFraction"],
        "balancedRootTurns": score_metrics.get("turns", {}).get("light", 0) == score_metrics.get("turns", {}).get("dark", 0),
        "finalStrengthEvidencePresent": bool(arena_reports),
        "productEnvelopeAvailable": bool(manifest["decisionRules"].get("productEnvelopeAvailable")),
    }
    result = {
        "schemaVersion": 1,
        "schema": "pathagon-curriculum-audit-v1",
        "manifest": str(args.manifest.relative_to(REPO_ROOT)),
        "sources": audits,
        "partitionOverlaps": overlaps,
        "proposal": {
            "sourceId": "portfolio-games",
            "familyShares": family_share,
            "rootScoreMetrics": score_metrics,
            "movementPositionFraction": movement_positions / position_total if position_total else 0.0,
        },
        "arenas": arena_reports,
        "commonArenas": common_arena_reports,
        "gates": gates,
        "promotionEligible": all(gates.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "promotionEligible": result["promotionEligible"], "gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
