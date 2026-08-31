#!/usr/bin/env python3
"""Aggregate Pathfinder strategy arena archives into a compact research report."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def empty_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": profile["id"],
        "label": profile.get("label", profile["id"]),
        "group": profile.get("group", "uncategorized"),
        "depth": profile["depth"],
        "beam": profile["beam"],
        "nodes": profile["nodes"],
        "deadlineMs": profile.get("deadlineMs"),
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "gamePoints": 0.0,
        "decisions": 0,
        "totalNodes": 0,
        "totalTableHits": 0,
        "completedDepthCounts": defaultdict(int),
        "budgetSaturatedDecisions": 0,
        "byColor": {
            "light": {"games": 0, "wins": 0, "losses": 0, "draws": 0},
            "dark": {"games": 0, "wins": 0, "losses": 0, "draws": 0},
        },
        "pointsByGame": [],
    }


def add_game(metrics: dict[str, Any], record: dict[str, Any], color: str) -> None:
    winner = record.get("winner")
    metrics["games"] += 1
    metrics["byColor"][color]["games"] += 1
    if winner is None:
        outcome = "draws"
        points = 0.5
    elif winner == color:
        outcome = "wins"
        points = 1.0
    else:
        outcome = "losses"
        points = 0.0
    metrics[outcome] += 1
    metrics["gamePoints"] += points
    metrics["pointsByGame"].append(points)
    metrics["byColor"][color][outcome] += 1
    for move in record.get("moves", []):
        if str(move.get("player", "")).lower() != color:
            continue
        nodes = int(move.get("nodes", 0))
        metrics["decisions"] += 1
        metrics["totalNodes"] += nodes
        metrics["totalTableHits"] += int(move.get("tableHits", 0))
        metrics["completedDepthCounts"][str(int(move.get("completedDepth", 0)))] += 1
        if nodes >= metrics["nodes"]:
            metrics["budgetSaturatedDecisions"] += 1


def finish_profile(metrics: dict[str, Any]) -> None:
    decisions = metrics["decisions"]
    games = metrics["games"]
    points = metrics["gamePoints"]
    metrics["gamePointShare"] = points / games if games else 0.0
    metrics["meanNodesPerDecision"] = metrics["totalNodes"] / decisions if decisions else 0.0
    metrics["meanNodesPerGame"] = metrics["totalNodes"] / games if games else 0.0
    metrics["meanNodesPerGamePoint"] = metrics["totalNodes"] / points if points else None
    metrics["budgetSaturationRate"] = metrics["budgetSaturatedDecisions"] / decisions if decisions else 0.0
    metrics["depth4Share"] = sum(
        count for depth, count in metrics["completedDepthCounts"].items() if int(depth) >= metrics["depth"]
    ) / decisions if decisions else 0.0
    metrics["completedDepthCounts"] = dict(sorted(metrics["completedDepthCounts"].items(), key=lambda item: int(item[0])))


def bootstrap_interval(values: list[float], rng: random.Random, samples: int) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    estimates = []
    for _ in range(samples):
        total = sum(values[rng.randrange(len(values))] for _ in values)
        estimates.append(total / len(values))
    estimates.sort()
    return [estimates[int(0.025 * (len(estimates) - 1))], estimates[int(0.975 * (len(estimates) - 1))]]


def validate_record(record: dict[str, Any], expected_candidate_id: str) -> list[str]:
    errors = []
    if record.get("contractVersion") != 1:
        errors.append("contractVersion")
    if record.get("winner") not in (None, "light", "dark"):
        errors.append("winner")
    agents = record.get("agents", {})
    if len(agents) != 2 or expected_candidate_id not in set(agents.values()):
        errors.append("agents")
    for move in record.get("moves", []):
        if str(move.get("player", "")).lower() not in {"light", "dark"}:
            errors.append("move-player")
            break
    return errors


def collect(campaign_dirs: list[Path], bootstrap_samples: int, bootstrap_seed: int) -> dict[str, Any]:
    profiles: dict[str, dict[str, Any]] = {}
    vs_control: dict[str, dict[str, Any]] = {}
    pairings: dict[str, dict[str, Any]] = {}
    validation_errors: list[dict[str, Any]] = []
    report_paths: set[Path] = set()
    rng = random.Random(bootstrap_seed)

    for campaign_dir in campaign_dirs:
        for report_path in sorted(campaign_dir.rglob("*.json")):
            if report_path.name == "campaign.json" or report_path in report_paths:
                continue
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("schema") != "pathagon-pathfinder-search-strategy-arena-v1":
                continue
            report_paths.add(report_path)
            champion = report["championProfile"]
            opponent = report["opponentProfile"]
            for profile in (champion, opponent):
                profiles.setdefault(profile["id"], empty_profile(profile))
            control_profile = next(
                (profile for profile in (champion, opponent) if profile.get("group") == "control"),
                None,
            )
            target_profile = None
            if control_profile is not None:
                target_profile = opponent if control_profile["id"] == champion["id"] else champion
                vs_control.setdefault(target_profile["id"], empty_profile(target_profile))
            archive = resolve_repo_path(report["gamesArchive"])
            pair_key = " vs ".join(sorted((champion["id"], opponent["id"])))
            pair = pairings.setdefault(pair_key, {
                "profiles": sorted((champion["id"], opponent["id"])),
                "reports": 0,
                "games": 0,
                "points": defaultdict(float),
                "pointsByGame": defaultdict(list),
                "wallSeconds": 0.0,
            })
            pair["reports"] += 1
            pair["wallSeconds"] += float(report.get("wallSeconds", 0.0))
            lines = archive.read_text(encoding="utf-8").splitlines()
            if len(lines) != int(report["protocol"]["games"]):
                validation_errors.append({"report": str(report_path), "error": "game-count"})
            for line_number, line in enumerate(lines, 1):
                record = json.loads(line)
                errors = validate_record(record, champion["id"])
                if errors:
                    validation_errors.append({"report": str(report_path), "line": line_number, "error": errors})
                for color, agent_id in record.get("agents", {}).items():
                    profile_id = champion["id"] if agent_id == champion["id"] else opponent["id"]
                    metrics = profiles[profile_id]
                    add_game(metrics, record, color)
                    points = metrics["pointsByGame"][-1]
                    pair["points"][profile_id] += points
                    pair["pointsByGame"][profile_id].append(points)
                    if target_profile is not None and profile_id == target_profile["id"]:
                        add_game(vs_control[profile_id], record, color)
                pair["games"] += 1

    for metrics in profiles.values():
        finish_profile(metrics)
        metrics["gamePointShare95Bootstrap"] = bootstrap_interval(metrics.pop("pointsByGame"), rng, bootstrap_samples)
    for metrics in vs_control.values():
        finish_profile(metrics)
        metrics["gamePointShare95Bootstrap"] = bootstrap_interval(metrics.pop("pointsByGame"), rng, bootstrap_samples)
    for pair in pairings.values():
        pair["points"] = dict(pair["points"])
        pair["pointsByGame"] = {
            profile_id: {"games": len(values), "gamePointShare": sum(values) / len(values) if values else 0.0,
                         "ci95Bootstrap": bootstrap_interval(values, rng, bootstrap_samples)}
            for profile_id, values in pair["pointsByGame"].items()
        }
    return {
        "schema": "pathagon-pathfinder-search-strategy-analysis-v1",
        "campaignDirs": [str(path) for path in campaign_dirs],
        "reports": len(report_paths),
        "profiles": list(sorted(profiles.values(), key=lambda item: (-item["gamePointShare"], item["id"]))),
        "vsControl": list(sorted(vs_control.values(), key=lambda item: (-item["gamePointShare"], item["id"]))),
        "pairings": dict(sorted(pairings.items())),
        "validation": {"reportsChecked": len(report_paths), "errors": validation_errors},
    }


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.1f}%"


def write_markdown(result: dict[str, Any], path: Path) -> None:
    profiles = result["profiles"]
    vs_control = result["vsControl"]
    lines = [
        "# Pathfinder search strategy analysis",
        "",
        f"Archives: {result['reports']} arena reports; structural validation errors: {len(result['validation']['errors'])}.",
        "",
        "## Fixed-control screen",
        "",
        "This is the comparable power screen: every profile below faced the same depth-4 / beam-8 / 2k control. Confidence intervals are deterministic bootstrap 95% intervals over raw game outcomes.",
        "",
        "| Rank | Profile | Group | Game points | 95% CI | W-L-D | Nodes/game | Nodes/game point | Depth target reached | Saturation |",
        "|---:|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, profile in enumerate(vs_control, 1):
        lines.append(
            f"| {rank} | `{profile['id']}` | {profile['group']} | {pct(profile['gamePointShare'])} | "
            f"{pct(profile['gamePointShare95Bootstrap'][0])}–{pct(profile['gamePointShare95Bootstrap'][1])} | "
            f"{profile['wins']}-{profile['losses']}-{profile['draws']} | {profile['meanNodesPerGame']:.0f} | "
            f"{profile['meanNodesPerGamePoint']:.0f} | {pct(profile['depth4Share'])} | {pct(profile['budgetSaturationRate'])} |"
        )
    lines += ["", "## Descriptive all-campaign metrics", "", "These totals mix fixed-control screens and direct tournament duels; use the direct pairing table for head-to-head conclusions.", "", "| Rank | Profile | Group | Game points | W-L-D | Nodes/game |", "|---:|---|---|---:|---:|---:|"]
    for rank, profile in enumerate(profiles, 1):
        lines.append(f"| {rank} | `{profile['id']}` | {profile['group']} | {pct(profile['gamePointShare'])} | {profile['wins']}-{profile['losses']}-{profile['draws']} | {profile['meanNodesPerGame']:.0f} |")
    lines += ["", "## Pairing outcomes", "", "| Pairing | Games | Game-point shares |", "|---|---:|---|"]
    for pair_key, pair in result["pairings"].items():
        shares = ", ".join(f"`{profile_id}` {pct(item['gamePointShare'])}" for profile_id, item in pair["pointsByGame"].items())
        lines.append(f"| `{pair_key}` | {pair['games']} | {shares} |")
    lines += ["", "## Validation", "", f"- Structural archive checks: {'PASS' if not result['validation']['errors'] else 'FAIL'}."]
    if result["validation"]["errors"]:
        lines.append(f"- First errors: {result['validation']['errors'][:5]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", action="append", required=True, type=Path)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026083131)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("bootstrap-samples must be positive")
    dirs = [resolve_repo_path(path) for path in args.campaign_dir]
    result = collect(dirs, args.bootstrap_samples, args.bootstrap_seed)
    out_json = resolve_repo_path(args.out_json)
    out_md = resolve_repo_path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, out_md)
    print(json.dumps({"reports": result["reports"], "profiles": len(result["profiles"]), "validationErrors": len(result["validation"]["errors"]), "markdown": str(out_md)}))


if __name__ == "__main__":
    main()
