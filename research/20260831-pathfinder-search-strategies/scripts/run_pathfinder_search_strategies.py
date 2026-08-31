#!/usr/bin/env python3
"""Run paired ordinary-Pathfinder configuration arenas.

The Rust self-play binary owns rules, move selection, and replay records. This
wrapper supplies independent champion/opponent search configurations, stores
complete game archives under the experiment workspace, and derives per-profile
node/depth/table-hit measurements from those records.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = REPO_ROOT / "research/20260831-pathfinder-search-strategies"
DEFAULT_BINARY = REPO_ROOT / "pathagon/engine-rs/target/release/pathagon-selfplay"
DEFAULT_PROFILES = EXPERIMENT_ROOT / "profiles.json"


@dataclass(frozen=True)
class Profile:
    id: str
    label: str
    group: str
    depth: int
    beam: int
    nodes: int
    deadline_ms: int | None

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "Profile":
        profile = cls(
            id=str(value["id"]),
            label=str(value.get("label", value["id"])),
            group=str(value.get("group", "uncategorized")),
            depth=int(value["depth"]),
            beam=int(value["beam"]),
            nodes=int(value["nodes"]),
            deadline_ms=(int(value["deadlineMs"]) if value.get("deadlineMs") is not None else None),
        )
        if (
            not profile.id
            or profile.depth <= 0
            or profile.beam <= 0
            or profile.nodes <= 0
            or (profile.deadline_ms is not None and profile.deadline_ms <= 0)
        ):
            raise ValueError(f"invalid Pathfinder profile: {value!r}")
        return profile


def load_profiles(path: Path) -> tuple[str, list[Profile]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schemaVersion") != 1:
        raise ValueError(f"unsupported profile catalog schema in {path}")
    profiles = [Profile.from_json(value) for value in document["profiles"]]
    ids = [profile.id for profile in profiles]
    if len(ids) != len(set(ids)):
        raise ValueError("profile IDs must be unique")
    control_id = str(document["controlProfileId"])
    if control_id not in set(ids):
        raise ValueError(f"control profile {control_id!r} is not in {path}")
    return control_id, profiles


def pairings(
    profiles: list[Profile], control_id: str, round_robin: bool, selected_groups: set[str], selected_ids: set[str]
) -> list[tuple[Profile, Profile]]:
    filtered = [
        profile
        for profile in profiles
        if (not selected_groups or profile.group in selected_groups)
        and (not selected_ids or profile.id in selected_ids)
    ]
    if round_robin:
        return list(combinations(filtered, 2))
    control = next(profile for profile in profiles if profile.id == control_id)
    return [(profile, control) for profile in filtered if profile.id != control.id]


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_engine_output(stdout: str, stderr: str, expected_games: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("contractVersion") == 1:
            records.append(value)

    summary: dict[str, Any] | None = None
    for line in stderr.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schemaVersion") == 2:
            summary = value
    if summary is None:
        raise RuntimeError("Rust arena did not emit a schemaVersion=2 summary")
    if len(records) != expected_games:
        raise RuntimeError(f"Rust arena emitted {len(records)} games; expected {expected_games}")
    return records, summary


def empty_profile_metrics(profile: Profile) -> dict[str, Any]:
    return {
        "id": profile.id,
        "label": profile.label,
        "depth": profile.depth,
        "beam": profile.beam,
        "nodes": profile.nodes,
        "deadlineMs": profile.deadline_ms,
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "gamePoints": 0.0,
        "decisions": 0,
        "totalNodes": 0,
        "totalTableHits": 0,
        "completedDepthCounts": {},
        "budgetSaturatedDecisions": 0,
        "byColor": {
            "light": {"games": 0, "wins": 0, "losses": 0, "draws": 0},
            "dark": {"games": 0, "wins": 0, "losses": 0, "draws": 0},
        },
    }


def finalize_profile_metrics(metrics: dict[str, Any]) -> None:
    decisions = metrics["decisions"]
    games = metrics["games"]
    metrics["meanNodesPerDecision"] = metrics["totalNodes"] / decisions if decisions else 0.0
    metrics["meanNodesPerGame"] = metrics["totalNodes"] / games if games else 0.0
    metrics["meanNodesPerGamePoint"] = (
        metrics["totalNodes"] / metrics["gamePoints"] if metrics["gamePoints"] else None
    )
    metrics["budgetSaturationRate"] = (
        metrics["budgetSaturatedDecisions"] / decisions if decisions else 0.0
    )


def summarize_records(records: list[dict[str, Any]], champion: Profile, opponent: Profile) -> dict[str, Any]:
    metrics = {profile.id: empty_profile_metrics(profile) for profile in (champion, opponent)}
    for game_index, record in enumerate(records):
        champion_color = "light" if game_index % 2 == 0 else "dark"
        colors = {champion_color: champion, ("dark" if champion_color == "light" else "light"): opponent}
        winner = record.get("winner")
        for color, profile in colors.items():
            profile_metrics = metrics[profile.id]
            profile_metrics["games"] += 1
            profile_metrics["byColor"][color]["games"] += 1
            if winner is None:
                outcome = "draws"
                profile_metrics["draws"] += 1
                profile_metrics["gamePoints"] += 0.5
            elif winner == color:
                outcome = "wins"
                profile_metrics["wins"] += 1
                profile_metrics["gamePoints"] += 1.0
            else:
                outcome = "losses"
                profile_metrics["losses"] += 1
            profile_metrics["byColor"][color][outcome] += 1

        for move in record.get("moves", []):
            color = str(move.get("player", "")).lower()
            profile = colors.get(color)
            if profile is None:
                raise RuntimeError(f"game {game_index} contains unknown move player {color!r}")
            profile_metrics = metrics[profile.id]
            nodes = int(move.get("nodes", 0))
            completed_depth = int(move.get("completedDepth", 0))
            profile_metrics["decisions"] += 1
            profile_metrics["totalNodes"] += nodes
            profile_metrics["totalTableHits"] += int(move.get("tableHits", 0))
            depth_key = str(completed_depth)
            counts = profile_metrics["completedDepthCounts"]
            counts[depth_key] = counts.get(depth_key, 0) + 1
            if nodes >= profile.nodes:
                profile_metrics["budgetSaturatedDecisions"] += 1

    for profile_metrics in metrics.values():
        finalize_profile_metrics(profile_metrics)
    return {"profiles": list(metrics.values())}


def profile_dict(profile: Profile) -> dict[str, Any]:
    return {
        "id": profile.id,
        "label": profile.label,
        "group": profile.group,
        "depth": profile.depth,
        "beam": profile.beam,
        "nodes": profile.nodes,
        "deadlineMs": profile.deadline_ms,
    }


def run_pairing(
    binary: Path,
    champion: Profile,
    opponent: Profile,
    pairing_index: int,
    args: argparse.Namespace,
    output_dir: Path,
    match_id: str | None = None,
    match_seed: int | None = None,
) -> dict[str, Any]:
    pair_id = f"{champion.id}-vs-{opponent.id}"
    match_id = match_id or pair_id
    seed = match_seed if match_seed is not None else args.seed + pairing_index * 10_000
    command = [
        str(binary),
        "--games", str(args.games),
        "--seed", str(seed),
        "--max-plies", str(args.max_plies),
        "--opening-random-plies", str(args.opening_random_plies),
        "--board-size", "7",
        "--reserve", "14",
        "--depth", str(opponent.depth),
        "--beam", str(opponent.beam),
        "--nodes", str(opponent.nodes),
        "--candidate-id", champion.id,
        "--candidate-depth", str(champion.depth),
        "--candidate-beam", str(champion.beam),
        "--candidate-nodes", str(champion.nodes),
        "--no-tactical-root-filter",
        "--opponent", "deep-search",
        "--workers", str(args.workers),
        "--progress-every", str(max(1, args.games // 10)),
        "--jsonl",
    ]
    if champion.deadline_ms is not None:
        command.extend(["--candidate-deadline-ms", str(champion.deadline_ms)])
    if opponent.deadline_ms is not None:
        command.extend(["--opponent-deadline-ms", str(opponent.deadline_ms)])
    output = output_dir / f"{match_id}.json"
    archive = output_dir / f"{match_id}.games.jsonl"
    if args.dry_run:
        print(" ".join(command))
        return {
            "matchId": match_id,
            "pairId": pair_id,
            "champion": champion.id,
            "opponent": opponent.id,
            "seed": seed,
            "command": command,
            "status": "dry-run",
        }

    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    if completed.returncode:
        failure = output_dir / f"{pair_id}.stderr.txt"
        failure.write_text(completed.stderr or completed.stdout, encoding="utf-8")
        raise RuntimeError(f"arena failed for {pair_id}; see {relative(failure)}")
    records, engine_summary = parse_engine_output(completed.stdout, completed.stderr, args.games)
    archive.write_text("\n".join(json.dumps(record, separators=(",", ":")) for record in records) + "\n", encoding="utf-8")
    report = {
        "schema": "pathagon-pathfinder-search-strategy-arena-v1",
        "matchId": match_id,
        "pairId": pair_id,
        "binary": relative(binary),
        "gamesArchive": relative(archive),
        "seed": seed,
        "protocol": {
            "games": args.games,
            "maxPlies": args.max_plies,
            "openingRandomPlies": args.opening_random_plies,
            "boardSize": 7,
            "reservePerPlayer": 14,
            "tacticalRootFilter": False,
            "opponentMode": "deep-search",
        },
        "championProfile": profile_dict(champion),
        "opponentProfile": profile_dict(opponent),
        "engineSummary": engine_summary,
        "wallSeconds": round(elapsed, 6),
        "observed": summarize_records(records, champion, opponent),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pairId": pair_id, "wallSeconds": round(elapsed, 3), "observed": report["observed"]}, sort_keys=True))
    profile_metrics = {
        item["id"]: item for item in report["observed"]["profiles"]
    }
    return {
        "matchId": match_id,
        "pairId": pair_id,
        "champion": champion.id,
        "opponent": opponent.id,
        "report": relative(output),
        "gamesArchive": relative(archive),
        "status": "completed",
        "gamePoints": {profile_id: item["gamePoints"] for profile_id, item in profile_metrics.items()},
        "gameLosses": {profile_id: item["losses"] for profile_id, item in profile_metrics.items()},
    }


def run_loss_elimination(
    binary: Path,
    profiles: list[Profile],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    rng = random.Random(args.shuffle_seed)
    active = {profile.id: profile for profile in profiles}
    loss_counts = {profile.id: 0 for profile in profiles}
    rounds: list[dict[str, Any]] = []

    for round_number in range(1, args.max_rounds + 1):
        if len(active) <= 1:
            break
        order = list(active.values())
        rng.shuffle(order)
        duels: list[dict[str, Any]] = []
        if args.dry_run:
            for duel_index in range(0, len(order) - 1, 2):
                champion = order[duel_index]
                opponent = order[duel_index + 1]
                match_id = f"round-{round_number:03d}-duel-{duel_index // 2 + 1:03d}-{champion.id}-vs-{opponent.id}"
                run_pairing(
                    binary,
                    champion,
                    opponent,
                    len(duels),
                    args,
                    output_dir,
                    match_id=match_id,
                    match_seed=args.seed + round_number * 1_000_000 + duel_index * 10_000,
                )
                duels.append({"matchId": match_id, "champion": champion.id, "opponent": opponent.id, "status": "dry-run"})
            rounds.append({"round": round_number, "order": [profile.id for profile in order], "duels": duels})
            break

        for duel_index in range(0, len(order) - 1, 2):
            champion = order[duel_index]
            opponent = order[duel_index + 1]
            match_id = f"round-{round_number:03d}-duel-{duel_index // 2 + 1:03d}-{champion.id}-vs-{opponent.id}"
            result = run_pairing(
                binary,
                champion,
                opponent,
                len(duels),
                args,
                output_dir,
                match_id=match_id,
                match_seed=args.seed + round_number * 1_000_000 + duel_index * 10_000,
            )
            champion_points = result["gamePoints"][champion.id]
            opponent_points = result["gamePoints"][opponent.id]
            matchup_loser = None
            if champion_points < opponent_points:
                matchup_loser = champion.id
            elif opponent_points < champion_points:
                matchup_loser = opponent.id
            if matchup_loser is not None:
                loss_counts[matchup_loser] += 1
            eliminated = []
            if matchup_loser is not None and loss_counts[matchup_loser] >= args.losses_to_eliminate:
                active.pop(matchup_loser, None)
                eliminated.append(matchup_loser)
            duels.append({
                "matchId": match_id,
                "champion": champion.id,
                "opponent": opponent.id,
                "gamePoints": result["gamePoints"],
                "matchupLoser": matchup_loser,
                "lossesAfter": dict(loss_counts),
                "eliminated": eliminated,
                "report": result["report"],
            })
        rounds.append({"round": round_number, "order": [profile.id for profile in order], "duels": duels})

    return {
        "status": "dry-run" if args.dry_run else ("completed" if len(active) == 1 else "stalled"),
        "winner": next(iter(active), None) if len(active) == 1 else None,
        "lossesToEliminate": args.losses_to_eliminate,
        "shuffleSeed": args.shuffle_seed,
        "maxRounds": args.max_rounds,
        "losses": loss_counts,
        "activeProfiles": list(active),
        "rounds": rounds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026083101)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--opening-random-plies", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--round-robin", action="store_true", help="run every unordered profile pair instead of each profile vs control")
    parser.add_argument("--tournament", action="store_true", help="run shuffled matchup-loss elimination instead of a static pairing set")
    parser.add_argument("--losses-to-eliminate", type=int, default=5)
    parser.add_argument("--shuffle-seed", type=int, default=2026083107)
    parser.add_argument("--max-rounds", type=int, default=100)
    parser.add_argument("--group", action="append", default=[], help="restrict the run to one or more catalog groups")
    parser.add_argument("--profile", action="append", default=[], help="restrict the run to one or more profile IDs")
    parser.add_argument("--out-dir", type=Path, default=EXPERIMENT_ROOT / "workspace/arena")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        args.games <= 0
        or args.max_plies <= 0
        or args.workers <= 0
        or args.losses_to_eliminate <= 0
        or args.max_rounds <= 0
    ):
        parser.error("games, max-plies, workers, losses-to-eliminate, and max-rounds must be positive")
    binary = args.binary if args.binary.is_absolute() else REPO_ROOT / args.binary
    profiles_path = args.profiles if args.profiles.is_absolute() else REPO_ROOT / args.profiles
    if not args.dry_run and not binary.is_file():
        parser.error(f"Rust binary does not exist: {binary}; build it first")
    control_id, profiles = load_profiles(profiles_path)
    selected_profiles = [
        profile
        for profile in profiles
        if (not args.group or profile.group in set(args.group))
        and (not args.profile or profile.id in set(args.profile))
    ]
    if args.tournament:
        if len(selected_profiles) < 2:
            parser.error("tournament requires at least two selected profiles")
    else:
        selected = pairings(profiles, control_id, args.round_robin, set(args.group), set(args.profile))
        if not selected:
            parser.error("no pairings selected")
    output_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    tournament = None
    if args.tournament:
        tournament = run_loss_elimination(binary, selected_profiles, args, output_dir)
    else:
        for index, (champion, opponent) in enumerate(selected):
            results.append(run_pairing(binary, champion, opponent, index, args, output_dir))
    campaign = {
        "schema": "pathagon-pathfinder-search-strategy-campaign-v1",
        "profiles": relative(profiles_path),
        "controlProfileId": control_id,
        "mode": "round-robin" if args.round_robin else "vs-control",
        "protocol": {
            "gamesPerPairing": args.games,
            "baseSeed": args.seed,
            "maxPlies": args.max_plies,
            "openingRandomPlies": args.opening_random_plies,
            "workers": args.workers,
        },
        "pairings": results,
    }
    if args.tournament:
        campaign["mode"] = "loss-elimination"
        campaign["tournament"] = tournament
    if not args.dry_run:
        campaign_path = output_dir / "campaign.json"
        campaign_path.write_text(json.dumps(campaign, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"campaign": relative(campaign_path), "pairings": len(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
