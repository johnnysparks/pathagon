#!/usr/bin/env python3
"""Replay and audit the 10k strong-teacher schema-v2 archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_ROOT = REPO_ROOT / "research/20260824-gnn-cnn-lab"
sys.path.insert(0, str(LEGACY_ROOT))

from python.data import initial_state_from_record, iter_records  # noqa: E402
from python.game import BoardConfig, Player, action_from_record, bits  # noqa: E402


TEACHER_ID = "rust-pathfinder-teacher-d5-b256-500k-v1"
OPPONENT_ID = "rust-pathfinder-v0.3.0"


def action_key(action: dict[str, Any]) -> tuple[Any, ...]:
    if action["kind"] == "place":
        return ("place", int(action["to"]))
    return ("relocate", int(action["from"]), int(action["to"]))


def parse_profile(value: str) -> tuple[int, int, int]:
    try:
        depth, beam, nodes = (int(part) for part in value.split(":", 2))
    except ValueError as error:
        raise ValueError(f"invalid opponent profile {value!r}; expected depth:beam:nodes") from error
    if depth < 1 or beam < 1 or nodes < 1:
        raise ValueError(f"invalid opponent profile {value!r}; values must be positive")
    return depth, beam, nodes


def validate_record(
    record: dict[str, Any],
    expected_seed: int,
    max_plies: int,
    opening_plies: int,
    opponent_profiles: set[tuple[int, int, int]],
) -> dict[str, Any]:
    seed = int(record.get("seed", -1))
    if seed != expected_seed:
        raise ValueError(f"seed order mismatch: expected {expected_seed}, found {seed}")
    config_json = record.get("config")
    if not isinstance(config_json, dict):
        raise ValueError(f"seed {seed}: missing config")
    if config_json.get("boardSize") != 7 or config_json.get("reservePerPlayer") != 14:
        raise ValueError(f"seed {seed}: wrong board configuration")
    if config_json.get("maxPlies") != max_plies:
        raise ValueError(f"seed {seed}: wrong max-plies")
    if record.get("contractVersion") != 1 or record.get("engine", {}).get("id") != "rust-bitboard":
        raise ValueError(f"seed {seed}: unsupported replay contract")
    agents = record.get("agents")
    if not isinstance(agents, dict) or set(agents.values()) != {TEACHER_ID, OPPONENT_ID}:
        raise ValueError(f"seed {seed}: teacher provenance is not present for both colors")
    moves = record.get("moves")
    if not isinstance(moves, list) or record.get("plies") != len(moves):
        raise ValueError(f"seed {seed}: invalid move list or ply count")
    state = initial_state_from_record(record, BoardConfig(7, 14, max_plies))
    captures = 0
    teacher_moves = 0
    completed_depths: Counter[str] = Counter()
    nodes = 0
    opening_signature: list[tuple[Any, ...]] = []
    for index, move in enumerate(moves):
        if not isinstance(move, dict):
            raise ValueError(f"seed {seed}: move {index} is not an object")
        if move.get("ply") != state.ply + 1:
            raise ValueError(f"seed {seed}: move {index} has an invalid ply marker")
        expected_player = "light" if state.turn is Player.LIGHT else "dark"
        if move.get("player") != expected_player:
            raise ValueError(f"seed {seed}: move {index} has an invalid player marker")
        action = action_from_record(move.get("action"))
        legal = state.legal_actions()
        if action not in legal:
            raise ValueError(f"seed {seed}: illegal action at ply {state.ply + 1}")
        if index < opening_plies:
            if move.get("nodes") != 1 or move.get("completedDepth") != 0:
                raise ValueError(f"seed {seed}: opening move {index} is not marked as seeded randomness")
            opening_signature.append(action_key(move["action"]))
        else:
            teacher_moves += 1
            if int(move.get("nodes", 0)) <= 0 or int(move.get("completedDepth", 0)) <= 0:
                raise ValueError(f"seed {seed}: post-opening move {index} lacks teacher telemetry")
        transition = state.apply_legal(action)
        # The legacy replay state stores the captured bit mask in `forbidden`
        # for the next position; `last_capture` is only the count.
        expected_captured = sorted(bits(transition.forbidden))
        actual_captured = sorted(int(square) for square in move.get("captured", []))
        if actual_captured != expected_captured:
            raise ValueError(f"seed {seed}: capture mismatch at ply {state.ply + 1}")
        captures += len(actual_captured)
        nodes += int(move.get("nodes", 0))
        completed_depths[str(move.get("completedDepth"))] += 1
        state = transition
    winner = None if state.winner is None else ("light" if state.winner is Player.LIGHT else "dark")
    if record.get("winner") != winner:
        raise ValueError(f"seed {seed}: winner mismatch ({record.get('winner')!r} != {winner!r})")
    if record.get("result") != ("win" if winner is not None else "draw"):
        raise ValueError(f"seed {seed}: result mismatch")
    reason = record.get("reason")
    if winner is not None and reason != "path":
        raise ValueError(f"seed {seed}: winning game has reason {reason!r}")
    if winner is None and reason not in {"max-plies", "threefold-repetition", "no-legal-action"}:
        raise ValueError(f"seed {seed}: unknown draw reason {reason!r}")
    if reason == "max-plies" and len(moves) != max_plies:
        raise ValueError(f"seed {seed}: max-plies game did not reach the cap")
    specifications = record.get("agentSpecifications")
    if not isinstance(specifications, dict):
        raise ValueError(f"seed {seed}: missing agent specifications")
    opponent_profile: tuple[int, int, int] | None = None
    for player in ("light", "dark"):
        manifest = specifications.get(player, {}).get("manifest")
        if not isinstance(manifest, dict):
            raise ValueError(f"seed {seed}: missing {player} agent manifest")
        if agents[player] == TEACHER_ID:
            expected = {"depth": 5, "beam": 256, "nodeBudget": 500_000}
            if any(manifest.get(key) != value for key, value in expected.items()):
                raise ValueError(f"seed {seed}: {player} is not the requested d5/b256/500k teacher")
        else:
            candidate = (int(manifest.get("depth", 0)), int(manifest.get("beam", 0)), int(manifest.get("nodeBudget", 0)))
            if candidate not in opponent_profiles:
                allowed = ", ".join(f"{depth}:{beam}:{nodes}" for depth, beam, nodes in sorted(opponent_profiles))
                raise ValueError(f"seed {seed}: opponent profile {candidate} is not allowed ({allowed})")
            opponent_profile = candidate
    if opponent_profile is None:
        raise ValueError(f"seed {seed}: opponent profile is missing")
    return {
        "seed": seed,
        "plies": len(moves),
        "teacherMoves": teacher_moves,
        "captures": captures,
        "nodes": nodes,
        "winner": winner,
        "reason": reason,
        "opening": opening_signature,
        "completedDepths": completed_depths,
        "opponentProfile": opponent_profile,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-games", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026090100)
    parser.add_argument("--max-plies", type=int, default=20)
    parser.add_argument("--opening-random-plies", type=int, default=4)
    parser.add_argument(
        "--opponent-profile",
        action="append",
        default=None,
        help="allowed opponent depth:beam:nodes profile; repeat for mixed corpora",
    )
    parser.add_argument(
        "--require-opponent-profile",
        action="append",
        default=[],
        help="require at least one game with this depth:beam:nodes profile",
    )
    parser.add_argument("--expected-seeds", type=Path)
    parser.add_argument(
        "--allow-duplicate-games",
        action="store_true",
        help="audit a source pool without failing on duplicates; final mixed archives must omit this",
    )
    args = parser.parse_args()
    records: list[dict[str, Any]] = []
    stats: list[dict[str, Any]] = []
    for record in iter_records(args.input):
        records.append(record)
    if len(records) != args.expected_games:
        raise SystemExit(f"expected {args.expected_games} games, found {len(records)}")
    if args.expected_seeds is not None:
        expected_seeds = json.loads(args.expected_seeds.read_text(encoding="utf-8"))
        if not isinstance(expected_seeds, list) or len(expected_seeds) != len(records):
            raise SystemExit("expected-seeds must be a JSON list with one entry per game")
        expected_seeds = [int(seed) for seed in expected_seeds]
    else:
        expected_seeds = list(range(args.seed, args.seed + args.expected_games))
    opponent_profiles = {
        parse_profile(value)
        for value in (args.opponent_profile or ["5:256:500000"])
    }
    required_profiles = {parse_profile(value) for value in args.require_opponent_profile}
    for index, record in enumerate(records):
        stats.append(
            validate_record(
                record,
                expected_seeds[index],
                args.max_plies,
                args.opening_random_plies,
                opponent_profiles,
            )
        )
    sequence_hashes = [
        hashlib.sha256(json.dumps(item["opening"], separators=(",", ":")).encode("utf-8")).hexdigest()
        for item in stats
    ]
    sequence_ids = {
        json.dumps([move["action"] for move in record["moves"]], separators=(",", ":"), sort_keys=True)
        for record in records
    }
    winners = Counter(item["winner"] or "draw" for item in stats)
    reasons = Counter(item["reason"] for item in stats)
    opponent_profile_counts = Counter(":".join(str(value) for value in item["opponentProfile"]) for item in stats)
    missing_profiles = required_profiles - set(item["opponentProfile"] for item in stats)
    if missing_profiles:
        missing = ", ".join(f"{depth}:{beam}:{nodes}" for depth, beam, nodes in sorted(missing_profiles))
        raise SystemExit(f"required opponent profiles are absent: {missing}")
    duplicate_full_games = len(records) - len(sequence_ids)
    if duplicate_full_games and not args.allow_duplicate_games:
        raise SystemExit(f"duplicate full games detected: {len(records) - len(sequence_ids)} duplicates")
    depth_counts: Counter[str] = Counter()
    for item in stats:
        depth_counts.update(item["completedDepths"])
    result = {
        "schemaVersion": 1,
        "status": "pass",
        "input": str(args.input),
        "inputSha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "games": len(records),
        "uniqueFullGames": len(sequence_ids),
        "duplicateFullGames": duplicate_full_games,
        "uniqueOpeningSequences": len(set(sequence_hashes)),
        "opponentProfiles": dict(sorted(opponent_profile_counts.items())),
        "positions": sum(item["plies"] for item in stats),
        "teacherPositions": sum(item["teacherMoves"] for item in stats),
        "captures": sum(item["captures"] for item in stats),
        "nodes": sum(item["nodes"] for item in stats),
        "winners": dict(sorted(winners.items())),
        "reasons": dict(sorted(reasons.items())),
        "completedDepths": dict(sorted(depth_counts.items(), key=lambda item: int(item[0]))),
        "seedStart": min(expected_seeds),
        "seedEnd": max(expected_seeds),
        "seedOrder": "explicit" if args.expected_seeds is not None else "contiguous",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
