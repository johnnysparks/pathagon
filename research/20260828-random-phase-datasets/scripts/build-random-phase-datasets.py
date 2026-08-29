#!/usr/bin/env python3
"""Build rule-valid random midgame and late-game Pathagon position sets.

The outputs are research artifacts. They are rooted positions rather than
historically reachable games, except for the optional one-move late-game
replays that intentionally provide a terminal path win to the golden-table
ingester.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
RESEARCH_ROOT = REPO_ROOT / "research/20260824-gnn-cnn-lab"
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from python.contract import (  # noqa: E402
    agent_manifest,
    agent_specification,
    engine_metadata,
    game_config,
    validate_replay_record,
)
from python.game import (  # noqa: E402
    Action,
    BoardConfig,
    GameState,
    Player,
    bits,
    count_bits,
)
from python.tactics import immediate_winning_actions  # noqa: E402


SCHEMA = "pathagon-random-phase-dataset-v1"
RULES_VERSION = "pathagon-rules-v1"
BOARD_SIZE = 7
RESERVE = 14
MAX_PLIES = 196


def mask(squares: Iterable[int]) -> int:
    value = 0
    for square in squares:
        value |= 1 << square
    return value


def action_json(action: Action) -> dict[str, int | str]:
    if action.kind == 0:
        return {"kind": "place", "to": action.to}
    return {"kind": "relocate", "from": action.from_square, "to": action.to}


def action_sort_key(action: Action) -> tuple[int, int, int]:
    return (action.kind, action.from_square, action.to)


def position_json(state: GameState) -> dict[str, Any]:
    board = [None] * state.config.cell_count
    for square in bits(state.light):
        board[square] = "light"
    for square in bits(state.dark):
        board[square] = "dark"
    return {
        "contractVersion": 1,
        "config": {
            "rulesVersion": RULES_VERSION,
            "boardSize": state.config.size,
            "reservePerPlayer": state.config.reserve_per_player,
            "maxPlies": state.config.max_plies,
            "repetitionLimit": 3,
        },
        "board": board,
        "reserve": {
            "light": state.reserves[Player.LIGHT],
            "dark": state.reserves[Player.DARK],
        },
        "turn": state.turn.name.lower(),
        "forbidden": list(bits(state.forbidden)),
        "lastRelocatedTo": {
            "light": state.last_relocated_to[Player.LIGHT],
            "dark": state.last_relocated_to[Player.DARK],
        },
        "winner": None,
        "ply": state.ply,
    }


def make_config(max_plies: int = MAX_PLIES) -> BoardConfig:
    return BoardConfig(BOARD_SIZE, RESERVE, max_plies)


def allocate_missing_inventory(rng: random.Random, none_count: int) -> tuple[int, int]:
    """Split capture-like missing inventory randomly between Light and Dark."""

    if not 0 <= none_count <= 2 * RESERVE:
        raise ValueError(f"none_count must be between 0 and {2 * RESERVE}")
    missing_light = rng.randint(max(0, none_count - RESERVE), min(RESERVE, none_count))
    return missing_light, none_count - missing_light


def shuffled_turns(rng: random.Random, count: int) -> list[Player]:
    turns = [Player.LIGHT if index % 2 == 0 else Player.DARK for index in range(count)]
    rng.shuffle(turns)
    return turns


def build_midgame(
    rng: random.Random,
    config: BoardConfig,
    turn: Player,
    none_count: int = 0,
) -> tuple[GameState, dict[str, Any]]:
    """Build a random non-terminal synthetic root.

    ``none_count`` is an inventory count, not an additional board state. A
    missing piece is represented by one extra reserve for its color; this is
    the only rule-valid way for an empty square to model a captured piece.
    """

    for _attempt in range(20_000):
        missing_light, missing_dark = allocate_missing_inventory(rng, none_count)
        light_count = RESERVE - missing_light
        dark_count = RESERVE - missing_dark
        squares = list(range(config.cell_count))
        rng.shuffle(squares)
        light_squares = squares[:light_count]
        dark_squares = squares[light_count : light_count + dark_count]
        try:
            state = GameState.seeded(
                config,
                mask(light_squares),
                mask(dark_squares),
                (missing_light, missing_dark),
                turn,
            )
        except ValueError:
            continue
        return state, {
            "requestedNoneCount": none_count,
            "missingInventory": {
                "light": missing_light,
                "dark": missing_dark,
            },
            "pieceCounts": {
                "light": count_bits(state.light),
                "dark": count_bits(state.dark),
            },
        }
    raise RuntimeError("could not construct a non-terminal random midgame root")


def random_completed_path(rng: random.Random, config: BoardConfig, player: Player) -> list[int]:
    """Return a connected path with one square on each goal-edge row/column."""

    if player is Player.LIGHT:
        column = rng.randrange(config.size)
        path: list[int] = []
        for row in range(config.size):
            if row:
                column = max(0, min(config.size - 1, column + rng.choice((-1, 0, 1))))
            path.append(row * config.size + column)
        return path

    row = rng.randrange(config.size)
    path = []
    for column in range(config.size):
        if column:
            row = max(0, min(config.size - 1, row + rng.choice((-1, 0, 1))))
        path.append(row * config.size + column)
    return path


def build_lategame(
    rng: random.Random,
    config: BoardConfig,
    target: Player,
    none_count: int = 0,
) -> tuple[GameState, dict[str, Any], tuple[Action, ...]]:
    """Build a non-terminal root with a guaranteed immediate target win."""

    for _attempt in range(50_000):
        missing_light, missing_dark = allocate_missing_inventory(rng, none_count)
        target_missing = missing_light if target is Player.LIGHT else missing_dark
        opponent_missing = missing_dark if target is Player.LIGHT else missing_light
        completed_path = random_completed_path(rng, config, target)
        missing_indices = sorted(rng.sample(range(config.size), 2))
        missing_path = [completed_path[index] for index in missing_indices]
        target_path = [
            square for index, square in enumerate(completed_path) if index not in missing_indices
        ]
        target_board_count = RESERVE - target_missing
        opponent_board_count = RESERVE - opponent_missing
        target_fill_count = target_board_count - len(target_path)
        if target_fill_count < 0:
            continue

        remaining = [square for square in range(config.cell_count) if square not in completed_path]
        rng.shuffle(remaining)
        if target_fill_count + opponent_board_count > len(remaining):
            continue
        target_fill = remaining[:target_fill_count]
        opponent_squares = remaining[target_fill_count : target_fill_count + opponent_board_count]
        target_squares = target_path + target_fill
        light_squares = target_squares if target is Player.LIGHT else opponent_squares
        dark_squares = opponent_squares if target is Player.LIGHT else target_squares
        try:
            state = GameState.seeded(
                config,
                mask(light_squares),
                mask(dark_squares),
                (missing_light, missing_dark),
                target,
            )
        except ValueError:
            continue

        winning_actions = tuple(
            sorted(
                (
                    action
                    for action in immediate_winning_actions(state, target)
                    if action.to in missing_path
                ),
                key=action_sort_key,
            )
        )
        if not winning_actions:
            continue
        return state, {
            "target": target.name.lower(),
            "completedPath": completed_path,
            "missingPath": missing_path,
            "missingPathIndices": missing_indices,
            "requestedNoneCount": none_count,
            "missingInventory": {
                "light": missing_light,
                "dark": missing_dark,
            },
            "pieceCounts": {
                "light": count_bits(state.light),
                "dark": count_bits(state.dark),
            },
            "winningActionCount": len(winning_actions),
        }, winning_actions
    raise RuntimeError("could not construct a late-game root with an immediate win")


def late_replay_record(
    state: GameState,
    action: Action,
    seed: int,
    identifier: str,
) -> dict[str, Any]:
    transition = state.apply_legal(action)
    captured = state.pieces(state.turn.other()) & ~transition.pieces(state.turn.other())
    winner = state.turn.name.lower()
    spec = agent_specification(
        "random-phase-one-ply-oracle-v1",
        "Random phase one-ply oracle",
        "1.0.0",
        "search",
        "python-random-phase",
        manifest=agent_manifest(runtime="python", depth=1, node_budget=0, beam=0),
    )
    record = {
        "contractVersion": 1,
        "seed": seed,
        "config": game_config(
            size=state.config.size,
            reserve=state.config.reserve_per_player,
            max_plies=state.config.max_plies,
        ),
        "engine": engine_metadata("python-random-phase", "python", "1.0.0"),
        "agents": {
            "light": spec["id"],
            "dark": spec["id"],
        },
        "agentSpecifications": {"light": spec, "dark": spec},
        "initialPosition": position_json(state),
        "provenance": {
            "dataset": SCHEMA,
            "sampleId": identifier,
            "oracle": "exhaustive-one-ply-immediate-win",
        },
        "winner": winner,
        "result": "win",
        "reason": "path",
        "plies": 1,
        "moves": [
            {
                "ply": 1,
                "player": winner,
                "action": action_json(action),
                "captured": list(bits(captured)),
                "nodes": 0,
                "completedDepth": 1,
                "tableHits": 0,
            }
        ],
    }
    validate_replay_record(record)
    if transition.winner is not state.turn:
        raise AssertionError("late-game oracle action did not produce the declared winner")
    return record


def mid_sample(sample_id: str, state: GameState, metadata: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "recordType": "sample",
        "id": sample_id,
        "phase": "midgame",
        "seed": seed,
        "position": position_json(state),
        "labels": {
            **metadata,
            "turn": state.turn.name.lower(),
            "legalActionCount": len(state.legal_actions()),
            "guaranteedImmediateWin": False,
        },
        "provenance": {"reachable": False, "generator": SCHEMA},
    }


def late_sample(
    sample_id: str,
    state: GameState,
    metadata: dict[str, Any],
    winning_actions: tuple[Action, ...],
    seed: int,
) -> dict[str, Any]:
    return {
        "recordType": "sample",
        "id": sample_id,
        "phase": "lategame",
        "seed": seed,
        "position": position_json(state),
        "labels": {
            **metadata,
            "turn": state.turn.name.lower(),
            "legalActionCount": len(state.legal_actions()),
            "guaranteedImmediateWin": True,
            "winningActions": [action_json(action) for action in winning_actions],
        },
        "provenance": {
            "reachable": False,
            "generator": SCHEMA,
            "oracle": "exhaustive-one-ply-immediate-win",
        },
    }


def write_jsonl(path: Path, header: dict[str, Any], records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, sort_keys=True) + "\n")
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def build_datasets(
    *,
    output_dir: Path,
    mid_count: int,
    late_count: int,
    seed: int,
    mid_none_count: int,
    late_none_count: int,
    max_plies: int = MAX_PLIES,
) -> dict[str, Any]:
    if mid_count < 0 or late_count < 0 or mid_count + late_count == 0:
        raise ValueError("at least one dataset record is required")
    config = make_config(max_plies)
    rng = random.Random(seed)
    mid_records: list[dict[str, Any]] = []
    late_records: list[dict[str, Any]] = []
    late_replays: list[dict[str, Any]] = []
    mid_turns = shuffled_turns(rng, mid_count)
    late_targets = shuffled_turns(rng, late_count)
    mid_none_by_color: Counter[str] = Counter()
    late_none_by_color: Counter[str] = Counter()
    late_winning_action_counts: list[int] = []

    for index, turn in enumerate(mid_turns):
        record_seed = seed + index
        state, metadata = build_midgame(rng, config, turn, mid_none_count)
        mid_records.append(mid_sample(f"midgame-{index:05d}", state, metadata, record_seed))
        mid_none_by_color.update(
            {color: int(metadata["missingInventory"][color]) for color in ("light", "dark")}
        )

    late_seed_base = seed + mid_count
    for index, target in enumerate(late_targets):
        record_seed = late_seed_base + index
        state, metadata, winning_actions = build_lategame(
            rng, config, target, late_none_count
        )
        identifier = f"lategame-{index:05d}"
        late_records.append(late_sample(identifier, state, metadata, winning_actions, record_seed))
        late_replays.append(late_replay_record(state, winning_actions[0], record_seed, identifier))
        late_winning_action_counts.append(len(winning_actions))
        late_none_by_color.update(
            {color: int(metadata["missingInventory"][color]) for color in ("light", "dark")}
        )

    common_header = {
        "schema": SCHEMA,
        "schemaVersion": 1,
        "seed": seed,
        "config": {
            "rulesVersion": RULES_VERSION,
            "boardSize": config.size,
            "reservePerPlayer": config.reserve_per_player,
            "maxPlies": config.max_plies,
            "repetitionLimit": 3,
        },
    }
    write_jsonl(
        output_dir / "midgame.jsonl",
        {**common_header, "phase": "midgame", "count": mid_count, "noneCount": mid_none_count},
        mid_records,
    )
    write_jsonl(
        output_dir / "lategame.jsonl",
        {**common_header, "phase": "lategame", "count": late_count, "noneCount": late_none_count},
        late_records,
    )
    write_jsonl(
        output_dir / "lategame-replays.jsonl",
        {
            **common_header,
            "phase": "lategame-replay",
            "count": late_count,
            "termination": "one-move-path-win",
        },
        late_replays,
    )

    summary = {
        "schema": "pathagon-random-phase-dataset-report-v1",
        "seed": seed,
        "outputDir": str(output_dir),
        "config": common_header["config"],
        "midgame": {
            "count": mid_count,
            "noneCount": mid_none_count,
            "turnCounts": dict(
                Counter(record["labels"]["turn"] for record in mid_records)
            ),
            "missingInventoryTotals": dict(sorted(mid_none_by_color.items())),
        },
        "lategame": {
            "count": late_count,
            "noneCount": late_none_count,
            "targetCounts": dict(
                Counter(record["labels"]["target"] for record in late_records)
            ),
            "winningActionCount": {
                "min": min(late_winning_action_counts) if late_winning_action_counts else None,
                "max": max(late_winning_action_counts) if late_winning_action_counts else None,
                "average": (
                    sum(late_winning_action_counts) / len(late_winning_action_counts)
                    if late_winning_action_counts
                    else None
                ),
            },
            "missingInventoryTotals": dict(sorted(late_none_by_color.items())),
            "replayCount": len(late_replays),
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mid-count", type=int, default=100)
    parser.add_argument("--late-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026082801)
    parser.add_argument(
        "--mid-none-count",
        type=int,
        default=0,
        help="missing inventory pieces split randomly between colors; 14 models the optional none variant",
    )
    parser.add_argument("--late-none-count", type=int, default=0)
    parser.add_argument("--max-plies", type=int, default=MAX_PLIES)
    args = parser.parse_args()
    if args.mid_count < 0 or args.late_count < 0:
        parser.error("counts cannot be negative")
    if not 0 <= args.mid_none_count <= 2 * RESERVE:
        parser.error(f"mid-none-count must be between 0 and {2 * RESERVE}")
    if not 0 <= args.late_none_count <= 2 * RESERVE:
        parser.error(f"late-none-count must be between 0 and {2 * RESERVE}")
    if not 1 <= args.max_plies <= 4096:
        parser.error("max-plies must be between 1 and 4096")
    summary = build_datasets(
        output_dir=args.output_dir,
        mid_count=args.mid_count,
        late_count=args.late_count,
        seed=args.seed,
        mid_none_count=args.mid_none_count,
        late_none_count=args.late_none_count,
        max_plies=args.max_plies,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
