#!/usr/bin/env python3
"""Build a deterministic, solver-labelled tactical fixture suite.

The suite deliberately uses the small 4x4 board where the existing exact
rule solver can label every legal root action. It mixes named tactical cases
with random legal-looking states, records rule context that makes a position
interesting, and emits enough diversity for fixed-budget search ablations.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from research.gnn.game import BoardConfig, GameState, Player, has_winning_path  # noqa: E402
from research.gnn.solver import ExactSolver  # noqa: E402
from research.gnn.tactics import tactical_root  # noqa: E402


def mask(squares: list[int]) -> int:
    return sum(1 << square for square in squares)


def action_json(action) -> dict[str, int | str]:
    if action.kind == 0:
        return {"kind": "place", "to": action.to}
    return {"kind": "relocate", "from": action.from_square, "to": action.to}


def base_fixtures(config: BoardConfig) -> list[tuple[str, GameState]]:
    """Seed the suite with the original immediate/block/fork audit cases."""

    fixtures = [
        (
            "seed-immediate",
            GameState(config, mask([4, 8, 12, 2, 10]), mask([1, 3, 6, 9, 14]), (0, 0), Player.LIGHT, ply=20),
        ),
        (
            "seed-block",
            GameState(config, mask([5, 7, 9, 11, 15]), mask([1, 2, 3, 6, 10]), (0, 0), Player.LIGHT, ply=20),
        ),
        (
            "seed-fork",
            GameState(config, mask([4, 5, 8, 10, 15]), mask([2, 3, 6, 9, 14]), (0, 0), Player.LIGHT, ply=20),
        ),
    ]
    # The fork is intentionally represented with several rule-context
    # variants. These are distinct roots even though the winning geometry is
    # unchanged, and they exercise forbidden-square and relocation-history
    # handling in the solver table.
    fork = fixtures[-1][1]
    empty = [square for square in range(config.cell_count) if not (fork.light | fork.dark) & (1 << square)]
    for forbidden_square in empty:
        if forbidden_square == 12:
            continue
        for previous in (4, 5, 8):
            fixtures.append(
                (
                    f"seed-fork-context-{forbidden_square}-{previous}",
                    GameState(
                        fork.config,
                        fork.light,
                        fork.dark,
                        fork.reserves,
                        fork.turn,
                        forbidden=1 << forbidden_square,
                        last_relocated_to=(previous, None),
                        ply=fork.ply,
                    ),
                )
            )
    return fixtures


def random_state(rng: random.Random, config: BoardConfig) -> GameState | None:
    cells = list(range(config.cell_count))
    rng.shuffle(cells)
    light_count = rng.randint(3, 5)
    dark_count = rng.randint(3, 5)
    light = mask(cells[:light_count])
    dark = mask(cells[light_count : light_count + dark_count])
    probe = GameState(config, light, dark, (0, 0), Player.LIGHT)
    if has_winning_path(probe, Player.LIGHT) or has_winning_path(probe, Player.DARK):
        return None

    turn = Player(rng.randrange(2))
    empty = [square for square in range(config.cell_count) if not (light | dark) & (1 << square)]
    forbidden = 0
    if empty and rng.random() < 0.35:
        forbidden = 1 << rng.choice(empty)

    relocated = [None, None]
    for player, pieces in ((Player.LIGHT, light), (Player.DARK, dark)):
        if pieces and rng.random() < 0.35:
            owned = [square for square in range(config.cell_count) if pieces & (1 << square)]
            relocated[player] = rng.choice(owned)

    return GameState(
        config,
        light,
        dark,
        (0, 0),
        turn,
        forbidden=forbidden,
        last_relocated_to=tuple(relocated),
        ply=rng.randint(12, 42),
    )


def categories(state: GameState) -> list[str]:
    tactical = tactical_root(state)
    labels: list[str] = []
    if tactical.immediate_wins:
        labels.append("immediate-win")
    if tactical.forced_blocks:
        labels.append("forced-defense")
    if tactical.forced_forks:
        labels.append("forced-fork")
    if any(state.apply_legal(action).last_capture > 0 for action in state.legal_actions()):
        labels.append("capture")
    if state.last_relocated_to[state.turn] is not None or state.forbidden:
        labels.append("repetition-avoidance")
    if all(action.kind == 1 for action in state.legal_actions()):
        labels.append("relocation")
    if not any(label in labels for label in ("immediate-win", "forced-defense", "forced-fork", "capture")):
        labels.append("quiet-setup")
    return labels


def encode_fixture(identifier: str, state: GameState, solver: ExactSolver, labels: list[str]) -> dict:
    analysis = solver.analyze(state)
    return {
        "id": identifier,
        "categories": labels,
        "config": {
            "boardSize": state.config.size,
            "reservePerPlayer": state.config.reserve_per_player,
            "maxPlies": state.config.max_plies,
        },
        "state": {
            "light": state.light,
            "dark": state.dark,
            "reserve": list(state.reserves),
            "turn": "light" if state.turn is Player.LIGHT else "dark",
            "forbidden": state.forbidden,
            "lastRelocatedTo": list(state.last_relocated_to),
            "ply": state.ply,
        },
        "labels": {
            "outcome": analysis.result.outcome,
            "optimalActions": [action_json(action) for action in analysis.optimal_actions],
            "actionCount": len(analysis.actions),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--min-per-category", type=int, default=20)
    args = parser.parse_args()
    if args.count < 1 or args.horizon < 1 or args.min_per_category < 0:
        parser.error("count, horizon, and min-per-category must be positive or zero as appropriate")

    config = BoardConfig(4, 5, 64)
    solver = ExactSolver(max_size=4, horizon=args.horizon)
    rng = random.Random(args.seed)
    fixtures: list[dict] = []
    seen: set[tuple] = set()

    for identifier, state in base_fixtures(config):
        key = (state.light, state.dark, state.reserves, int(state.turn), state.forbidden, state.last_relocated_to)
        seen.add(key)
        fixtures.append(encode_fixture(identifier, state, solver, categories(state)))

    attempts = 0
    max_attempts = max(args.count * 200, 10_000)
    while len(fixtures) < args.count and attempts < max_attempts:
        attempts += 1
        state = random_state(rng, config)
        if state is None:
            continue
        key = (state.light, state.dark, state.reserves, int(state.turn), state.forbidden, state.last_relocated_to)
        if key in seen:
            continue
        labels = categories(state)
        seen.add(key)
        fixtures.append(encode_fixture(f"random-{len(fixtures):04d}", state, solver, labels))

    counts = Counter(label for fixture in fixtures for label in fixture["categories"])
    required = ("immediate-win", "forced-defense", "forced-fork", "capture", "repetition-avoidance", "relocation", "quiet-setup")
    missing = [label for label in required if counts[label] < args.min_per_category]
    if len(fixtures) < args.count or missing:
        raise SystemExit(
            f"could not build suite: {len(fixtures)}/{args.count} fixtures; "
            f"missing category quotas: {', '.join(missing) or 'none'}"
        )

    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"schema": "pathagon-tactical-suite-v1", "seed": args.seed, "horizon": args.horizon, "count": len(fixtures), "categoryCounts": dict(sorted(counts.items()))}, sort_keys=True) + "\n")
        for fixture in fixtures:
            handle.write(json.dumps(fixture, sort_keys=True) + "\n")

    print(json.dumps({"schema": "pathagon-tactical-suite-report-v1", "output": str(output), "count": len(fixtures), "categoryCounts": dict(sorted(counts.items())), "solverNodes": solver.stats.nodes, "solverCacheHits": solver.stats.cache_hits}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
