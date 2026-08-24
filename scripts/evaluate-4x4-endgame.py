#!/usr/bin/env python3
"""Evaluate the GNN search on exact 4x4 five-piece tactical positions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from learning.gnn.game import BoardConfig, GameState, Player  # noqa: E402
from learning.gnn.mcts import PUCTSearch  # noqa: E402
from learning.gnn.solver import ExactSolver  # noqa: E402
from learning.gnn.tactics import tactical_root  # noqa: E402
from learning.gnn.train import load_model  # noqa: E402


DEFAULT_CHECKPOINT = REPO_ROOT / "training/gnn/benchmark-7x7/generated/batch-20260824-neural-reval-20260824/reval-gnn-30k.pt"


def parse_budgets(value: str) -> tuple[int, ...]:
    budgets = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not budgets or any(budget < 0 for budget in budgets):
        raise argparse.ArgumentTypeError("budgets must be a non-empty comma-separated list of non-negative integers")
    return budgets


def mask(squares: tuple[int, ...]) -> int:
    return sum(1 << square for square in squares)


def map_mask(value: int, transform) -> int:
    result = 0
    for square in range(16):
        if value & (1 << square):
            row, column = divmod(square, 4)
            new_row, new_column = transform(row, column)
            result |= 1 << (new_row * 4 + new_column)
    return result


def transform_state(state: GameState, transform, swaps_players: bool) -> GameState:
    light = map_mask(state.light, transform)
    dark = map_mask(state.dark, transform)
    turn = state.turn
    if swaps_players:
        light, dark = dark, light
        turn = turn.other()
    return replace(state, light=light, dark=dark, turn=turn, winner=None, ply=20)


def positions() -> list[tuple[str, str, GameState]]:
    config = BoardConfig(4, 5, 64)
    bases = {
        "immediate": GameState(config, mask((4, 8, 12, 2, 10)), mask((1, 3, 6, 9, 14)), (0, 0), Player.LIGHT, ply=20),
        "block": GameState(config, mask((5, 7, 9, 11, 15)), mask((1, 2, 3, 6, 10)), (0, 0), Player.LIGHT, ply=20),
        "fork": GameState(config, mask((4, 5, 8, 10, 15)), mask((2, 3, 6, 9, 14)), (0, 0), Player.LIGHT, ply=20),
    }
    transforms = (
        ("identity", lambda row, column: (row, column), False),
        ("mirror-cols", lambda row, column: (row, 3 - column), False),
        ("mirror-rows", lambda row, column: (3 - row, column), False),
        ("rotate-180", lambda row, column: (3 - row, 3 - column), False),
        ("transpose+swap", lambda row, column: (column, row), True),
        ("transpose+swap+mirror-cols", lambda row, column: (column, 3 - row), True),
        ("transpose+swap+mirror-rows", lambda row, column: (3 - column, row), True),
        ("transpose+swap+rotate-180", lambda row, column: (3 - column, 3 - row), True),
    )
    result = []
    seen = set()
    for category, base in bases.items():
        for name, transform, swaps_players in transforms:
            state = transform_state(base, transform, swaps_players)
            identity = (state.light, state.dark, state.turn)
            if identity in seen:
                continue
            tactical = tactical_root(state)
            field = {"immediate": "immediate_wins", "block": "forced_blocks", "fork": "forced_forks"}[category]
            if not getattr(tactical, field):
                raise RuntimeError(f"fixture lost its {category} tactic under {name}")
            seen.add(identity)
            result.append((category, name, state))
    return result


def action_key(action) -> int:
    return action.to if action.kind == 0 else action.from_square * 100 + action.to


def choose_policy(actions, probabilities):
    return max(range(len(actions)), key=lambda index: (probabilities[index], -action_key(actions[index])))


def choose_visits(root, actions):
    return max(actions, key=lambda action: (root.children[action].visit_count, -action_key(action)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--budgets", type=parse_budgets, default=(0, 4, 8, 16, 32, 64, 128))
    parser.add_argument(
        "--solver-horizon",
        type=int,
        default=3,
        help="number of plies for the exact solver labels (default: 3)",
    )
    args = parser.parse_args()
    if args.solver_horizon < 1:
        parser.error("--solver-horizon must be positive")

    model = load_model(args.checkpoint.resolve(), torch.device("cpu"))
    model.eval()
    torch.set_num_threads(1)
    test_positions = positions()
    solver = ExactSolver(max_size=4, horizon=args.solver_horizon)
    counts = defaultdict(lambda: defaultdict(lambda: {"policyCorrect": 0, "visitCorrect": 0, "total": 0}))
    tree = defaultdict(
        lambda: {
            "positions": 0,
            "rootActions": set(),
            "replyEdges": set(),
            "correctMoves": [],
            "solverOutcomes": set(),
            "guardOptimalPositions": 0,
        }
    )

    for category, _name, state in test_positions:
        analysis = solver.analyze(state)
        correct = set(analysis.optimal_actions)
        if not correct:
            raise RuntimeError(f"solver produced no optimal action for {category}")
        tactical = tactical_root(state)
        guard_actions = set(tactical.priority_actions)
        category_tree = tree[category]
        category_tree["positions"] += 1
        category_tree["rootActions"].add(tactical.root_action_count)
        category_tree["replyEdges"].add(tactical.root_reply_edges)
        category_tree["correctMoves"].append(len(correct))
        category_tree["solverOutcomes"].add(analysis.result.outcome)
        category_tree["guardOptimalPositions"] += int(bool(guard_actions) and guard_actions <= correct)
        for guarded in (False, True):
            mode = "guarded" if guarded else "unguarded"
            for budget in args.budgets:
                search = PUCTSearch(model, simulations=budget, tactical_guard=guarded)
                root, actions, probabilities = search.run(state, add_root_noise=False, history=set())
                policy_action = actions[choose_policy(actions, probabilities)]
                visit_action = choose_visits(root, actions)
                row = counts[(mode, category)][budget]
                row["policyCorrect"] += int(policy_action in correct)
                row["visitCorrect"] += int(visit_action in correct)
                row["total"] += 1

    output = {
        "checkpoint": str(args.checkpoint),
        "positions": len(test_positions),
        "solver": {
            "horizon": args.solver_horizon,
            "outcomePerspective": "side-to-move",
            "nodes": solver.stats.nodes,
            "cacheHits": solver.stats.cache_hits,
            "tableEntries": solver.stats.table_entries,
        },
        "tree": {
            category: {
                "positions": value["positions"],
                "rootActions": sorted(value["rootActions"]),
                "replyEdges": sorted(value["replyEdges"]),
                "meanCorrectMoves": sum(value["correctMoves"]) / len(value["correctMoves"]),
                "solverOutcomes": sorted(value["solverOutcomes"]),
                "guardOptimalPositions": value["guardOptimalPositions"],
            }
            for category, value in tree.items()
        },
        "results": {
            f"{mode}:{category}": {
                str(budget): {
                    "policyAccuracy": row["policyCorrect"] / row["total"],
                    "visitAccuracy": row["visitCorrect"] / row["total"],
                    "policyCorrect": row["policyCorrect"],
                    "visitCorrect": row["visitCorrect"],
                    "total": row["total"],
                }
                for budget, row in sorted(budget_rows.items())
            }
            for (mode, category), budget_rows in sorted(counts.items())
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
