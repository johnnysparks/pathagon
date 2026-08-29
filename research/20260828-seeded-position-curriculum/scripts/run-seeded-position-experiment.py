#!/usr/bin/env python3
"""Generate, score, and summarize a seeded-position training batch."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

# Scripts are launched by path, so make the repository package importable
# without requiring callers to set PYTHONPATH explicitly.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.gnn.evaluation import connection_distance
from research.gnn.game import Action, BoardConfig, GameState, Player, bits, count_bits
from research.gnn.pathfinder import PathfinderGuide, action_sort_key
from research.gnn.selfplay import generate_game, game_record
from research.gnn.tactics import immediate_winning_actions
from research.gnn.train import choose_device, load_model, model_state_hash


def _mask(squares: list[int]) -> int:
    return sum(1 << square for square in squares)


def _state_position(state: GameState) -> dict[str, Any]:
    board = [None] * state.config.cell_count
    for square in bits(state.light):
        board[square] = "light"
    for square in bits(state.dark):
        board[square] = "dark"
    return {
        "contractVersion": 1,
        "config": {
            "rulesVersion": "pathagon-rules-v1",
            "boardSize": state.config.size,
            "reservePerPlayer": state.config.reserve_per_player,
            "maxPlies": state.config.max_plies,
            "repetitionLimit": 3,
        },
        "board": board,
        "reserve": {"light": state.reserves[Player.LIGHT], "dark": state.reserves[Player.DARK]},
        "turn": "light" if state.turn is Player.LIGHT else "dark",
        "forbidden": list(bits(state.forbidden)),
        "lastRelocatedTo": {
            "light": state.last_relocated_to[Player.LIGHT],
            "dark": state.last_relocated_to[Player.DARK],
        },
        "winner": None,
        "ply": state.ply,
    }


def _reset_reachable_root(state: GameState) -> GameState:
    """Make a reachable board a fresh episode while preserving move legality."""

    return GameState.seeded(
        state.config,
        state.light,
        state.dark,
        state.reserves,
        state.turn,
        forbidden=state.forbidden,
        last_relocated_to=state.last_relocated_to,
        ply=0,
    )


def _random_reachable_root(config: BoardConfig, rng: random.Random) -> tuple[GameState, list[str]]:
    for _attempt in range(2_000):
        state = GameState.initial(config)
        prefix: list[str] = []
        target_plies = rng.randint(6, 20)
        for _ in range(target_plies):
            if state.winner is not None:
                break
            actions = tuple(state.legal_actions())
            if not actions:
                break
            action = rng.choice(actions)
            prefix.append(action.short())
            state = state.apply_legal(action)
        if state.winner is not None or not state.legal_actions():
            continue
        try:
            return _reset_reachable_root(state), prefix
        except ValueError:
            continue
    raise RuntimeError("could not construct a non-terminal reachable root")


def _random_synthetic_root(
    config: BoardConfig,
    rng: random.Random,
    kind: str,
    seek_near: bool,
) -> GameState:
    if seek_near:
        # Construct a real 1–3-cell path gap first, then fill the opposing
        # material outside that path. This gives the near-terminal stratum a
        # deterministic meaning instead of hoping uniform occupancy lands
        # there by chance.
        for _attempt in range(2_000):
            advantaged = rng.choice((Player.LIGHT, Player.DARK))
            missing = rng.randint(1, 3)
            goal_path = (
                [row * config.size + rng.randrange(config.size) for row in range(config.size)]
                if advantaged is Player.LIGHT
                else [rng.randrange(config.size) * config.size + column for column in range(config.size)]
            )
            missing_indices = set(rng.sample(range(config.size), missing))
            own_squares = [square for index, square in enumerate(goal_path) if index not in missing_indices]
            opponent_count = len(own_squares) if kind == "parity" else rng.randint(1, 4)
            remaining = [square for square in range(config.cell_count) if square not in own_squares]
            rng.shuffle(remaining)
            opponent_squares = remaining[:opponent_count]
            light_squares = own_squares if advantaged is Player.LIGHT else opponent_squares
            dark_squares = opponent_squares if advantaged is Player.LIGHT else own_squares
            try:
                state = GameState.seeded(
                    config,
                    _mask(light_squares),
                    _mask(dark_squares),
                    (config.reserve_per_player - len(light_squares), config.reserve_per_player - len(dark_squares)),
                    rng.choice((Player.LIGHT, Player.DARK)),
                )
            except ValueError:
                # The random opposing material may itself contain a winning
                # path; retry while retaining the deliberately short gap.
                continue
            if min(connection_distance(state, Player.LIGHT), connection_distance(state, Player.DARK)) <= 3:
                return state
    for _attempt in range(20_000):
        if kind == "parity":
            light_count = dark_count = rng.randint(2, 8)
        elif kind == "asymmetric":
            advantaged = rng.choice((Player.LIGHT, Player.DARK))
            high = rng.randint(5, 9)
            low = rng.randint(1, min(4, high - 1))
            light_count, dark_count = (high, low) if advantaged is Player.LIGHT else (low, high)
        else:
            raise ValueError(f"unsupported synthetic root kind: {kind}")
        squares = list(range(config.cell_count))
        rng.shuffle(squares)
        light_squares = squares[:light_count]
        dark_squares = squares[light_count : light_count + dark_count]
        try:
            state = GameState.seeded(
                config,
                _mask(light_squares),
                _mask(dark_squares),
                (config.reserve_per_player - light_count, config.reserve_per_player - dark_count),
                rng.choice((Player.LIGHT, Player.DARK)),
            )
        except ValueError:
            continue
        minimum_distance = min(
            connection_distance(state, Player.LIGHT),
            connection_distance(state, Player.DARK),
        )
        if seek_near and minimum_distance > 3:
            continue
        if not seek_near and minimum_distance < 2:
            continue
        return state
    raise RuntimeError(f"could not construct a {kind} synthetic root")


def make_root(config: BoardConfig, rng: random.Random, kind: str, seek_near: bool) -> tuple[GameState, dict[str, Any]]:
    if kind == "ordinary":
        return GameState.initial(config), {"rootClass": "ordinary", "reachable": True, "prefix": []}
    if kind == "reachable":
        state, prefix = _random_reachable_root(config, rng)
        return state, {"rootClass": "reachable", "reachable": True, "prefix": prefix}
    state = _random_synthetic_root(config, rng, kind, seek_near)
    return state, {"rootClass": kind, "reachable": False, "prefix": []}


def root_class_schedule(rng: random.Random, games: int, seeded_fraction: float) -> list[str]:
    """Build a reproducible, exactly-sized 40:30:30 seeded mixture."""

    seeded = round(games * seeded_fraction)
    reachable = round(seeded * 0.4)
    parity = round(seeded * 0.3)
    asymmetric = seeded - reachable - parity
    schedule = ["ordinary"] * (games - seeded)
    schedule.extend(["reachable"] * reachable)
    schedule.extend(["parity"] * parity)
    schedule.extend(["asymmetric"] * asymmetric)
    rng.shuffle(schedule)
    return schedule


def score_root(model: Any, guide: PathfinderGuide, state: GameState) -> dict[str, Any]:
    actions = tuple(state.legal_actions())
    path_scores = guide.score_actions(state, actions)
    ordered = sorted(
        zip(path_scores, actions),
        key=lambda item: (item[0], -action_sort_key(item[1])),
        reverse=True,
    )
    with torch.no_grad():
        logits, value = model.policy_value(state, list(actions))
        probabilities = torch.softmax(logits, dim=0).detach().cpu().tolist()
    model_order = sorted(range(len(actions)), key=lambda index: (probabilities[index], -action_sort_key(actions[index])), reverse=True)
    distances = {
        "light": connection_distance(state, Player.LIGHT),
        "dark": connection_distance(state, Player.DARK),
    }
    return {
        "pieceCounts": {"light": count_bits(state.light), "dark": count_bits(state.dark)},
        "reserves": {"light": state.reserves[Player.LIGHT], "dark": state.reserves[Player.DARK]},
        "turn": "light" if state.turn is Player.LIGHT else "dark",
        "legalActions": len(actions),
        "connectionDistance": distances,
        "minimumConnectionDistance": min(distances.values()),
        "nearTerminal": min(distances.values()) <= 3,
        "immediateWins": [action.short() for action in immediate_winning_actions(state, state.turn)],
        "pathfinderTop": ordered[0][1].short() if ordered else None,
        "pathfinderTopScore": ordered[0][0] if ordered else None,
        "pathfinderMargin": (ordered[0][0] - ordered[1][0]) if len(ordered) > 1 else None,
        "modelTop": actions[model_order[0]].short() if model_order else None,
        "modelValue": float(value.detach().cpu()),
        "modelPathfinderTopRank": next((rank + 1 for rank, index in enumerate(model_order) if actions[index] == ordered[0][1]), None) if ordered else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--games", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026082800)
    parser.add_argument("--seeded-fraction", type=float, default=0.5)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--temperature-moves", type=int, default=6)
    parser.add_argument("--max-plies", type=int, default=196)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pathfinder-depth", type=int, default=2)
    parser.add_argument("--pathfinder-beam", type=int, default=8)
    parser.add_argument("--pathfinder-nodes", type=int, default=512)
    args = parser.parse_args()
    if args.games < 1 or not 0.0 <= args.seeded_fraction <= 1.0:
        raise SystemExit("games must be positive and seeded-fraction must be in [0, 1]")
    config = BoardConfig(7, 14, args.max_plies)
    device = choose_device(args.device)
    model = load_model(Path(args.checkpoint), device)
    model.eval()
    guide = PathfinderGuide(args.pathfinder_depth, args.pathfinder_beam, args.pathfinder_nodes)
    rng = random.Random(args.seed)
    root_classes = root_class_schedule(rng, args.games, args.seeded_fraction)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    games_path = output_dir / "games.jsonl"
    scores_path = output_dir / "root-scores.jsonl"
    counts: Counter[str] = Counter()
    terminal_buckets: Counter[str] = Counter()
    plies: list[int] = []
    started = time.perf_counter()
    with games_path.open("w", encoding="utf-8") as games_file, scores_path.open("w", encoding="utf-8") as scores_file:
        for index in range(args.games):
            game_seed = args.seed + index
            root_class = root_classes[index]
            seek_near = index % 2 == 0 and root_class in {"parity", "asymmetric"}
            state, provenance = make_root(config, rng, root_class, seek_near)
            score = score_root(model, guide, state)
            provenance.update({
                "rootFamilyId": f"seeded-{args.seed}-{index:05d}",
                "scenarioSeed": game_seed,
                "seekNearTerminal": seek_near,
                "rootPosition": _state_position(state),
            })
            score_record = {"seed": game_seed, **provenance, **score}
            scores_file.write(json.dumps(score_record, sort_keys=True) + "\n")
            examples, final_state = generate_game(
                model,
                config,
                simulations=args.simulations,
                temperature_moves=args.temperature_moves,
                seed=game_seed,
                add_root_noise=True,
                initial_state=state,
            )
            record = game_record(
                examples,
                final_state,
                game_seed,
                simulations=args.simulations,
                model_hash=model_state_hash(model),
                agent_id="seeded-gnn-puct-v0.1.0",
                agent_name="Seeded GNN PUCT",
                initial_state=state,
            )
            record["provenance"] = {key: value for key, value in provenance.items() if key != "rootPosition"}
            record["provenance"]["rootPosition"] = _state_position(state)
            games_file.write(json.dumps(record, sort_keys=True) + "\n")
            counts[root_class] += 1
            terminal_buckets["near" if score["nearTerminal"] else "far"] += 1
            plies.append(len(examples))
            if (index + 1) % max(1, args.games // 20) == 0 or index + 1 == args.games:
                elapsed = time.perf_counter() - started
                print(f"seeded: {index + 1}/{args.games} roots={dict(counts)} average_plies={sum(plies) / len(plies):.1f} elapsed={elapsed:.1f}s", flush=True)
    summary = {
        "schema": "pathagon-seeded-position-experiment-v1",
        "checkpoint": str(args.checkpoint),
        "modelHash": model_state_hash(model),
        "seed": args.seed,
        "games": args.games,
        "seededFraction": args.seeded_fraction,
        "composition": dict(sorted(counts.items())),
        "distanceBuckets": dict(sorted(terminal_buckets.items())),
        "averagePlies": sum(plies) / len(plies) if plies else 0.0,
        "minPlies": min(plies) if plies else 0,
        "maxPlies": max(plies) if plies else 0,
        "simulations": args.simulations,
        "config": {"boardSize": 7, "reservePerPlayer": 14, "maxPlies": args.max_plies},
        "gamesPath": str(games_path),
        "scoresPath": str(scores_path),
        "seconds": time.perf_counter() - started,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
