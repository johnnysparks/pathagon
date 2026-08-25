"""Evaluate a GNN checkpoint against a seeded random baseline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import torch

from .game import Action, BoardConfig, GameState, Player
from .mcts import PUCTSearch
from .selfplay import avoid_repeated_successors, run_match
from .train import choose_device, load_model


def play_against_random(model, config: BoardConfig, gnn_player: Player, simulations: int, seed: int) -> Dict:
    search = PUCTSearch(model, simulations=simulations)
    def choose_action(state: GameState, actions: tuple[Action, ...], rng: random.Random, history: set[tuple]) -> Action | None:
        if state.turn is gnn_player:
            _root, search_actions, probabilities = search.run(
                state,
                add_root_noise=False,
                history=history,
                rng=rng,
            )
            if actions != tuple(search_actions):
                raise AssertionError("evaluation action ordering diverged from PUCT")
            _, probabilities = avoid_repeated_successors(state, actions, probabilities, history)
            action = actions[max(range(len(actions)), key=lambda index: (probabilities[index], -actions[index].to))]
        else:
            action = rng.choice(actions)
        return action

    match = run_match(config, seed, choose_action)
    if match.state.winner is gnn_player:
        outcome = "win"
    elif match.state.winner is None:
        outcome = "draw"
    else:
        outcome = "loss"
    return {"seed": seed, "gnnPlayer": gnn_player.name.lower(), "result": outcome, "plies": match.state.ply}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--size", type=int, default=7)
    parser.add_argument("--reserve", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20265000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = choose_device(args.device)
    model = load_model(Path(args.checkpoint), device)
    model.eval()
    config = BoardConfig(args.size, args.reserve)
    games = [
        play_against_random(model, config, Player.LIGHT if index % 2 == 0 else Player.DARK, args.simulations, args.seed + index)
        for index in range(args.games)
    ]
    counts = {result: sum(game["result"] == result for game in games) for result in ("win", "loss", "draw")}
    print(json.dumps({
        "checkpoint": args.checkpoint,
        "device": str(device),
        "size": args.size,
        "games": args.games,
        "simulations": args.simulations,
        "counts": counts,
        "averagePlies": sum(game["plies"] for game in games) / len(games) if games else 0.0,
        "results": games,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
