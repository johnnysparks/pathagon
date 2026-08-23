"""Evaluate a GNN checkpoint against a seeded random baseline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import torch

from .game import BoardConfig, GameState, Player, repetition_key
from .mcts import PUCTSearch
from .selfplay import avoid_repeated_successors
from .train import choose_device, load_model


def play_against_random(model, config: BoardConfig, gnn_player: Player, simulations: int, seed: int) -> Dict:
    rng = random.Random(seed)
    search = PUCTSearch(model, simulations=simulations)
    state = GameState.initial(config)
    repetitions = {}
    while state.winner is None and state.ply < config.max_plies:
        position = repetition_key(state)
        repetitions[position] = repetitions.get(position, 0) + 1
        if repetitions[position] >= 3:
            break
        actions = list(state.legal_actions())
        if not actions:
            break
        if state.turn is gnn_player:
            root, search_actions, probabilities = search.run(
                state,
                add_root_noise=False,
                history=set(repetitions),
            )
            if actions != search_actions:
                raise AssertionError("evaluation action ordering diverged from PUCT")
            _, probabilities = avoid_repeated_successors(state, actions, probabilities, set(repetitions))
            action = actions[max(range(len(actions)), key=lambda index: (probabilities[index], -actions[index].to))]
        else:
            action = rng.choice(actions)
        state = state.apply_legal(action)
    if state.winner is gnn_player:
        result = "win"
    elif state.winner is None:
        result = "draw"
    else:
        result = "loss"
    return {"seed": seed, "gnnPlayer": gnn_player.name.lower(), "result": result, "plies": state.ply}


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
