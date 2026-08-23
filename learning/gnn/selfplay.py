"""Neural-guided self-play data generation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

from .game import Action, BoardConfig, GameState, Player
from .mcts import PUCTSearch
from .model import PathagonGNN


@dataclass(frozen=True)
class SearchExample:
    state: GameState
    actions: Tuple[Action, ...]
    policy: Tuple[float, ...]
    value: float


def generate_game(
    model: PathagonGNN,
    config: BoardConfig,
    simulations: int = 64,
    temperature_moves: int = 8,
    seed: int = 0,
    add_root_noise: bool = True,
) -> Tuple[List[SearchExample], GameState]:
    random.seed(seed)
    search = PUCTSearch(model, simulations=simulations)
    state = GameState.initial(config)
    examples: List[Tuple[GameState, Tuple[Action, ...], Tuple[float, ...]]] = []
    repetitions = {}
    while state.winner is None and state.ply < config.max_plies:
        repetitions[state] = repetitions.get(state, 0) + 1
        if repetitions[state] >= 3:
            break
        actions = list(state.legal_actions())
        if not actions:
            break
        _, search_actions, probabilities = search.run(state, add_root_noise=add_root_noise)
        if tuple(actions) != tuple(search_actions):
            raise AssertionError("MCTS action order diverged from the state action list")
        examples.append((state, tuple(actions), tuple(probabilities)))
        if state.ply < temperature_moves:
            action = random.choices(actions, weights=probabilities, k=1)[0]
        else:
            action = actions[max(range(len(actions)), key=lambda index: (probabilities[index], -actions[index].to))]
        state = state.apply_legal(action)

    final_winner = state.winner
    labeled = []
    for sample_state, sample_actions, sample_policy in examples:
        if final_winner is None:
            value = 0.0
        else:
            value = 1.0 if final_winner is sample_state.turn else -1.0
        labeled.append(SearchExample(sample_state, sample_actions, sample_policy, value))
    return labeled, state

