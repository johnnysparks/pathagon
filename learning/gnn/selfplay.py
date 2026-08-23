"""Neural-guided self-play data generation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

from .game import Action, BoardConfig, GameState, Player, bits, repetition_key
from .mcts import PUCTSearch
from .model import PathagonGNN


@dataclass(frozen=True)
class SearchExample:
    state: GameState
    actions: Tuple[Action, ...]
    policy: Tuple[float, ...]
    selected_action: Action
    value: float


def avoid_repeated_successors(
    state: GameState,
    actions: Iterable[Action],
    probabilities: Iterable[float],
    history: Set[tuple],
) -> Tuple[Tuple[Action, ...], Tuple[float, ...]]:
    """Prefer actions that do not revisit a previously seen position.

    This is a search-policy guard, not a rules change. If every legal move
    revisits history, the original policy is preserved and the game can reach
    the engine's threefold draw condition.
    """

    action_list = tuple(actions)
    probability_list = tuple(float(probability) for probability in probabilities)
    safe = tuple(
        index
        for index, action in enumerate(action_list)
        if repetition_key(state.apply_legal(action)) not in history
    )
    if not safe or len(safe) == len(action_list):
        return action_list, probability_list

    total = sum(probability_list[index] for index in safe)
    if total <= 0:
        safe_probability = 1.0 / len(safe)
        filtered = tuple(safe_probability if index in safe else 0.0 for index in range(len(action_list)))
    else:
        safe_set = set(safe)
        filtered = tuple(
            probability_list[index] / total if index in safe_set else 0.0
            for index in range(len(action_list))
        )
    return action_list, filtered


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
    examples: List[Tuple[GameState, Tuple[Action, ...], Tuple[float, ...], Action]] = []
    repetitions: Dict[tuple, int] = {}
    while state.winner is None and state.ply < config.max_plies:
        position = repetition_key(state)
        repetitions[position] = repetitions.get(position, 0) + 1
        if repetitions[position] >= 3:
            break
        actions = list(state.legal_actions())
        if not actions:
            break
        _, search_actions, probabilities = search.run(
            state,
            add_root_noise=add_root_noise,
            history=set(repetitions),
        )
        if tuple(actions) != tuple(search_actions):
            raise AssertionError("MCTS action order diverged from the state action list")
        _, probabilities = avoid_repeated_successors(state, actions, probabilities, set(repetitions))
        if state.ply < temperature_moves:
            action = random.choices(actions, weights=probabilities, k=1)[0]
        else:
            action = actions[max(range(len(actions)), key=lambda index: (probabilities[index], -actions[index].to))]
        examples.append((state, tuple(actions), tuple(probabilities), action))
        state = state.apply_legal(action)

    final_winner = state.winner
    labeled = []
    for sample_state, sample_actions, sample_policy, selected_action in examples:
        if final_winner is None:
            value = 0.0
        else:
            value = 1.0 if final_winner is sample_state.turn else -1.0
        labeled.append(SearchExample(sample_state, sample_actions, sample_policy, selected_action, value))
    return labeled, state


def game_record(examples: List[SearchExample], final_state: GameState, seed: int) -> dict:
    """Serialize a neural game in the archive's schema-v2 shape."""

    moves = []
    for example in examples:
        transition = example.state.apply_legal(example.selected_action)
        action = example.selected_action
        action_json = {"kind": "place", "to": action.to} if action.kind == 0 else {
            "kind": "relocate",
            "from": action.from_square,
            "to": action.to,
        }
        moves.append({
            "ply": example.state.ply + 1,
            "player": "light" if example.state.turn is Player.LIGHT else "dark",
            "action": action_json,
            "captured": list(bits(transition.forbidden)),
            "nodes": 0,
            "completedDepth": 0,
            "tableHits": 0,
            "score": 0,
            "bookHit": False,
        })
    winner = None if final_state.winner is None else ("light" if final_state.winner is Player.LIGHT else "dark")
    if winner is not None:
        reason = "path"
    elif not final_state.legal_actions():
        reason = "no-legal-action"
    elif final_state.ply >= final_state.config.max_plies:
        reason = "max-plies"
    else:
        reason = "threefold-repetition"
    return {
        "schemaVersion": 2,
        "seed": seed,
        "agents": {"light": "python-gnn-puct-v0.1.0", "dark": "python-gnn-puct-v0.1.0"},
        "winner": winner,
        "result": "win" if winner is not None else "draw",
        "reason": reason,
        "plies": len(moves),
        "moves": moves,
    }
