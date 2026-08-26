"""Neural-guided self-play data generation."""

from __future__ import annotations

import random
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from .game import Action, BoardConfig, GameState, Player, bits, repetition_key
from .contract import ROOT_Q_SOURCE, agent_manifest, agent_specification, engine_metadata, game_config
from .mcts import PUCTSearch
from .model import PathagonGNN
from .pathfinder import PathfinderGuide


@dataclass(frozen=True)
class SearchExample:
    state: GameState
    actions: Tuple[Action, ...]
    policy: Tuple[float, ...]
    selected_action: Action
    value: float
    action_values: Tuple[float, ...] = ()
    action_visits: Tuple[int, ...] = ()


@dataclass(frozen=True)
class MatchResult:
    state: GameState
    reason: str


ActionChooser = Callable[[GameState, Tuple[Action, ...], random.Random, Set[tuple]], Optional[Action]]
MoveObserver = Callable[[GameState, Action, GameState], None]


def run_match(
    config: BoardConfig,
    seed: int,
    choose_action: ActionChooser,
    observe_move: Optional[MoveObserver] = None,
    progress: Optional[Callable[[GameState], None]] = None,
) -> MatchResult:
    """Run one deterministic match with shared termination and legality rules."""

    rng = random.Random(seed)
    state = GameState.initial(config)
    repetitions: Dict[tuple, int] = {}
    while state.winner is None and state.ply < config.max_plies:
        position = repetition_key(state)
        repetitions[position] = repetitions.get(position, 0) + 1
        if repetitions[position] >= 3:
            return MatchResult(state, "threefold-repetition")
        actions = tuple(state.legal_actions())
        if not actions:
            return MatchResult(state, "no-legal-action")
        action = choose_action(state, actions, rng, set(repetitions))
        if action is None or action not in actions:
            return MatchResult(state, "no-legal-action")
        next_state = state.apply_legal(action)
        if observe_move is not None:
            observe_move(state, action, next_state)
        state = next_state
        if progress is not None:
            progress(state)
    if state.winner is not None:
        return MatchResult(state, "path")
    return MatchResult(state, "max-plies")


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


def _softmax_scores(scores: Iterable[float], temperature: float) -> Tuple[float, ...]:
    values = tuple(float(score) for score in scores)
    if not values:
        return ()
    if temperature <= 0:
        best = max(range(len(values)), key=lambda index: (values[index], -index))
        return tuple(1.0 if index == best else 0.0 for index in range(len(values)))
    scale = max(1.0, 3_500.0 * temperature)
    maximum = max(values)
    weights = tuple(math.exp((value - maximum) / scale) for value in values)
    total = sum(weights)
    return tuple(weight / total for weight in weights)


def _blend_probabilities(
    base: Iterable[float],
    guide: Iterable[float],
    weight: float,
) -> Tuple[float, ...]:
    base_values = tuple(float(value) for value in base)
    guide_values = tuple(float(value) for value in guide)
    if len(base_values) != len(guide_values):
        raise ValueError("base and guidance policies must have the same length")
    blended = tuple((1.0 - weight) * base_value + weight * guide_value for base_value, guide_value in zip(base_values, guide_values))
    total = sum(blended)
    if total <= 0:
        return tuple(1.0 / len(blended) for _ in blended) if blended else ()
    return tuple(value / total for value in blended)


def _mix_uniform(probabilities: Iterable[float], weight: float) -> Tuple[float, ...]:
    values = tuple(float(value) for value in probabilities)
    if not values:
        return ()
    uniform = 1.0 / len(values)
    return tuple((1.0 - weight) * value + weight * uniform for value in values)


def _is_tactical_state(state: GameState, capture_threshold: int) -> bool:
    for action in state.legal_actions():
        afterstate = state.apply_legal(action)
        if afterstate.winner is state.turn or afterstate.last_capture >= capture_threshold:
            return True
    return False


def generate_game(
    model: PathagonGNN,
    config: BoardConfig,
    simulations: int = 64,
    temperature_moves: int = 8,
    seed: int = 0,
    add_root_noise: bool = True,
    progress: Optional[Callable[[GameState], None]] = None,
    policy_temperature: float = 1.0,
    opening_moves: int = 0,
    opening_temperature: float = 1.0,
    opening_randomness: float = 0.0,
    pathfinder_guidance: float = 0.0,
    placement_guidance: Optional[float] = None,
    pathfinder_temperature: float = 1.0,
    pathfinder_depth: int = 2,
    pathfinder_beam: int = 8,
    pathfinder_nodes: int = 1_000,
    tactical_simulations: int = 0,
    tactical_capture_threshold: int = 1,
) -> Tuple[List[SearchExample], GameState]:
    for name, value in (
        ("policy_temperature", policy_temperature),
        ("opening_temperature", opening_temperature),
        ("pathfinder_temperature", pathfinder_temperature),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    for name, value in (
        ("opening_randomness", opening_randomness),
        ("pathfinder_guidance", pathfinder_guidance),
        ("placement_guidance", pathfinder_guidance if placement_guidance is None else placement_guidance),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if opening_moves < 0 or tactical_simulations < 0 or tactical_capture_threshold < 1:
        raise ValueError("opening moves, tactical simulations, and capture threshold must be valid")
    placement_weight = pathfinder_guidance if placement_guidance is None else placement_guidance
    guide = PathfinderGuide(pathfinder_depth, pathfinder_beam, pathfinder_nodes) if max(pathfinder_guidance, placement_weight) > 0 else None
    examples: List[Tuple[GameState, Tuple[Action, ...], Tuple[float, ...], Action, Tuple[float, ...], Tuple[int, ...]]] = []

    def choose_action(state: GameState, actions: Tuple[Action, ...], rng: random.Random, history: Set[tuple]) -> Action:
        tactical = tactical_simulations > simulations and _is_tactical_state(state, tactical_capture_threshold)
        search = PUCTSearch(model, simulations=tactical_simulations if tactical else simulations)
        in_opening = state.ply < opening_moves
        effective_temperature = opening_temperature if in_opening else policy_temperature
        root, search_actions, probabilities = search.run(
            state,
            add_root_noise=add_root_noise,
            history=history,
            rng=rng,
            policy_temperature=effective_temperature,
        )
        if actions != tuple(search_actions):
            raise AssertionError("MCTS action order diverged from the state action list")
        action_values, action_visits = search.root_action_values(root, list(actions))
        _, probabilities = avoid_repeated_successors(state, actions, probabilities, history)
        if guide is not None:
            guidance_weight = placement_weight if state.reserves[state.turn] > 0 else pathfinder_guidance
            if guidance_weight > 0:
                path_scores = guide.score_actions(state, actions)
                probabilities = _blend_probabilities(
                    probabilities,
                    _softmax_scores(path_scores, pathfinder_temperature),
                    guidance_weight,
                )
        if in_opening and opening_randomness > 0:
            probabilities = _mix_uniform(probabilities, opening_randomness)
        # Blending can reintroduce probability on a repeated successor, so
        # apply the repetition guard again after all exploration dials.
        _, probabilities = avoid_repeated_successors(state, actions, probabilities, history)
        if state.ply < temperature_moves:
            action = rng.choices(actions, weights=probabilities, k=1)[0]
        else:
            action = actions[max(range(len(actions)), key=lambda index: (probabilities[index], -actions[index].to))]
        examples.append((state, tuple(actions), tuple(probabilities), action, tuple(action_values), tuple(action_visits)))
        return action

    result = run_match(config, seed, choose_action, progress=progress)
    final_winner = result.state.winner
    labeled = []
    for sample_state, sample_actions, sample_policy, selected_action, action_values, action_visits in examples:
        if final_winner is None:
            value = 0.0
        else:
            value = 1.0 if final_winner is sample_state.turn else -1.0
        labeled.append(SearchExample(sample_state, sample_actions, sample_policy, selected_action, value, action_values, action_visits))
    return labeled, result.state


def game_record(
    examples: List[SearchExample],
    final_state: GameState,
    seed: int,
    simulations: int = 0,
    model_hash: str | None = None,
    agent_id: str = "python-gnn-puct-v0.1.0",
    agent_name: str = "Python GNN PUCT",
    agent_version: str = "0.1.0",
    agent_kind: str = "puct",
    engine_id: str = "python-gnn",
) -> dict:
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
            "policy": list(example.policy),
            "captured": list(bits(transition.forbidden)),
            "nodes": 0,
            "completedDepth": 0,
            "tableHits": 0,
            "score": 0,
            "bookHit": False,
        })
        if bool(example.action_values) != bool(example.action_visits):
            raise ValueError("action values and visits must be provided together")
        if example.action_values:
            if len(example.action_values) != len(example.actions) or len(example.action_visits) != len(example.actions):
                raise ValueError("action values and visits must align with legal actions")
            moves[-1]["actionValues"] = list(example.action_values)
            moves[-1]["actionVisits"] = list(example.action_visits)
            moves[-1]["actionValueSource"] = ROOT_Q_SOURCE
    winner = None if final_state.winner is None else ("light" if final_state.winner is Player.LIGHT else "dark")
    if winner is not None:
        reason = "path"
    elif not final_state.legal_actions():
        reason = "no-legal-action"
    elif final_state.ply >= final_state.config.max_plies:
        reason = "max-plies"
    else:
        reason = "threefold-repetition"
    manifest = agent_manifest(runtime="python", node_budget=simulations, model_hash=model_hash)
    specification = agent_specification(
        agent_id,
        agent_name,
        agent_version,
        agent_kind,
        engine_id,
        manifest=manifest,
    )
    return {
        "contractVersion": 1,
        "seed": seed,
        "config": game_config(final_state.config.size, final_state.config.reserve_per_player, final_state.config.max_plies),
        "engine": engine_metadata(engine_id, "python"),
        "agents": {"light": agent_id, "dark": agent_id},
        "agentSpecifications": {"light": specification, "dark": specification},
        "winner": winner,
        "result": "win" if winner is not None else "draw",
        "reason": reason,
        "plies": len(moves),
        "moves": moves,
    }
