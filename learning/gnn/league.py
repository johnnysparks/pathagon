"""Run checkpoint and heuristic agents in a color-balanced Elo league."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import torch

from .game import Action, BoardConfig, GameState, Player, repetition_key
from .contract import agent_manifest, agent_specification, engine_metadata, game_config
from .evaluation import connection_distance, evaluate_position, normalize_heuristic, squares_from_mask
from .mcts import PUCTSearch
from .selfplay import avoid_repeated_successors, run_match
from .train import choose_device, load_model


@dataclass(frozen=True)
class AgentSpec:
    id: str
    label: str
    kind: str
    choose: object
    manifest: dict


class RandomAgent:
    def choose_action(self, state: GameState, rng: random.Random, _history: Set[tuple]) -> Action | None:
        actions = list(state.legal_actions())
        return rng.choice(actions) if actions else None


class LunaticAgent:
    """One-ply local-pattern baseline matching the browser Lunatic opponent."""

    def choose_action(self, state: GameState, _rng: random.Random, _history: Set[tuple]) -> Action | None:
        actions = list(state.legal_actions())
        if not actions:
            return None
        player = state.turn
        before_own_distance = connection_distance(state, player)
        before_opponent_distance = connection_distance(state, player.other())
        best_action = actions[0]
        best_score = float("-inf")
        for action in actions:
            next_state = state.apply_legal(action)
            captured = next_state.last_capture
            if next_state.winner is player:
                score = 1_000_000_000
            else:
                own_distance = connection_distance(next_state, player)
                opponent_distance = connection_distance(next_state, player.other())
                score = (
                    captured * 10_000
                    + (before_own_distance - own_distance) * 500
                    + (opponent_distance - before_opponent_distance) * 350
                    + (10 if action.kind == 1 else 0)
                )
            if score > best_score or (score == best_score and action_sort_key(action) < action_sort_key(best_action)):
                best_action = action
                best_score = score
        return best_action


class HeuristicAgent:
    def __init__(self, depth: int, beam_width: int, max_nodes: int) -> None:
        self.depth = depth
        self.beam_width = beam_width
        self.max_nodes = max_nodes
        self.nodes = 0

    def choose_action(self, state: GameState, _rng: random.Random, _history: Set[tuple]) -> Action | None:
        actions = self._ordered_actions(state, state.turn)[: self.beam_width]
        if not actions:
            return None
        self.nodes = 0
        best_action = actions[0]
        best_score = float("-inf")
        alpha = float("-inf")
        for action in actions:
            if self.nodes >= self.max_nodes:
                break
            self.nodes += 1
            score = self._search(state.apply_legal(action), state.turn, self.depth - 1, alpha, float("inf"))
            if score > best_score or (score == best_score and action < best_action):
                best_action = action
                best_score = score
            alpha = max(alpha, best_score)
        return best_action

    def _search(self, state: GameState, root: Player, depth: int, alpha: float, beta: float) -> float:
        if state.winner is not None or depth <= 0 or self.nodes >= self.max_nodes:
            return evaluate_position(state, root)
        actions = self._ordered_actions(state, root)[: self.beam_width]
        if not actions:
            return evaluate_position(state, root)
        maximizing = state.turn is root
        best = float("-inf") if maximizing else float("inf")
        for action in actions:
            if self.nodes >= self.max_nodes:
                break
            self.nodes += 1
            score = self._search(state.apply_legal(action), root, depth - 1, alpha, beta)
            if maximizing:
                best = max(best, score)
                alpha = max(alpha, best)
            else:
                best = min(best, score)
                beta = min(beta, best)
            if beta <= alpha:
                break
        return best

    def _ordered_actions(self, state: GameState, root: Player) -> List[Action]:
        scored = []
        for action in state.legal_actions():
            next_state = state.apply_legal(action)
            tactical = 2_000_000_000 if next_state.winner is state.turn else next_state.last_capture * 10_000
            scored.append((tactical + evaluate_position(next_state, root), action))
        maximizing = state.turn is root
        scored.sort(key=lambda item: (item[0], action_sort_key(item[1])), reverse=maximizing)
        return [action for _, action in scored]


class GNNAgent:
    def __init__(self, model: torch.nn.Module, simulations: int) -> None:
        self.search = PUCTSearch(model, simulations=simulations)

    def choose_action(self, state: GameState, _rng: random.Random, history: Set[tuple]) -> Action | None:
        _, actions, probabilities = self.search.run(state, add_root_noise=False, history=history)
        if not actions:
            return None
        _, filtered = avoid_repeated_successors(state, actions, probabilities, history)
        return actions[max(range(len(actions)), key=lambda index: (filtered[index], -action_sort_key(actions[index])))]


class PolicyBeamAgent:
    """Iterative beam search with a learned policy and value at each node.

    This is a breadth-limited search rather than a full minimax tree. Keeping
    only the best frontier states makes the Scout variants useful for bulk
    cross-play while retaining an explicit per-move expansion budget.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        depth: int,
        beam_width: int,
        max_nodes: int,
        heuristic_blend: float = 0.0,
    ) -> None:
        self.model = model
        self.depth = depth
        self.beam_width = beam_width
        self.max_nodes = max_nodes
        self.heuristic_blend = heuristic_blend
        self.nodes = 0
        self.completed_depth = 0

    def choose_action(self, state: GameState, _rng: random.Random, history: Set[tuple]) -> Action | None:
        actions = tuple(state.legal_actions())
        if not actions:
            return None
        self.nodes = 0
        self.completed_depth = 0
        best_action = actions[0]
        previous_positions = set(history)
        for depth in range(1, self.depth + 1):
            try:
                action, _ = self._search_depth(state, depth, previous_positions)
            except _SearchBudgetExhausted:
                break
            best_action = action
            self.completed_depth = depth
        return best_action

    def _search_depth(self, root_state: GameState, depth: int, history: Set[tuple]) -> Tuple[Action, float]:
        root = root_state.turn
        frontier: List[Tuple[GameState, Action, float, Set[tuple]]] = []
        current: List[Tuple[GameState, Action | None, float, Set[tuple]]] = [(root_state, None, 0.0, set(history))]
        for _ in range(depth):
            expanded: List[Tuple[GameState, Action, float, Set[tuple]]] = []
            for state, first_action, path_score, path_history in current:
                if state.winner is not None:
                    if first_action is not None:
                        expanded.append((state, first_action, path_score, path_history))
                    continue
                actions = list(state.legal_actions())
                if not actions:
                    continue
                logits, value = self._evaluate(state, actions)
                state_value = float(value) if state.turn is root else -float(value)
                heuristic_value = normalize_heuristic(evaluate_position(state, root))
                state_signal = (1.0 - self.heuristic_blend) * state_value + self.heuristic_blend * heuristic_value
                direction = 1.0 if state.turn is root else -1.0
                safe = [
                    (action, logit) for action, logit in zip(actions, logits)
                    if repetition_key(state.apply_legal(action)) not in path_history
                ] or list(zip(actions, logits))
                ranked = sorted(
                    safe,
                    key=lambda item: (direction * (float(item[1]) + state_signal), -action_sort_key(item[0])),
                    reverse=True,
                )
                for action, logit in ranked[: self.beam_width]:
                    next_state = state.apply_legal(action)
                    next_first = first_action or action
                    next_history = path_history | {repetition_key(next_state)}
                    terminal_bonus = 1_000_000.0 if next_state.winner is root else -1_000_000.0 if next_state.winner is not None else 0.0
                    expanded.append((next_state, next_first, path_score + direction * float(logit) + direction * state_signal + terminal_bonus, next_history))
            if not expanded:
                break
            expanded.sort(key=lambda item: (item[2], -action_sort_key(item[1])), reverse=True)
            current = expanded[: self.beam_width]
            frontier = [(state, first_action, score, path_history) for state, first_action, score, path_history in current if first_action is not None]
        if not frontier:
            return root_state.legal_actions()[0], 0.0
        best_state, best_action, best_score, _ = max(frontier, key=lambda item: (item[2], -action_sort_key(item[1])))
        _ = best_state
        return best_action, best_score

    def _evaluate(self, state: GameState, actions: List[Action]) -> Tuple[List[float], float]:
        if self.nodes >= self.max_nodes:
            raise _SearchBudgetExhausted
        with torch.no_grad():
            logits, value = self.model.policy_value(state, actions)
        self.nodes += 1
        return logits.detach().cpu().tolist(), float(value.detach().cpu())


class _SearchBudgetExhausted(Exception):
    pass


def action_sort_key(action: Action) -> int:
    return action.to if action.kind == 0 else action.from_square * 10_000 + action.to


def build_roster(size: int, reserve: int, simulations: int, device: torch.device) -> List[AgentSpec]:
    roster: List[AgentSpec] = []
    if size == 5:
        checkpoints = [
            ("gnn-generation-10-5x5-r8", "Generation 10 · 5x5 reserve 8", "training/gnn/pathagon-generation-10-5x5-r8.pt"),
            ("gnn-generation-9-5x5-r8", "Generation 9 · 5x5 reserve 8", "training/gnn/pathagon-generation-9-5x5-r8.pt"),
            ("gnn-generation-7-5x5", "Generation 7 · 5x5 reserve 10", "training/gnn/pathagon-generation-7-5x5.pt"),
            ("gnn-generation-6-5x5", "Generation 6 · 5x5 reserve 10", "training/gnn/pathagon-generation-6-5x5.pt"),
        ]
    elif size in (4, 6):
        checkpoints = [
            ("gnn-generation-10-transfer-5x5", "Generation 10 · transfer from 5x5", "training/gnn/pathagon-generation-10-5x5-r8.pt"),
            ("gnn-generation-9-transfer-5x5", "Generation 9 · transfer from 5x5", "training/gnn/pathagon-generation-9-5x5-r8.pt"),
            ("gnn-generation-6-transfer-5x5", "Generation 6 · transfer from 5x5", "training/gnn/pathagon-generation-6-5x5.pt"),
        ]
    elif size == 7:
        checkpoints = [
            ("gnn-rust-generation-2-7x7", "Rust AlphaZero generation 2 · 7x7", "training/gnn/pathagon-rust-7x7-generation-2.pt"),
            ("gnn-rust-generation-1-7x7", "Rust warm-start generation 1 · 7x7", "training/gnn/pathagon-rust-7x7-generation-1.pt"),
            ("gnn-generation-8-7x7", "Generation 8 · 7x7", "training/gnn/pathagon-generation-8-7x7.pt"),
            ("gnn-generation-5-7x7", "Generation 5 · 7x7", "training/gnn/pathagon-generation-5.pt"),
            ("gnn-generation-4-7x7", "Generation 4 · 7x7", "training/gnn/pathagon-generation-4.pt"),
            ("gnn-warmstart-7x7", "Warm start · 7x7", "training/gnn/pathagon-warmstart.pt"),
        ]
    else:
        raise ValueError("league supports only 4x4, 5x5, 6x6, and 7x7 boards")
    if size == 7:
        optional_checkpoints = [
            ("cnn-warmstart-7x7", "CNN warm start · 7x7", "training/gnn/pathagon-cnn-7x7-warmstart.pt"),
        ]
        for agent_id, label, checkpoint in optional_checkpoints:
            if Path(checkpoint).exists():
                checkpoints.insert(0, (agent_id, label, checkpoint))
    for agent_id, label, checkpoint in checkpoints:
        checkpoint_path = Path(checkpoint)
        model = load_model(checkpoint_path, device)
        model.eval()
        roster.append(AgentSpec(
            agent_id,
            label,
            "gnn",
            GNNAgent(model, simulations),
            agent_manifest(runtime="python", node_budget=simulations, model_hash=checkpoint_hash(checkpoint_path)),
        ))
    if size == 4:
        pathfinder = HeuristicAgent(depth=3, beam_width=12, max_nodes=1_200)
        surveyor = HeuristicAgent(depth=2, beam_width=16, max_nodes=800)
    elif size == 5:
        pathfinder = HeuristicAgent(depth=3, beam_width=12, max_nodes=3_000)
        surveyor = HeuristicAgent(depth=2, beam_width=16, max_nodes=1_800)
    else:
        pathfinder = HeuristicAgent(depth=2, beam_width=8, max_nodes=1_000)
        surveyor = HeuristicAgent(depth=1, beam_width=12, max_nodes=500)
    roster.extend([
        AgentSpec("pathfinder-v0.3.0", "The Pathfinder", "heuristic", pathfinder, agent_manifest(runtime="python", depth=pathfinder.depth, node_budget=pathfinder.max_nodes, beam=pathfinder.beam_width)),
        AgentSpec("surveyor-v0.2.0", "The Surveyor", "heuristic", surveyor, agent_manifest(runtime="python", depth=surveyor.depth, node_budget=surveyor.max_nodes, beam=surveyor.beam_width)),
        AgentSpec("lunatic-v0.1.0", "Lunatic", "heuristic", LunaticAgent(), agent_manifest(runtime="python", depth=1)),
        AgentSpec("coin-flip-v0.0.1", "Coin Flip", "random", RandomAgent(), agent_manifest(runtime="python")),
    ])
    return roster


def play_game(light: AgentSpec, dark: AgentSpec, config: BoardConfig, seed: int) -> dict:
    moves = []

    def choose_action(state: GameState, _actions: Tuple[Action, ...], rng: random.Random, history: Set[tuple]) -> Action | None:
        agent = light if state.turn is Player.LIGHT else dark
        return agent.choose.choose_action(state, rng, history)

    def observe_move(state: GameState, action: Action, next_state: GameState) -> None:
        moves.append({
            "ply": state.ply + 1,
            "player": "light" if state.turn is Player.LIGHT else "dark",
            "action": {"kind": "place", "to": action.to} if action.kind == 0 else {"kind": "relocate", "from": action.from_square, "to": action.to},
            "captured": list(squares_from_mask(next_state.forbidden)),
            "nodes": 0,
            "completedDepth": 0,
            "tableHits": 0,
            "score": 0,
            "bookHit": False,
        })

    result = run_match(config, seed, choose_action, observe_move)
    winner = None if result.state.winner is None else ("light" if result.state.winner is Player.LIGHT else "dark")
    return record_game(light, dark, config, seed, winner, result.reason, moves)


def record_game(light: AgentSpec, dark: AgentSpec, config: BoardConfig, seed: int, winner: str | None, reason: str, moves: list) -> dict:
    return {
        "contractVersion": 1,
        "seed": seed,
        "config": game_config(config.size, config.reserve_per_player, config.max_plies),
        "engine": engine_metadata("python-gnn", "python"),
        "agents": {"light": light.id, "dark": dark.id},
        "agentSpecifications": {
            "light": agent_specification(light.id, light.label, agent_version(light.id), "puct" if light.kind == "gnn" else light.kind, "python-gnn", manifest=light.manifest),
            "dark": agent_specification(dark.id, dark.label, agent_version(dark.id), "puct" if dark.kind == "gnn" else dark.kind, "python-gnn", manifest=dark.manifest),
        },
        "winner": winner,
        "result": "win" if winner else "draw",
        "reason": reason,
        "plies": len(moves),
        "moves": moves,
    }


def checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def agent_version(agent_id: str) -> str:
    marker = agent_id.rsplit("-v", 1)
    return marker[1] if len(marker) == 2 and marker[1] else "1.0.0"


def outcome_for(record: dict, agent_id: str) -> str:
    if record["winner"] is None:
        return "draw"
    target = "light" if record["agents"]["light"] == agent_id else "dark"
    return "win" if record["winner"] == target else "loss"


def update_elo(ratings: Dict[str, float], record: dict, k_factor: float = 24.0) -> None:
    light = record["agents"]["light"]
    dark = record["agents"]["dark"]
    light_rating = ratings[light]
    dark_rating = ratings[dark]
    expected_light = 1.0 / (1.0 + 10 ** ((dark_rating - light_rating) / 400.0))
    actual_light = 1.0 if record["winner"] == "light" else 0.0 if record["winner"] == "dark" else 0.5
    ratings[light] = light_rating + k_factor * (actual_light - expected_light)
    ratings[dark] = dark_rating + k_factor * ((1.0 - actual_light) - (1.0 - expected_light))


def summarize(records: Sequence[dict], agent_id: str) -> dict:
    wins = sum(outcome_for(record, agent_id) == "win" for record in records)
    losses = sum(outcome_for(record, agent_id) == "loss" for record in records)
    draws = sum(outcome_for(record, agent_id) == "draw" for record in records)
    return {"games": len(records), "wins": wins, "losses": losses, "draws": draws, "points": wins + draws * 0.5}


def run_league(args: argparse.Namespace) -> dict:
    device = choose_device(args.device)
    config = BoardConfig(args.size, args.reserve)
    roster = build_roster(args.size, config.reserve_per_player, args.simulations, device)
    ratings = {agent.id: 1_000.0 for agent in roster}
    records: List[dict] = []
    head_to_head = []
    for left_index, left in enumerate(roster):
        for right_index in range(left_index + 1, len(roster)):
            right = roster[right_index]
            matchup: List[dict] = []
            for game_index in range(args.games_per_match):
                left_is_light = game_index % 2 == 0
                light, dark = (left, right) if left_is_light else (right, left)
                record = play_game(light, dark, config, args.seed + left_index * 100_000 + right_index * 1_000 + game_index)
                matchup.append(record)
                records.append(record)
                update_elo(ratings, record, args.k_factor)
            left_summary = summarize(matchup, left.id)
            right_summary = summarize(matchup, right.id)
            head_to_head.append({"left": left.id, "right": right.id, "games": len(matchup), "leftSummary": left_summary, "rightSummary": right_summary})
    standings = []
    for agent in roster:
        summary = summarize(records_for_agent(records, agent.id), agent.id)
        standings.append({"id": agent.id, "label": agent.label, "kind": agent.kind, "rating": round(ratings[agent.id]), **summary})
    standings.sort(key=lambda entry: (-entry["rating"], -entry["points"], entry["id"]))
    return {
        "schemaVersion": 1,
        "mode": "gnn-league",
        "boardSize": config.size,
        "reservePerPlayer": config.reserve_per_player,
        "seed": args.seed,
        "gamesPerMatch": args.games_per_match,
        "simulations": args.simulations,
        "kFactor": args.k_factor,
        "standings": standings,
        "headToHead": head_to_head,
        "games": records,
        "device": str(device),
    }


def records_for_agent(records: Sequence[dict], agent_id: str) -> List[dict]:
    return [record for record in records if agent_id in record["agents"].values()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, choices=(4, 5, 6, 7), required=True)
    parser.add_argument("--reserve", type=int, default=0)
    parser.add_argument("--games-per-match", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=4)
    parser.add_argument("--k-factor", type=float, default=24.0)
    parser.add_argument("--seed", type=int, default=20280000)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = run_league(args)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(path), "boardSize": result["boardSize"], "reservePerPlayer": result["reservePerPlayer"], "games": len(result["games"]), "standings": result["standings"]}, sort_keys=True))


if __name__ == "__main__":
    main()
