"""Run checkpoint and heuristic agents in a color-balanced Elo league."""

from __future__ import annotations

import argparse
import heapq
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import torch

from .game import Action, BoardConfig, GameState, Player, repetition_key
from .mcts import PUCTSearch
from .selfplay import avoid_repeated_successors
from .train import choose_device, load_model


@dataclass(frozen=True)
class AgentSpec:
    id: str
    label: str
    kind: str
    choose: object


class RandomAgent:
    def choose_action(self, state: GameState, rng: random.Random, _history: Set[tuple]) -> Action | None:
        actions = list(state.legal_actions())
        return rng.choice(actions) if actions else None


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


def evaluate_position(state: GameState, player: Player) -> float:
    if state.winner is player:
        return 1_000_000_000 - state.ply
    if state.winner is player.other():
        return -1_000_000_000 + state.ply
    own_distance = connection_distance(state, player)
    opponent_distance = connection_distance(state, player.other())
    own_pieces = count_bits(state.pieces(player))
    opponent_pieces = count_bits(state.pieces(player.other()))
    capture_direction = 1 if state.last_player is player else -1
    structure = largest_component(state, player) - largest_component(state, player.other())
    threats = capture_opportunities(state, player) - capture_opportunities(state, player.other())
    edges = edge_presence(state, player) - edge_presence(state, player.other())
    return (
        (opponent_distance - own_distance) * 240
        + (own_pieces - opponent_pieces) * 110
        + capture_direction * state.last_capture * 700
        + structure * 55
        + threats * 130
        + edges * 80
    )


def connection_distance(state: GameState, player: Player) -> int:
    size = state.config.size
    opponent = player.other()
    distances = [float("inf")] * state.config.cell_count
    frontier: List[Tuple[int, int]] = []
    starts = range((size - 1) * size, size * size) if player is Player.LIGHT else range(0, size * size, size)
    for square in starts:
        if state.board_at(square) is opponent:
            continue
        distance = 0 if state.board_at(square) is player else 1
        if distance < distances[square]:
            distances[square] = distance
            heapq.heappush(frontier, (distance, square))
    while frontier:
        distance, square = heapq.heappop(frontier)
        if distance != distances[square]:
            continue
        row, column = divmod(square, size)
        if (player is Player.LIGHT and row == 0) or (player is Player.DARK and column == size - 1):
            return distance
        for neighbor in neighbors(state.config, square):
            if state.board_at(neighbor) is opponent:
                continue
            next_distance = distance + (0 if state.board_at(neighbor) is player else 1)
            if next_distance < distances[neighbor]:
                distances[neighbor] = next_distance
                heapq.heappush(frontier, (next_distance, neighbor))
    return state.config.cell_count


def largest_component(state: GameState, player: Player) -> int:
    remaining = set(squares_from_mask(state.pieces(player)))
    largest = 0
    while remaining:
        stack = [remaining.pop()]
        component = 0
        while stack:
            square = stack.pop()
            component += 1
            for neighbor in neighbors(state.config, square):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        largest = max(largest, component)
    return largest


def capture_opportunities(state: GameState, player: Player) -> int:
    forbidden = state.forbidden
    opponent = player.other()
    victims: Set[int] = set()
    for origin in range(state.config.cell_count):
        if state.board_at(origin) is not None or forbidden & (1 << origin):
            continue
        row, column = divmod(origin, state.config.size)
        for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            far_row = row + row_delta * 2
            far_column = column + column_delta * 2
            if not (0 <= far_row < state.config.size and 0 <= far_column < state.config.size):
                continue
            near = (row + row_delta) * state.config.size + column + column_delta
            far = far_row * state.config.size + far_column
            if state.board_at(near) is opponent and state.board_at(far) is player:
                victims.add(near)
    return len(victims)


def edge_presence(state: GameState, player: Player) -> int:
    size = state.config.size
    if player is Player.LIGHT:
        near = range((size - 1) * size, size * size)
        far = range(size)
    else:
        near = range(0, size * size, size)
        far = range(size - 1, size * size, size)
    return int(any(state.board_at(square) is player for square in near)) + int(any(state.board_at(square) is player for square in far))


def neighbors(config: BoardConfig, square: int) -> Iterable[int]:
    row, column = divmod(square, config.size)
    if row:
        yield square - config.size
    if row + 1 < config.size:
        yield square + config.size
    if column:
        yield square - 1
    if column + 1 < config.size:
        yield square + 1


def squares_from_mask(mask: int) -> Iterable[int]:
    while mask:
        lowest = mask & -mask
        yield lowest.bit_length() - 1
        mask ^= lowest


def count_bits(mask: int) -> int:
    return bin(mask).count("1")


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
    elif size == 7:
        checkpoints = [
            ("gnn-generation-8-7x7", "Generation 8 · 7x7", "training/gnn/pathagon-generation-8-7x7.pt"),
            ("gnn-generation-5-7x7", "Generation 5 · 7x7", "training/gnn/pathagon-generation-5.pt"),
            ("gnn-generation-4-7x7", "Generation 4 · 7x7", "training/gnn/pathagon-generation-4.pt"),
            ("gnn-warmstart-7x7", "Warm start · 7x7", "training/gnn/pathagon-warmstart.pt"),
        ]
    else:
        raise ValueError("league supports only 5x5 and 7x7 boards")
    for agent_id, label, checkpoint in checkpoints:
        model = load_model(Path(checkpoint), device)
        model.eval()
        roster.append(AgentSpec(agent_id, label, "gnn", GNNAgent(model, simulations)))
    pathfinder = (HeuristicAgent(depth=3, beam_width=12, max_nodes=3_000)
                  if size == 5 else HeuristicAgent(depth=2, beam_width=8, max_nodes=1_000))
    surveyor = (HeuristicAgent(depth=2, beam_width=16, max_nodes=1_800)
                if size == 5 else HeuristicAgent(depth=1, beam_width=12, max_nodes=500))
    roster.extend([
        AgentSpec("pathfinder-v0.3.0", "The Pathfinder", "heuristic", pathfinder),
        AgentSpec("surveyor-v0.2.0", "The Surveyor", "heuristic", surveyor),
        AgentSpec("coin-flip-v0.0.1", "Coin Flip", "random", RandomAgent()),
    ])
    return roster


def play_game(light: AgentSpec, dark: AgentSpec, config: BoardConfig, seed: int) -> dict:
    rng = random.Random(seed)
    state = GameState.initial(config)
    moves = []
    repetitions: Dict[tuple, int] = {}
    while state.winner is None and state.ply < config.max_plies:
        position = repetition_key(state)
        repetitions[position] = repetitions.get(position, 0) + 1
        if repetitions[position] >= 3:
            return record_game(light, dark, config, seed, None, "threefold-repetition", moves)
        actions = list(state.legal_actions())
        if not actions:
            return record_game(light, dark, config, seed, None, "no-legal-action", moves)
        agent = light if state.turn is Player.LIGHT else dark
        chooser = agent.choose
        action = chooser.choose_action(state, rng, set(repetitions))
        if action is None or action not in actions:
            return record_game(light, dark, config, seed, None, "no-legal-action", moves)
        next_state = state.apply_legal(action)
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
        state = next_state
    if state.winner is not None:
        return record_game(light, dark, config, seed, "light" if state.winner is Player.LIGHT else "dark", "path", moves)
    return record_game(light, dark, config, seed, None, "max-plies", moves)


def record_game(light: AgentSpec, dark: AgentSpec, config: BoardConfig, seed: int, winner: str | None, reason: str, moves: list) -> dict:
    return {
        "schemaVersion": 2,
        "seed": seed,
        "boardSize": config.size,
        "reservePerPlayer": config.reserve_per_player,
        "agents": {"light": light.id, "dark": dark.id},
        "winner": winner,
        "result": "win" if winner else "draw",
        "reason": reason,
        "plies": len(moves),
        "moves": moves,
    }


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
    parser.add_argument("--size", type=int, choices=(5, 7), required=True)
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
