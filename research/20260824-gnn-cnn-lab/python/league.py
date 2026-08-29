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
from .selfplay import _blend_probabilities, _mix_uniform, _softmax_scores, avoid_repeated_successors, run_match
from .tactics import immediate_winning_actions
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
        actions = self._root_actions(state)
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

    def _root_actions(self, state: GameState) -> List[Action]:
        """Return the bounded root beam used by Pathfinder.

        Keeping this as a separate hook lets learned sorters change only root
        ordering/candidate selection while retaining Pathfinder's alpha-beta
        evaluator and recursive move ordering.
        """

        return self._ordered_actions(state, state.turn)[: self.beam_width]

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


class SorterPathfinderAgent(HeuristicAgent):
    """Use a compact policy model to order Pathfinder's root beam.

    The model is deliberately a sorter, not a replacement evaluator. We
    always retain Pathfinder's cheap heuristic candidates as a fallback and
    preserve immediate wins before taking the model's top-k suggestions. The
    candidate can additionally use a bounded transposition table and tactical
    leaf extension; `SorterOnlyPathfinderAgent` isolates the ordering-only
    effect at the same node budget.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        depth: int,
        beam_width: int,
        max_nodes: int,
        top_k: int,
    ) -> None:
        super().__init__(depth=depth, beam_width=beam_width, max_nodes=max_nodes)
        if top_k < 1:
            raise ValueError("sorter top_k must be positive")
        self.model = model
        self.top_k = top_k
        self._table: Dict[tuple, float] = {}
        self._best_moves: Dict[tuple, Action] = {}

    def choose_action(self, state: GameState, rng: random.Random, history: Set[tuple]) -> Action | None:
        # Search state is intentionally scoped to one move. Reusing entries
        # across moves would require repetition history in the key and could
        # turn a useful bound into an invalid one.
        self._table = {}
        self._best_moves = {}
        return super().choose_action(state, rng, history)

    def _search(self, state: GameState, root: Player, depth: int, alpha: float, beta: float) -> float:
        if state.winner is not None:
            return evaluate_position(state, root)
        if self.nodes >= self.max_nodes:
            return evaluate_position(state, root)

        # A one-ply tactical extension prevents the normal depth horizon from
        # overlooking an immediate win (including the opponent's win).
        if depth <= 0:
            winning = [
                action
                for action in state.legal_actions()
                if state.apply_legal(action).winner is state.turn
            ]
            if not winning:
                return evaluate_position(state, root)
            self.nodes += 1
            return evaluate_position(state.apply_legal(winning[0]), root)

        key = (state, root, depth)
        cached = self._table.get(key)
        if cached is not None:
            return cached
        actions = self._ordered_actions(state, root)[: self.beam_width]
        if not actions:
            return evaluate_position(state, root)
        preferred = self._best_moves.get(key)
        if preferred in actions:
            actions.remove(preferred)
            actions.insert(0, preferred)

        maximizing = state.turn is root
        best = float("-inf") if maximizing else float("inf")
        best_action = actions[0]
        cut_off = False
        for action in actions:
            if self.nodes >= self.max_nodes:
                break
            self.nodes += 1
            score = self._search(state.apply_legal(action), root, depth - 1, alpha, beta)
            if maximizing:
                if score > best or (score == best and action_sort_key(action) < action_sort_key(best_action)):
                    best, best_action = score, action
                alpha = max(alpha, best)
            else:
                if score < best or (score == best and action_sort_key(action) < action_sort_key(best_action)):
                    best, best_action = score, action
                beta = min(beta, best)
            if beta <= alpha:
                cut_off = True
                break
        if not cut_off and self.nodes < self.max_nodes:
            self._table[key] = best
            self._best_moves[key] = best_action
        return best

    def _root_actions(self, state: GameState) -> List[Action]:
        actions = list(state.legal_actions())
        if not actions:
            return []
        heuristic_actions = super()._root_actions(state)
        # Keep the candidate set identical to Pathfinder's bounded beam. The
        # compact model is a sorter only; it cannot spend its weaker policy
        # estimate to replace a heuristic candidate and quietly change the
        # search width.
        sort_pool = heuristic_actions[: max(1, min(self.top_k, len(heuristic_actions)))]
        with torch.no_grad():
            if getattr(self.model, "qadv", False):
                _logits, _value, sort_scores, _advantages = self.model.policy_value_q(state, sort_pool)
            else:
                sort_scores, _value = self.model.policy_value(state, sort_pool)
        ranked = sorted(
            zip(sort_pool, sort_scores.detach().cpu().tolist()),
            key=lambda item: (float(item[1]), -action_sort_key(item[0])),
            reverse=True,
        )
        model_actions = [action for action, _logit in ranked]
        immediate_wins = [
            action for action in actions if state.apply_legal(action).winner is state.turn
        ]
        forced_blocks: List[Action] = []
        if not immediate_wins:
            opponent_wins = immediate_winning_actions(state, state.turn.other())
            if opponent_wins:
                forced_blocks = [
                    action
                    for action in actions
                    if not immediate_winning_actions(state.apply_legal(action), state.turn.other())
                ]
        # The result remains capped at Pathfinder's original beam width, so the
        # comparison does not quietly purchase more search. Immediate wins are
        # a correctness guard; the model reorders only the existing head and
        # the untouched Pathfinder tail provides deterministic fallback.
        ordered: List[Action] = []
        for action in immediate_wins + forced_blocks + model_actions + heuristic_actions[len(sort_pool) :]:
            if action not in ordered:
                ordered.append(action)
            if len(ordered) >= self.beam_width:
                break
        return ordered


class SorterOnlyPathfinderAgent(SorterPathfinderAgent):
    """Ablation: compact root ordering with the original Pathfinder search."""

    def _search(self, state: GameState, root: Player, depth: int, alpha: float, beta: float) -> float:
        return HeuristicAgent._search(self, state, root, depth, alpha, beta)


class GNNAgent:
    def __init__(self, model: torch.nn.Module, simulations: int) -> None:
        self.search = PUCTSearch(model, simulations=simulations)

    def choose_action(self, state: GameState, _rng: random.Random, history: Set[tuple]) -> Action | None:
        _, actions, probabilities = self.search.run(state, add_root_noise=False, history=history)
        if not actions:
            return None
        _, filtered = avoid_repeated_successors(state, actions, probabilities, history)
        return actions[max(range(len(actions)), key=lambda index: (filtered[index], -action_sort_key(actions[index])))]


class QAdvAgent:
    """Direct legal-action selector backed by a dueling Q/advantage head."""

    def __init__(self, model: torch.nn.Module) -> None:
        if not getattr(model, "qadv", False):
            raise ValueError("QAdvAgent requires a qadv-enabled model")
        self.model = model

    def choose_action(self, state: GameState, _rng: random.Random, history: Set[tuple]) -> Action | None:
        actions = list(state.legal_actions())
        if not actions:
            return None
        with torch.no_grad():
            _logits, _value, q_values, _advantages = self.model.policy_value_q(state, actions)
        safe = [
            index for index, action in enumerate(actions)
            if repetition_key(state.apply_legal(action)) not in history
        ]
        candidate_indices = safe or list(range(len(actions)))
        chosen_index = max(
            candidate_indices,
            key=lambda index: (float(q_values[index].detach().cpu()), -action_sort_key(actions[index])),
        )
        return actions[chosen_index]


class QAdvGuidedAgent:
    """Use QAdv to narrow actions, then verify them against the best reply.

    The Q/advantage head is a broad action-ranking prior. It is not trusted as
    a complete player: every root candidate is screened for immediate losses,
    and the surviving top-Q/top-heuristic candidates are evaluated against a
    shallow adversarial reply search.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        top_k: int = 12,
        reply_k: int = 8,
        q_weight: float = 0.50,
        reply_weight: float = 0.35,
        heuristic_weight: float = 0.15,
        temperature_moves: int = 48,
        policy_temperature: float = 1.15,
        opening_moves: int = 16,
        opening_temperature: float = 1.8,
        opening_randomness: float = 0.30,
        pathfinder_temperature: float = 1.15,
    ) -> None:
        if not getattr(model, "qadv", False):
            raise ValueError("QAdvGuidedAgent requires a qadv-enabled model")
        if top_k < 1 or reply_k < 1:
            raise ValueError("top_k and reply_k must be positive")
        total = q_weight + reply_weight + heuristic_weight
        if total <= 0.0:
            raise ValueError("guided-search weights must have a positive sum")
        if temperature_moves < 0 or opening_moves < 0:
            raise ValueError("temperature and opening move counts must be non-negative")
        for name, value in (
            ("policy_temperature", policy_temperature),
            ("opening_temperature", opening_temperature),
            ("pathfinder_temperature", pathfinder_temperature),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= opening_randomness <= 1.0:
            raise ValueError("opening_randomness must be between 0 and 1")
        self.model = model
        self.top_k = top_k
        self.reply_k = reply_k
        self.q_weight = q_weight / total
        self.reply_weight = reply_weight / total
        self.heuristic_weight = heuristic_weight / total
        self.temperature_moves = temperature_moves
        self.policy_temperature = policy_temperature
        self.opening_moves = opening_moves
        self.opening_temperature = opening_temperature
        self.opening_randomness = opening_randomness
        self.pathfinder_temperature = pathfinder_temperature

    def choose_action(self, state: GameState, rng: random.Random, history: Set[tuple]) -> Action | None:
        actions = list(state.legal_actions())
        if not actions:
            return None
        safe_actions = [
            action for action in actions
            if repetition_key(state.apply_legal(action)) not in history
        ] or actions
        with torch.no_grad():
            _logits, _value, q_values, _advantages = self.model.policy_value_q(state, safe_actions)
        root_q = {action: float(value) for action, value in zip(safe_actions, q_values.detach().cpu().tolist())}

        immediate_wins = [
            action for action in safe_actions
            if state.apply_legal(action).winner is state.turn
        ]
        if immediate_wins:
            return min(immediate_wins, key=action_sort_key)

        entries = []
        for action in safe_actions:
            afterstate = state.apply_legal(action)
            heuristic = normalize_heuristic(evaluate_position(afterstate, state.turn))
            entries.append((action, afterstate, heuristic))

        candidate_pool = entries
        by_q = sorted(candidate_pool, key=lambda entry: (root_q[entry[0]], -action_sort_key(entry[0])), reverse=True)
        by_heuristic = sorted(candidate_pool, key=lambda entry: (entry[2], -action_sort_key(entry[0])), reverse=True)
        candidates = {entry[0]: entry for entry in by_q[: self.top_k]}
        for entry in by_heuristic[: max(1, self.top_k // 2)]:
            candidates[entry[0]] = entry

        scored = []
        for action, afterstate, heuristic in candidates.values():
            replies = list(afterstate.legal_actions())
            safe_replies = [
                reply for reply in replies
                if repetition_key(afterstate.apply_legal(reply)) not in (history | {repetition_key(afterstate)})
            ] or replies
            if not safe_replies:
                scored.append((root_q[action], action))
                continue
            with torch.no_grad():
                _reply_logits, _reply_value, reply_q_values, _reply_advantages = self.model.policy_value_q(afterstate, safe_replies)
            reply_q = reply_q_values.detach().cpu().tolist()
            reply_order = sorted(range(len(safe_replies)), key=lambda index: (reply_q[index], -action_sort_key(safe_replies[index])), reverse=True)
            tactical_reply_indices = [
                index for index, reply in enumerate(safe_replies)
                if afterstate.apply_legal(reply).winner is afterstate.turn
            ]
            if tactical_reply_indices:
                scored.append((-1.0, action))
                continue
            reply_indices = reply_order[: self.reply_k]
            worst_reply_q = max(float(reply_q[index]) for index in reply_indices)
            worst_reply_heuristic = min(
                normalize_heuristic(evaluate_position(afterstate.apply_legal(safe_replies[index]), state.turn))
                for index in reply_indices
            )
            reply_score = -worst_reply_q
            score = (
                self.q_weight * root_q[action]
                + self.reply_weight * reply_score
                + self.heuristic_weight * min(heuristic, worst_reply_heuristic)
            )
            scored.append((score, action))
        if not scored:
            return safe_actions[0]
        scored.sort(key=lambda item: (-item[0], action_sort_key(item[1])))
        score_values = [item[0] for item in scored]
        actions_by_score = [item[1] for item in scored]
        in_opening = state.ply < self.opening_moves
        effective_temperature = self.opening_temperature if in_opening else self.policy_temperature
        probabilities = _softmax_scores(score_values, effective_temperature)
        if in_opening and self.opening_randomness > 0.0:
            probabilities = _mix_uniform(probabilities, self.opening_randomness)
        if state.ply < self.temperature_moves:
            return rng.choices(actions_by_score, weights=probabilities, k=1)[0]
        return actions_by_score[max(range(len(actions_by_score)), key=lambda index: (probabilities[index], -action_sort_key(actions_by_score[index])))]


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
            ("gnn-generation-10-5x5-r8", "Generation 10 · 5x5 reserve 8", "research/runs/gnn/pathagon-generation-10-5x5-r8.pt"),
            ("gnn-generation-9-5x5-r8", "Generation 9 · 5x5 reserve 8", "research/runs/gnn/pathagon-generation-9-5x5-r8.pt"),
            ("gnn-generation-7-5x5", "Generation 7 · 5x5 reserve 10", "research/runs/gnn/pathagon-generation-7-5x5.pt"),
            ("gnn-generation-6-5x5", "Generation 6 · 5x5 reserve 10", "research/runs/gnn/pathagon-generation-6-5x5.pt"),
        ]
    elif size in (4, 6):
        checkpoints = [
            ("gnn-generation-10-transfer-5x5", "Generation 10 · transfer from 5x5", "research/runs/gnn/pathagon-generation-10-5x5-r8.pt"),
            ("gnn-generation-9-transfer-5x5", "Generation 9 · transfer from 5x5", "research/runs/gnn/pathagon-generation-9-5x5-r8.pt"),
            ("gnn-generation-6-transfer-5x5", "Generation 6 · transfer from 5x5", "research/runs/gnn/pathagon-generation-6-5x5.pt"),
        ]
    elif size == 7:
        checkpoints = [
            ("gnn-rust-generation-2-7x7", "Rust AlphaZero generation 2 · 7x7", "research/runs/gnn/pathagon-rust-7x7-generation-2.pt"),
            ("gnn-rust-generation-1-7x7", "Rust warm-start generation 1 · 7x7", "research/runs/gnn/pathagon-rust-7x7-generation-1.pt"),
            ("gnn-generation-8-7x7", "Generation 8 · 7x7", "research/runs/gnn/pathagon-generation-8-7x7.pt"),
            ("gnn-generation-5-7x7", "Generation 5 · 7x7", "research/runs/gnn/pathagon-generation-5.pt"),
            ("gnn-generation-4-7x7", "Generation 4 · 7x7", "research/runs/gnn/pathagon-generation-4.pt"),
            ("gnn-warmstart-7x7", "Warm start · 7x7", "research/runs/gnn/pathagon-warmstart.pt"),
        ]
    else:
        raise ValueError("league supports only 4x4, 5x5, 6x6, and 7x7 boards")
    if size == 7:
        optional_checkpoints = [
            ("cnn-warmstart-7x7", "CNN warm start · 7x7", "research/runs/gnn/pathagon-cnn-7x7-warmstart.pt"),
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


def play_game(
    light: AgentSpec,
    dark: AgentSpec,
    config: BoardConfig,
    seed: int,
    opening_random_plies: int = 0,
) -> dict:
    moves = []

    def choose_action(state: GameState, actions: Tuple[Action, ...], rng: random.Random, history: Set[tuple]) -> Action | None:
        if state.ply < opening_random_plies:
            return rng.choice(actions)
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
