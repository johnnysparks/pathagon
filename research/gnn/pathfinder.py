"""Pathfinder-style heuristic guidance for exploratory self-play."""

from __future__ import annotations

from typing import Iterable, List

from .evaluation import evaluate_position
from .game import Action, GameState, Player


def action_sort_key(action: Action) -> int:
    return action.to if action.kind == 0 else action.from_square * 10_000 + action.to


class PathfinderGuide:
    """Score root actions with the shallow search used by The Pathfinder.

    The guide is deliberately a soft prior. It never replaces MCTS root-Q
    targets; callers blend its scores into the sampled policy so exploration
    can increase without turning the data generator into a pure heuristic
    player.
    """

    def __init__(self, depth: int = 2, beam_width: int = 8, max_nodes: int = 1_000) -> None:
        if depth < 1 or beam_width < 1 or max_nodes < 1:
            raise ValueError("Pathfinder guidance depth, beam, and node budget must be positive")
        self.depth = depth
        self.beam_width = beam_width
        self.max_nodes = max_nodes
        self.nodes = 0

    def score_actions(self, state: GameState, actions: Iterable[Action]) -> List[float]:
        self.nodes = 0
        root = state.turn
        scores: List[float] = []
        for action in actions:
            afterstate = state.apply_legal(action)
            if afterstate.winner is root:
                scores.append(1_000_000_000.0)
                continue
            if self.nodes >= self.max_nodes:
                scores.append(float(evaluate_position(afterstate, root)))
                continue
            self.nodes += 1
            scores.append(float(self._search(afterstate, root, self.depth - 1, float("-inf"), float("inf"))))
        return scores

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
        scored.sort(key=lambda item: (item[0], -action_sort_key(item[1])), reverse=True)
        return [action for _, action in scored]
