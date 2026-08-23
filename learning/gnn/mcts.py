"""PUCT search over the dynamic action space."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Set, Tuple

import torch

from .game import Action, GameState, winner_value
from .model import PathagonGNN


class MCTSNode:
    def __init__(self, state: GameState, parent: Optional["MCTSNode"] = None, action: Optional[Action] = None) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.children: Dict[Action, MCTSNode] = {}
        self.priors: Dict[Action, float] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.expanded = False

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


class PUCTSearch:
    def __init__(
        self,
        model: PathagonGNN,
        simulations: int = 64,
        cpuct: float = 1.5,
        dirichlet_epsilon: float = 0.25,
        dirichlet_alpha: float = 0.30,
    ) -> None:
        self.model = model
        self.simulations = simulations
        self.cpuct = cpuct
        self.dirichlet_epsilon = dirichlet_epsilon
        self.dirichlet_alpha = dirichlet_alpha

    @torch.no_grad()
    def expand(self, node: MCTSNode) -> float:
        if node.state.winner is not None:
            node.expanded = True
            return winner_value(node.state, node.state.turn)
        actions = list(node.state.legal_actions())
        if not actions:
            node.expanded = True
            return 0.0
        logits, value = self.model.policy_value(node.state, actions)
        probabilities = torch.softmax(logits, dim=0).detach().cpu().tolist()
        node.priors = {action: float(probability) for action, probability in zip(actions, probabilities)}
        node.expanded = True
        return float(value.detach().cpu())

    def run(self, state: GameState, add_root_noise: bool = False) -> Tuple[MCTSNode, List[Action], List[float]]:
        root = MCTSNode(state)
        self.expand(root)
        if add_root_noise and root.priors:
            self._add_root_noise(root)
        for _ in range(self.simulations):
            self._simulate(root, set())
        actions = list(state.legal_actions())
        probabilities = self.visit_policy(root, actions, temperature=1.0)
        return root, actions, probabilities

    def _simulate(self, node: MCTSNode, path_states: Set[GameState]) -> float:
        if node.state in path_states:
            return 0.0
        path_states.add(node.state)
        try:
            if not node.expanded:
                value = self.expand(node)
            elif node.state.winner is not None or not node.priors:
                value = winner_value(node.state, node.state.turn)
            else:
                action = self._select_action(node)
                child = node.children.get(action)
                if child is None:
                    child = MCTSNode(node.state.apply_legal(action), node, action)
                    node.children[action] = child
                value = -self._simulate(child, path_states)
            node.visit_count += 1
            node.value_sum += value
            return value
        finally:
            path_states.remove(node.state)

    def _select_action(self, node: MCTSNode) -> Action:
        parent_scale = math.sqrt(max(1, node.visit_count))
        best_action = None
        best_score = float("-inf")
        for action in node.state.legal_actions():
            child = node.children.get(action)
            child_visits = 0 if child is None else child.visit_count
            child_value = 0.0 if child is None else -child.mean_value
            prior = node.priors.get(action, 0.0)
            score = child_value + self.cpuct * prior * parent_scale / (1.0 + child_visits)
            if best_action is None or score > best_score or (score == best_score and action < best_action):
                best_action = action
                best_score = score
        if best_action is None:
            raise RuntimeError("PUCT selected from a state without legal actions")
        return best_action

    def visit_policy(self, root: MCTSNode, actions: List[Action], temperature: float = 1.0) -> List[float]:
        counts = [float(root.children[action].visit_count) if action in root.children else 0.0 for action in actions]
        if not any(counts):
            return [1.0 / len(actions)] * len(actions) if actions else []
        if temperature <= 0:
            best = max(range(len(counts)), key=lambda index: (counts[index], -actions[index].to, -actions[index].from_square))
            return [1.0 if index == best else 0.0 for index in range(len(actions))]
        powered = [count ** (1.0 / temperature) for count in counts]
        total = sum(powered)
        return [value / total for value in powered]

    def _add_root_noise(self, root: MCTSNode) -> None:
        actions = list(root.priors)
        noise = [random.gammavariate(self.dirichlet_alpha, 1.0) for _ in actions]
        total = sum(noise)
        if total == 0:
            return
        for action, sample in zip(actions, noise):
            root.priors[action] = (1.0 - self.dirichlet_epsilon) * root.priors[action] + self.dirichlet_epsilon * sample / total

