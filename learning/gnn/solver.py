"""Exact small-board endgame search with a repetition-aware transposition table.

The solver deliberately knows nothing about named tactics. It only evaluates
legal successors, terminal wins, no-move draws, the ply cap, and threefold
repetition. The tactical audit uses it as an oracle while the neural search
remains the system under evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from .game import Action, GameState, repetition_key, winner_value


@dataclass(frozen=True)
class SolverResult:
    """The exact win/draw/loss result from the side-to-move perspective."""

    outcome: int


@dataclass(frozen=True)
class SolvedAction:
    """The result of one root action from the root player's perspective."""

    action: Action
    result: SolverResult


@dataclass(frozen=True)
class SolverAnalysis:
    """Exact root result plus the action-level labels used by the audit."""

    result: SolverResult
    actions: Tuple[SolvedAction, ...]

    @property
    def optimal_actions(self) -> Tuple[Action, ...]:
        return tuple(item.action for item in self.actions if item.result.outcome == self.result.outcome)


@dataclass(frozen=True)
class SolverStats:
    nodes: int
    cache_hits: int
    table_entries: int


RepetitionHistory = Mapping[tuple, int]
HistorySignature = frozenset[tuple[tuple, int]]
TableKey = tuple[tuple, int, Optional[int], HistorySignature]


@dataclass(frozen=True)
class _TableEntry:
    result: SolverResult
    bound: str


class ExactSolver:
    """Solve queried positions on boards of size four or smaller exactly.

    The table includes the repetition-count signature because a position that
    has occurred twice is not equivalent to the same position seen once. This
    makes the cache safe for threefold repetition rather than merely treating
    a cycle encountered on the current recursion path as a draw.

    ``history`` contains occurrences before ``state``. The current state is
    added automatically. Most callers, including the audit, should omit it.

    ``horizon`` optionally limits the proof to a fixed number of plies. A
    non-terminal position at the horizon is scored as a draw/unknown, which
    is useful for tactical audits and keeps the audit independent of named
    tactical predicates. With ``horizon=None`` the search continues until a
    rule terminal, repetition draw, or ply cap.
    """

    def __init__(self, max_size: int = 4, horizon: Optional[int] = None) -> None:
        if max_size < 3 or max_size > 4:
            raise ValueError("ExactSolver supports board sizes from 3 through 4")
        if horizon is not None and horizon < 0:
            raise ValueError("horizon must be non-negative")
        self.max_size = max_size
        self.horizon = horizon
        self._table: Dict[TableKey, _TableEntry] = {}
        self._nodes = 0
        self._cache_hits = 0

    @property
    def stats(self) -> SolverStats:
        return SolverStats(self._nodes, self._cache_hits, len(self._table))

    def clear(self) -> None:
        self._table.clear()
        self._nodes = 0
        self._cache_hits = 0

    def solve(self, state: GameState, history: Optional[RepetitionHistory] = None) -> SolverResult:
        """Return the exact result for ``state``.

        The transposition table intentionally survives between calls so that
        symmetry-related audit positions can reuse solved subtrees. Call
        ``clear`` when measuring an isolated search.
        """

        self._validate_state(state)
        counts = dict(history or {})
        position = repetition_key(state)
        counts[position] = counts.get(position, 0) + 1
        return self._solve(state, counts, self.horizon)

    def analyze(self, state: GameState, history: Optional[RepetitionHistory] = None) -> SolverAnalysis:
        """Return exact root labels for every legal action."""

        self._validate_state(state)
        counts = dict(history or {})
        position = repetition_key(state)
        counts[position] = counts.get(position, 0) + 1
        result = self._solve(state, counts, self.horizon)
        solved_actions = []
        for action in state.legal_actions():
            child = state.apply_legal(action)
            child_counts = self._next_counts(counts, child)
            child_result = self._solve(child, child_counts, self._next_depth(self.horizon))
            solved_actions.append(SolvedAction(action, self._from_child(child_result)))
        return SolverAnalysis(result, tuple(solved_actions))

    def _validate_state(self, state: GameState) -> None:
        if state.config.size > self.max_size:
            raise ValueError(f"ExactSolver is limited to board size {self.max_size}")

    @staticmethod
    def _next_counts(counts: Dict[tuple, int], state: GameState) -> Dict[tuple, int]:
        next_counts = dict(counts)
        position = repetition_key(state)
        next_counts[position] = next_counts.get(position, 0) + 1
        return next_counts

    @staticmethod
    def _next_depth(depth_remaining: Optional[int]) -> Optional[int]:
        return None if depth_remaining is None else depth_remaining - 1

    @staticmethod
    def _history_signature(counts: Dict[tuple, int]) -> HistorySignature:
        return frozenset((position, count) for position, count in counts.items() if count)

    @staticmethod
    def _table_key(
        state: GameState,
        counts: Dict[tuple, int],
        depth_remaining: Optional[int],
    ) -> TableKey:
        config = (state.config.size, state.config.reserve_per_player, state.config.max_plies)
        position = (
            config,
            state.ply,
            state.light,
            state.dark,
            state.reserves,
            state.turn,
            state.forbidden,
            state.last_relocated_to,
        )
        return position, state.ply, depth_remaining, ExactSolver._history_signature(counts)

    @staticmethod
    def _from_child(child: SolverResult) -> SolverResult:
        return SolverResult(-child.outcome)

    @staticmethod
    def _is_better(candidate: SolverResult, incumbent: Optional[SolverResult]) -> bool:
        if incumbent is None or candidate.outcome != incumbent.outcome:
            return incumbent is None or candidate.outcome > incumbent.outcome
        return False

    @staticmethod
    def _ordered_actions(state: GameState) -> Tuple[Action, ...]:
        """Visit immediate terminal successors first, then stable action order."""

        actions = list(state.legal_actions())
        actions.sort(
            key=lambda action: (
                state.apply_legal(action).winner is not state.turn,
                action,
            )
        )
        return tuple(actions)

    def _solve(
        self,
        state: GameState,
        counts: Dict[tuple, int],
        depth_remaining: Optional[int],
        alpha: int = -1,
        beta: int = 1,
    ) -> SolverResult:
        key = self._table_key(state, counts, depth_remaining)
        cached = self._table.get(key)
        if cached is not None:
            if cached.bound == "exact":
                self._cache_hits += 1
                return cached.result
            if cached.bound == "lower":
                alpha = max(alpha, cached.result.outcome)
            else:
                beta = min(beta, cached.result.outcome)
            if alpha >= beta:
                self._cache_hits += 1
                return cached.result

        original_alpha = alpha
        original_beta = beta
        self._nodes += 1
        if state.winner is not None:
            result = SolverResult(int(winner_value(state, state.turn)))
        elif counts.get(repetition_key(state), 0) >= 3:
            result = SolverResult(0)
        elif state.ply >= state.config.max_plies:
            result = SolverResult(0)
        elif depth_remaining == 0:
            result = SolverResult(0)
        else:
            actions = self._ordered_actions(state)
            if not actions:
                result = SolverResult(0)
            else:
                best: Optional[SolverResult] = None
                for action in actions:
                    child = state.apply_legal(action)
                    child_result = self._solve(
                        child,
                        self._next_counts(counts, child),
                        self._next_depth(depth_remaining),
                        -beta,
                        -alpha,
                    )
                    candidate = self._from_child(child_result)
                    if self._is_better(candidate, best):
                        best = candidate
                    alpha = max(alpha, candidate.outcome)
                    if alpha >= beta:
                        break
                if best is None:
                    raise RuntimeError("exact solver evaluated a state without a result")
                result = best

        if result.outcome <= original_alpha:
            bound = "upper"
        elif result.outcome >= original_beta:
            bound = "lower"
        else:
            bound = "exact"
        self._table[key] = _TableEntry(result, bound)
        return result


__all__ = ["ExactSolver", "SolvedAction", "SolverAnalysis", "SolverResult", "SolverStats"]
