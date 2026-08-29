#!/usr/bin/env python3
"""Run color-balanced games between Luna's GPT-guided policy and Pathfinder.

Luna is not a learned checkpoint or the existing Lunatic bot.  This is a
small, explicit distillation of the move-selection principles used by the GPT
player: protect immediate connection threats, prefer robust whole-board paths,
and use a broader adversarial lookahead than the narrow Pathfinder evaluator.
The script keeps the two evaluators separate and stores complete replays in
the experiment workspace.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import random
import sys
import time
import types
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
LAB_DIR = REPO_ROOT / "research/20260824-gnn-cnn-lab/python"
DEFAULT_WORKSPACE = REPO_ROOT / "research/20260828-luna-vs-pathfinder-depth/workspace"

# This is the promoted filter-aware evaluator's current weight set.  The
# Python mirror is intentionally kept here rather than imported from a
# production module so the research run remains self-contained.
PATHFINDER_WEIGHTS = {
    "path": 241,
    "material": 112,
    "capture": 887,
    "structure": 40,
    "threat": 154,
    "edge": 74,
}

LUNA_GAME_ID = "luna"
PATHFINDER_GAME_ID = "pathfinder"


def load_lab_modules() -> tuple[Any, Any]:
    """Load only the dependency-light historical rules/evaluation modules."""

    package_name = "pathagon_lab"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(LAB_DIR)]
        package.__package__ = package_name
        sys.modules[package_name] = package

    def load(name: str) -> Any:
        full_name = f"{package_name}.{name}"
        if full_name in sys.modules:
            return sys.modules[full_name]
        path = LAB_DIR / f"{name}.py"
        spec = importlib.util.spec_from_file_location(full_name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        return module

    game = load("game")
    evaluation = load("evaluation")
    return game, evaluation


GAME, EVALUATION = load_lab_modules()
Action = GAME.Action
BoardConfig = GAME.BoardConfig
GameState = GAME.GameState
Player = GAME.Player
repetition_key = GAME.repetition_key
connection_distance = EVALUATION.connection_distance
largest_component = EVALUATION.largest_component
capture_opportunities = EVALUATION.capture_opportunities
edge_presence = EVALUATION.edge_presence


def count_bits(mask: int) -> int:
    return bin(mask).count("1")


def action_sort_key(action: Any) -> int:
    return action.to if action.kind == 0 else action.from_square * 10_000 + action.to


def immediate_winning_actions(state: Any, player: Any) -> list[Any]:
    view = state if state.turn is player else replace(state, turn=player)
    return [
        action
        for action in view.legal_actions()
        if view.apply_legal(action).winner is player
    ]


def pathfinder_eval(state: Any, player: Any) -> float:
    """Mirror the production Pathfinder evaluator with promoted weights."""

    opponent = player.other()
    if state.winner is player:
        return 1_000_000_000 - state.ply
    if state.winner is opponent:
        return -1_000_000_000 + state.ply
    path = connection_distance(state, opponent) - connection_distance(state, player)
    material = count_bits(state.pieces(player)) - count_bits(state.pieces(opponent))
    capture_direction = 1 if state.last_player is player else -1
    structure = largest_component(state, player) - largest_component(state, opponent)
    threats = capture_opportunities(state, player) - capture_opportunities(state, opponent)
    edges = edge_presence(state, player) - edge_presence(state, opponent)
    return (
        path * PATHFINDER_WEIGHTS["path"]
        + material * PATHFINDER_WEIGHTS["material"]
        + capture_direction * state.last_capture * PATHFINDER_WEIGHTS["capture"]
        + structure * PATHFINDER_WEIGHTS["structure"]
        + threats * PATHFINDER_WEIGHTS["threat"]
        + edges * PATHFINDER_WEIGHTS["edge"]
    )


def luna_eval(state: Any, player: Any) -> float:
    """Broad strategic score selected for the GPT-guided player.

    This deliberately emphasizes robust path distance and connected structure,
    while retaining material/capture/edge signals.  Immediate terminal wins
    and forced blocks are handled explicitly at the move-selection boundary.
    """

    opponent = player.other()
    if state.winner is player:
        return 1_000_000_000 - state.ply
    if state.winner is opponent:
        return -1_000_000_000 + state.ply
    path = connection_distance(state, opponent) - connection_distance(state, player)
    structure = largest_component(state, player) - largest_component(state, opponent)
    own_pieces = count_bits(state.pieces(player))
    opponent_pieces = count_bits(state.pieces(opponent))
    edges = edge_presence(state, player) - edge_presence(state, opponent)
    # The path and structure terms are intentionally larger than material:
    # Luna's stated priority is making and preserving a connection, not merely
    # accumulating pieces.
    return (
        path * 620
        + structure * 150
        + edges * 105
        + (own_pieces - opponent_pieces) * 18
        + (state.last_capture if state.last_player is player else -state.last_capture) * 180
    )


def luna_root_eval(state: Any, player: Any) -> float:
    """Add the GPT player's explicit tactical/capture preference at the root."""

    opponent = player.other()
    captures = capture_opportunities(state, player) - capture_opportunities(state, opponent)
    return luna_eval(state, player) + captures * 320


def ordered_actions(state: Any, root: Any, evaluator: Any, maximizing: bool, actions: Iterable[Any] | None = None) -> list[Any]:
    candidates = list(state.legal_actions() if actions is None else actions)
    scored = []
    for action in candidates:
        next_state = state.apply_legal(action)
        tactical = 2_000_000_000 if next_state.winner is state.turn else next_state.last_capture * 10_000
        scored.append((tactical + evaluator(next_state, root), action))
    if maximizing:
        scored.sort(key=lambda item: (item[0], -action_sort_key(item[1])), reverse=True)
    else:
        scored.sort(key=lambda item: (item[0], action_sort_key(item[1])))
    return [action for _score, action in scored]


class PathfinderAgent:
    """Dependency-light mirror of Rust's filter-aware Pathfinder control."""

    def __init__(self, depth: int, beam: int, nodes: int) -> None:
        self.depth = depth
        self.beam = beam
        self.node_budget = nodes
        self.nodes = 0
        self.exhausted = False
        self.completed_depth = 0
        self.table_hits = 0
        self._table: dict[tuple[Any, Any], tuple[int, float, str, Any]] = {}
        self._killers: dict[int, list[Any]] = {}
        self._history: dict[tuple[int, Any, bool], int] = {}

    def choose_action(self, state: Any, _rng: random.Random, _history: set[tuple]) -> Any | None:
        actions = self._root_actions(state)
        if not actions:
            return None
        self.nodes = 0
        self.exhausted = False
        self.completed_depth = 0
        self.table_hits = 0
        self._table = {}
        self._killers = {}
        self._history = {}
        root = state.turn
        best_action = actions[0]
        best_score = float("-inf")
        for search_depth in range(1, self.depth + 1):
            iteration_actions = list(actions)
            if best_action in iteration_actions:
                iteration_actions.remove(best_action)
                iteration_actions.insert(0, best_action)
            iteration_action = iteration_actions[0]
            iteration_score = float("-inf")
            alpha = float("-inf")
            complete = True
            for action in iteration_actions:
                if self.nodes >= self.node_budget:
                    self.exhausted = True
                    complete = False
                    break
                next_state = state.apply_legal(action)
                self.nodes += 1
                score = self._search(next_state, root, search_depth - 1, alpha, float("inf"), 1)
                if score > iteration_score or (
                    score == iteration_score and action_sort_key(action) < action_sort_key(iteration_action)
                ):
                    iteration_action, iteration_score = action, score
                alpha = max(alpha, iteration_score)
                if self.exhausted:
                    complete = False
                    break
            if not complete:
                break
            best_action, best_score = iteration_action, iteration_score
            self.completed_depth = search_depth
        if self.completed_depth == 0:
            best_score = pathfinder_eval(state.apply_legal(best_action), root)
        return best_action

    def _root_actions(self, state: Any) -> list[Any]:
        fallback = ordered_actions(state, state.turn, pathfinder_eval, True)
        if not fallback:
            return []
        opponent = state.turn.other()
        safe = []
        risky = False
        for action in fallback:
            next_state = state.apply_legal(action)
            allows_win = next_state.winner is not state.turn and bool(immediate_winning_actions(next_state, opponent))
            if allows_win:
                risky = True
            else:
                safe.append(action)
        if safe and risky:
            fallback = safe
        return fallback

    def _search(self, state: Any, root: Any, depth: int, alpha: float, beta: float, ply_from_root: int) -> float:
        if state.winner is not None:
            return pathfinder_eval(state, root)
        if depth == 0:
            return pathfinder_eval(state, root)
        if self.nodes >= self.node_budget:
            self.exhausted = True
            return pathfinder_eval(state, root)
        key = (state, root)
        original_alpha = alpha
        original_beta = beta
        preferred_action = None
        entry = self._table.get(key)
        if entry is not None:
            entry_depth, entry_score, bound, preferred_action = entry
            if entry_depth >= depth:
                self.table_hits += 1
                if bound == "exact":
                    return entry_score
                if bound == "lower":
                    alpha = max(alpha, entry_score)
                else:
                    beta = min(beta, entry_score)
                if alpha >= beta:
                    return entry_score
        maximizing = state.turn is root
        actions = ordered_actions(state, root, pathfinder_eval, maximizing)
        if preferred_action in actions:
            actions.remove(preferred_action)
            actions.insert(0, preferred_action)
        killers = self._killers.get(ply_from_root, [])
        for killer in reversed(killers):
            if killer in actions:
                actions.remove(killer)
                actions.insert(0, killer)
        actions.sort(key=lambda action: -self._history.get((ply_from_root, action, maximizing), 0))
        actions = actions[: self.beam]
        if not actions:
            return pathfinder_eval(state, root)
        best = float("-inf") if maximizing else float("inf")
        best_action = actions[0]
        for action in actions:
            next_state = state.apply_legal(action)
            self.nodes += 1
            score = self._search(next_state, root, depth - 1, alpha, beta, ply_from_root + 1)
            if maximizing:
                if score > best or (score == best and action_sort_key(action) < action_sort_key(best_action)):
                    best, best_action = score, action
                alpha = max(alpha, best)
            else:
                if score < best or (score == best and action_sort_key(action) < action_sort_key(best_action)):
                    best, best_action = score, action
                beta = min(beta, best)
            if beta <= alpha or self.nodes >= self.node_budget:
                if beta <= alpha and next_state.winner is None and next_state.last_capture == 0:
                    history_key = (ply_from_root, action, maximizing)
                    self._history[history_key] = min(
                        1_000_000,
                        self._history.get(history_key, 0) + max(1, depth * depth),
                    )
                    current_killers = self._killers.setdefault(ply_from_root, [])
                    if action not in current_killers:
                        current_killers.insert(0, action)
                        del current_killers[2:]
                break
        if not self.exhausted:
            bound = "upper" if best <= original_alpha else "lower" if best >= original_beta else "exact"
            self._table[key] = (depth, best, bound, best_action)
        if self.nodes >= self.node_budget:
            self.exhausted = True
        return best


class LunaAgent:
    """GPT-guided broad search with explicit tactical safety checks."""

    def __init__(self, depth: int, beam: int, nodes: int, root_beam: int) -> None:
        self.depth = depth
        self.beam = beam
        self.node_budget = nodes
        self.root_beam = root_beam
        self.nodes = 0
        self._eval_cache: dict[tuple[Any, Any], float] = {}

    def _evaluate(self, state: Any, player: Any) -> float:
        key = (state, player)
        cached = self._eval_cache.get(key)
        if cached is None:
            cached = luna_eval(state, player)
            self._eval_cache[key] = cached
        return cached

    def choose_action(self, state: Any, _rng: random.Random, history: set[tuple]) -> Any | None:
        legal = list(state.legal_actions())
        if not legal:
            return None
        self.nodes = 0
        root = state.turn
        safe = [action for action in legal if repetition_key(state.apply_legal(action)) not in history]
        if not safe:
            safe = legal
        # The tactical ordering score makes an immediate win sort above every
        # quiet action. This avoids a second full relocation-action scan just
        # to discover a terminal move.
        base_ordered = ordered_actions(state, root, self._evaluate, True, safe)
        if base_ordered and state.apply_legal(base_ordered[0]).winner is root:
            return base_ordered[0]
        # Match the promoted Pathfinder's root invariant: remove a move only
        # when it hands the opponent an immediate win and at least one safe
        # alternative exists. This is Luna's tactical floor, not a learned
        # signal, and it is intentionally not repeated at every search node.
        # During relocation the legal set can exceed 250 actions; scan a
        # bounded, strategically ordered prefix to keep bulk experiments
        # tractable while retaining exhaustive checks in placement positions.
        opponent = root.other()
        scan_limit = len(safe) if len(safe) <= 100 else max(self.root_beam * 2, 32)
        safe_tactical = []
        risky = False
        for action in base_ordered[:scan_limit]:
            next_state = state.apply_legal(action)
            allows_win = next_state.winner is not root and bool(immediate_winning_actions(next_state, opponent))
            if allows_win:
                risky = True
            else:
                safe_tactical.append(action)
        if risky and safe_tactical:
            candidates = safe_tactical + [action for action in base_ordered[scan_limit:] if action not in safe_tactical]
        else:
            candidates = safe
        ordered = ordered_actions(state, root, luna_root_eval, True, candidates)
        candidates = ordered[: min(self.root_beam, len(ordered))]
        best_action = candidates[0]
        best_score = float("-inf")
        alpha = float("-inf")
        for action in candidates:
            if self.nodes >= self.node_budget:
                break
            self.nodes += 1
            score = self._search(state.apply_legal(action), root, self.depth - 1, alpha, float("inf"), history)
            if score > best_score or (score == best_score and action_sort_key(action) < action_sort_key(best_action)):
                best_action, best_score = action, score
            alpha = max(alpha, best_score)
        return best_action

    def _search(self, state: Any, root: Any, depth: int, alpha: float, beta: float, history: set[tuple]) -> float:
        if state.winner is not None or depth <= 0 or self.nodes >= self.node_budget:
            return luna_eval(state, root)
        actions = [
            action for action in state.legal_actions()
            if repetition_key(state.apply_legal(action)) not in history
        ] or list(state.legal_actions())
        actions = ordered_actions(state, root, self._evaluate, state.turn is root, actions)[: self.beam]
        if not actions:
            return luna_eval(state, root)
        maximizing = state.turn is root
        best = float("-inf") if maximizing else float("inf")
        for action in actions:
            if self.nodes >= self.node_budget:
                break
            self.nodes += 1
            next_state = state.apply_legal(action)
            score = self._search(next_state, root, depth - 1, alpha, beta, history | {repetition_key(next_state)})
            if maximizing:
                best = max(best, score)
                alpha = max(alpha, best)
            else:
                best = min(best, score)
                beta = min(beta, best)
            if beta <= alpha:
                break
        return best


def action_json(action: Any) -> dict[str, Any]:
    return {"kind": "place", "to": action.to} if action.kind == 0 else {"kind": "relocate", "from": action.from_square, "to": action.to}


def play_game(
    light_agent: Any,
    dark_agent: Any,
    light_id: str,
    dark_id: str,
    config: Any,
    seed: int,
    opening_plies: int,
    max_plies: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    state = GameState.initial(config)
    repetitions: dict[tuple, int] = {}
    history: set[tuple] = set()
    moves: list[dict[str, Any]] = []
    reason = "max_plies"
    while state.winner is None and state.ply < max_plies:
        key = repetition_key(state)
        repetitions[key] = repetitions.get(key, 0) + 1
        history.add(key)
        if repetitions[key] >= 3:
            reason = "threefold_repetition"
            break
        legal = list(state.legal_actions())
        if not legal:
            reason = "no_legal_actions"
            break
        actor = "opening-random"
        active_agent = None
        if state.ply < opening_plies:
            action = rng.choice(legal)
        elif state.turn is Player.LIGHT:
            actor = light_id
            active_agent = light_agent
            action = active_agent.choose_action(state, rng, history)
        else:
            actor = dark_id
            active_agent = dark_agent
            action = active_agent.choose_action(state, rng, history)
        if action is None or action not in legal:
            reason = "no_legal_action_returned"
            break
        next_state = state.apply_legal(action)
        moves.append({
            "ply": state.ply + 1,
            "player": "light" if state.turn is Player.LIGHT else "dark",
            "actor": actor,
            "action": action_json(action),
            "captured": next_state.last_capture,
            "lunaNodes": active_agent.nodes if actor == "luna" else None,
            "pathfinderNodes": active_agent.nodes if actor == "pathfinder" else None,
        })
        state = next_state
    winner = None if state.winner is None else ("light" if state.winner is Player.LIGHT else "dark")
    return {
        "seed": seed,
        "agents": {"light": light_id, "dark": dark_id},
        "winner": winner,
        "result": "win" if winner else "draw",
        "reason": "connection" if winner else reason,
        "plies": len(moves),
        "moves": moves,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(record["winner"] == "light" for record in records)
    losses = sum(record["winner"] == "dark" for record in records)
    draws = len(records) - wins - losses
    return {"games": len(records), "wins": wins, "losses": losses, "draws": draws, "points": wins + draws * 0.5}


def luna_won(record: dict[str, Any]) -> bool:
    return (record["winner"] == "light" and record["agents"]["light"] == LUNA_GAME_ID) or (
        record["winner"] == "dark" and record["agents"]["dark"] == LUNA_GAME_ID
    )


def run_one_game(index: int, values: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    config = BoardConfig(values["size"], values["reserve"], values["max_plies"])
    luna = LunaAgent(values["luna_depth"], values["luna_beam"], values["luna_nodes"], values["luna_root_beam"])
    pathfinder = PathfinderAgent(values["pathfinder_depth"], values["pathfinder_beam"], values["pathfinder_nodes"])
    if index % 2 == 0:
        record = play_game(luna, pathfinder, LUNA_GAME_ID, PATHFINDER_GAME_ID, config, values["seed"] + index, values["opening_plies"], values["max_plies"])
    else:
        record = play_game(pathfinder, luna, PATHFINDER_GAME_ID, LUNA_GAME_ID, config, values["seed"] + index, values["opening_plies"], values["max_plies"])
    return index, record


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = BoardConfig(args.size, args.reserve, args.max_plies)
    values = {
        key: getattr(args, key)
        for key in (
            "size", "reserve", "max_plies", "luna_depth", "luna_beam", "luna_nodes",
            "luna_root_beam", "pathfinder_depth", "pathfinder_beam", "pathfinder_nodes",
            "seed", "opening_plies",
        )
    }
    records_by_index: dict[int, dict[str, Any]] = {}
    started = time.perf_counter()

    if args.workers == 1:
        completed = (run_one_game(index, values) for index in range(args.games))
        for index, record in completed:
            records_by_index[index] = record
            finished = len(records_by_index)
            if finished % args.progress_every == 0 or finished == args.games:
                elapsed = time.perf_counter() - started
                luna_wins = sum(luna_won(r) for r in records_by_index.values())
                draws = sum(r["winner"] is None for r in records_by_index.values())
                print(f"{finished}/{args.games}: Luna {luna_wins}-{finished - luna_wins - draws}-{draws} ({elapsed:.1f}s)", flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_one_game, index, values) for index in range(args.games)]
            for future in concurrent.futures.as_completed(futures):
                index, record = future.result()
                records_by_index[index] = record
                finished = len(records_by_index)
                if finished % args.progress_every == 0 or finished == args.games:
                    elapsed = time.perf_counter() - started
                    luna_wins = sum(luna_won(r) for r in records_by_index.values())
                    draws = sum(r["winner"] is None for r in records_by_index.values())
                    print(f"{finished}/{args.games}: Luna {luna_wins}-{finished - luna_wins - draws}-{draws} ({elapsed:.1f}s)", flush=True)

    records = [records_by_index[index] for index in range(args.games)]
    luna_summary = {
        "games": len(records),
        "wins": sum(
            luna_won(r)
            for r in records
        ),
        "losses": sum(
            not luna_won(r) and r["winner"] is not None
            for r in records
        ),
        "draws": sum(r["winner"] is None for r in records),
    }
    luna_summary["points"] = luna_summary["wins"] + 0.5 * luna_summary["draws"]
    pathfinder_summary = {
        "games": len(records),
        "wins": luna_summary["losses"],
        "losses": luna_summary["wins"],
        "draws": luna_summary["draws"],
    }
    pathfinder_summary["points"] = pathfinder_summary["wins"] + 0.5 * pathfinder_summary["draws"]
    elapsed = time.perf_counter() - started
    return {
        "schemaVersion": 1,
        "mode": "luna-vs-pathfinder-depth",
        "protocol": {
            "boardSize": args.size,
            "reservePerPlayer": config.reserve_per_player,
            "maxPlies": args.max_plies,
            "openingRandomPlies": args.opening_plies,
            "seed": args.seed,
            "colorBalanced": True,
            "workers": args.workers,
        },
        "luna": {
            "id": "luna-general-gpt-guided",
            "description": "GPT-selected strategic policy; not a learned checkpoint",
            "depth": args.luna_depth,
            "beam": args.luna_beam,
            "rootBeam": args.luna_root_beam,
            "nodes": args.luna_nodes,
        },
        "pathfinder": {
            "id": "pathfinder-v0.5.0-trained-evaluator-mirror",
            "depth": args.pathfinder_depth,
            "beam": args.pathfinder_beam,
            "nodes": args.pathfinder_nodes,
            "weights": PATHFINDER_WEIGHTS,
            "tacticalRootFilter": True,
        },
        "elapsedSeconds": round(elapsed, 6),
        "lunaSummary": luna_summary,
        "pathfinderSummary": pathfinder_summary,
        "games": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026082800)
    parser.add_argument("--size", type=int, default=7)
    parser.add_argument("--reserve", type=int, default=14)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--opening-plies", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--luna-depth", type=int, default=2)
    parser.add_argument("--luna-beam", type=int, default=10)
    parser.add_argument("--luna-root-beam", type=int, default=16)
    parser.add_argument("--luna-nodes", type=int, default=1500)
    parser.add_argument("--pathfinder-depth", type=int, default=2)
    parser.add_argument("--pathfinder-beam", type=int, default=8)
    parser.add_argument("--pathfinder-nodes", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.games < 1 or args.games % 2:
        parser.error("--games must be a positive even number for color balance")
    if args.max_plies < 1 or args.opening_plies < 0 or args.workers < 1:
        parser.error("--max-plies must be positive and --opening-plies cannot be negative")
    if args.out is None:
        args.out = DEFAULT_WORKSPACE / f"depth-{args.pathfinder_depth}-{args.games}.json"
    report = run(args)
    output = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "games"}, sort_keys=True))


if __name__ == "__main__":
    main()
