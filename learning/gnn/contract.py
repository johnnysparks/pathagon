"""Pathagon contract v1 values shared with the TypeScript and Rust runtimes."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping

CONTRACT_VERSION = 1
RULES_VERSION = "pathagon-rules-v1"
TERMINATION_REASONS = {"path", "threefold-repetition", "max-plies", "no-legal-action"}
PLAYER_VALUES = {"light", "dark"}
DEFAULT_EVALUATOR_WEIGHTS = {
    "path": 240,
    "material": 110,
    "capture": 700,
    "structure": 55,
    "threat": 130,
    "edge": 80,
}


def game_config(size: int = 7, reserve: int = 14, max_plies: int = 180, repetition_limit: int = 3) -> dict:
    value = {
        "rulesVersion": RULES_VERSION,
        "boardSize": size,
        "reservePerPlayer": reserve,
        "maxPlies": max_plies,
        "repetitionLimit": repetition_limit,
    }
    validate_game_config(value)
    return value


def engine_metadata(engine_id: str, runtime: str, version: str = "1.0.0") -> dict:
    value = {"id": engine_id, "runtime": runtime, "version": version, "rulesVersion": RULES_VERSION}
    validate_engine_metadata(value)
    return value


def agent_manifest(
    runtime: str = "python",
    evaluator_weights: dict | None = None,
    depth: int = 0,
    node_budget: int = 0,
    beam: int = 0,
    model_hash: str | None = None,
) -> dict:
    value = {
        "manifestVersion": 1,
        "runtime": runtime,
        "rulesVersion": RULES_VERSION,
        "evaluatorWeights": {**DEFAULT_EVALUATOR_WEIGHTS, **(evaluator_weights or {})},
        "depth": depth,
        "nodeBudget": node_budget,
        "beam": beam,
        "modelHash": model_hash,
    }
    validate_agent_manifest(value)
    return value


def agent_specification(
    agent_id: str,
    name: str,
    version: str,
    kind: str,
    engine_id: str,
    parameters: dict | None = None,
    manifest: dict | None = None,
) -> dict:
    value = {"id": agent_id, "name": name, "version": version, "kind": kind, "engineId": engine_id}
    if manifest is None:
        runtime = "python" if "python" in engine_id else "rust" if "rust" in engine_id else "typescript"
        manifest = agent_manifest(runtime=runtime)
    value["manifest"] = manifest
    if parameters is not None:
        value["parameters"] = parameters
    validate_agent_specification(value)
    return value


def validate_game_config(value: Any) -> dict:
    _record(value, "game config")
    if value.get("rulesVersion") != RULES_VERSION:
        raise ValueError("unsupported Pathagon rules version")
    _integer_range(value.get("boardSize"), 3, 8, "board size")
    _integer_range(value.get("reservePerPlayer"), 1, 64, "reserve per player")
    _integer_range(value.get("maxPlies"), 1, 4096, "maximum plies")
    if value.get("repetitionLimit") != 3:
        raise ValueError("Pathagon repetition limit must be 3")
    return dict(value)


def validate_action(value: Any, board_size: int = 8) -> dict:
    _record(value, "action")
    kind = value.get("kind")
    if kind == "place" and _in_range(value.get("to"), 0, board_size * board_size - 1):
        return {"kind": "place", "to": int(value["to"])}
    if kind == "relocate" and _in_range(value.get("from"), 0, board_size * board_size - 1) and _in_range(value.get("to"), 0, board_size * board_size - 1):
        return {"kind": "relocate", "from": int(value["from"]), "to": int(value["to"])}
    raise ValueError("invalid action")


def validate_position(value: Any) -> dict:
    _record(value, "position")
    if value.get("contractVersion") != CONTRACT_VERSION:
        raise ValueError("unsupported position contract version")
    config = validate_game_config(value.get("config"))
    cells = config["boardSize"] ** 2
    board = value.get("board")
    if not isinstance(board, list) or len(board) != cells or any(piece not in (None, "light", "dark") for piece in board):
        raise ValueError("invalid board")
    if value.get("turn") not in PLAYER_VALUES or value.get("winner") not in PLAYER_VALUES | {None}:
        raise ValueError("invalid position player")
    _squares(value.get("forbidden"), cells, "forbidden")
    markers = value.get("lastRelocatedTo")
    _record(markers, "last relocation markers")
    for player in PLAYER_VALUES:
        if markers.get(player) is not None and not _in_range(markers.get(player), 0, cells - 1):
            raise ValueError("invalid relocation square")
    _integer_range(value.get("ply"), 0, config["maxPlies"], "position ply")
    reserve = value.get("reserve")
    _record(reserve, "reserve")
    _integer_range(reserve.get("light"), 0, 255, "light reserve")
    _integer_range(reserve.get("dark"), 0, 255, "dark reserve")
    return dict(value)


def validate_engine_metadata(value: Any) -> dict:
    _record(value, "engine metadata")
    _field(value.get("id"), "engine ID")
    _field(value.get("version"), "engine version")
    if value.get("runtime") not in {"typescript", "rust", "python"} or value.get("rulesVersion") != RULES_VERSION:
        raise ValueError("invalid engine metadata")
    return dict(value)


def validate_agent_specification(value: Any) -> dict:
    _record(value, "agent specification")
    for field in ("id", "name", "version", "engineId"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"invalid agent {field}")
    if value.get("kind") not in {"random", "heuristic", "search", "learned", "puct"}:
        raise ValueError("invalid agent kind")
    validate_agent_manifest(value.get("manifest"))
    if "parameters" in value and not isinstance(value["parameters"], dict):
        raise ValueError("invalid agent parameters")
    return dict(value)


def validate_agent_manifest(value: Any) -> dict:
    _record(value, "agent manifest")
    if value.get("manifestVersion") != 1 or value.get("runtime") not in {"typescript", "rust", "python"} or value.get("rulesVersion") != RULES_VERSION:
        raise ValueError("invalid agent manifest metadata")
    weights = value.get("evaluatorWeights")
    _record(weights, "evaluator weights")
    for name in DEFAULT_EVALUATOR_WEIGHTS:
        _integer_range(weights.get(name), -2_147_483_648, 2_147_483_647, f"evaluator weight {name}")
    for name in ("depth", "nodeBudget", "beam"):
        if not _in_range(value.get(name), 0, 4_294_967_295):
            raise ValueError(f"invalid agent {name}")
    model_hash = value.get("modelHash")
    if model_hash is not None and (not isinstance(model_hash, str) or re.fullmatch(r"sha256:[A-Fa-f0-9]{64}", model_hash) is None):
        raise ValueError("invalid agent model hash")
    return dict(value)


def validate_replay_record(value: Any) -> dict:
    _record(value, "replay")
    if value.get("contractVersion") != CONTRACT_VERSION:
        raise ValueError("unsupported replay contract version")
    config = validate_game_config(value.get("config"))
    validate_engine_metadata(value.get("engine"))
    agents = value.get("agents")
    specs = value.get("agentSpecifications")
    _record(agents, "replay agents")
    _record(specs, "agent specifications")
    for player in PLAYER_VALUES:
        _field(agents.get(player), f"{player} agent ID")
        validate_agent_specification(specs.get(player))
        if agents[player] != specs[player]["id"]:
            raise ValueError("agent ID does not match specification")
    winner = value.get("winner")
    if winner not in PLAYER_VALUES | {None} or value.get("result") != ("win" if winner else "draw"):
        raise ValueError("replay result does not match winner")
    if value.get("reason") not in TERMINATION_REASONS:
        raise ValueError("invalid termination reason")
    plies = value.get("plies")
    _integer_range(plies, 0, config["maxPlies"], "replay plies")
    moves = value.get("moves")
    if not isinstance(moves, list) or len(moves) != plies:
        raise ValueError("replay plies do not match moves")
    cells = config["boardSize"] ** 2
    for index, move in enumerate(moves, start=1):
        _record(move, f"move {index}")
        if move.get("ply") != index or move.get("player") not in PLAYER_VALUES:
            raise ValueError(f"invalid move {index}")
        validate_action(move.get("action"), config["boardSize"])
        _squares(move.get("captured"), cells, "captured")
        for field in ("nodes", "completedDepth", "tableHits"):
            if not isinstance(move.get(field), int) or move[field] < 0:
                raise ValueError(f"invalid move {field}")
    return dict(value)


def _record(value: Any, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"invalid {label}")


def _field(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError(f"invalid {label}")


def _in_range(value: Any, minimum: int, maximum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum


def _integer_range(value: Any, minimum: int, maximum: int, label: str) -> None:
    if not _in_range(value, minimum, maximum):
        raise ValueError(f"invalid {label}")


def _squares(value: Any, cells: int, label: str) -> None:
    if not isinstance(value, list) or len(set(value)) != len(value) or any(not _in_range(square, 0, cells - 1) for square in value):
        raise ValueError(f"invalid {label} squares")
