#!/usr/bin/env python3
"""Build the persistent historyless golden table from every local replay archive."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = PROJECT_ROOT / "research/20260824-gnn-cnn-lab"
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from python.data import initial_state_from_record  # noqa: E402
from python.game import Action, BoardConfig, GameState, Player, action_from_record  # noqa: E402
from python.golden import (  # noqa: E402
    DRAW,
    GoldenTable,
    LOSS,
    WIN,
    key_bytes_for_board_size,
    rows_sha256,
)


ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
BOARD_SIZE = 7
RESERVE = 14
SUPPORTED_RULES = "pathagon-rules-v1"


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def decode_action(
    token: str,
    cells: int = BOARD_SIZE * BOARD_SIZE,
    encoding_cells: int = BOARD_SIZE * BOARD_SIZE,
) -> Action:
    """Decode a fixed-width corpus token for a configured board.

    Corpus v1 keeps a 49-cell stride so every action remains a two-byte token
    even for historical 5x5 games. The configured board size is a validation
    boundary, not the encoding stride.
    """

    if len(token) != 2:
        raise ValueError("action token must be exactly two bytes")
    try:
        code = (ALPHABET.index(token[0]) << 6) | ALPHABET.index(token[1])
    except ValueError as error:
        raise ValueError(f"invalid action token {token!r}") from error
    if code < encoding_cells:
        return Action.place(code)
    relocation = code - encoding_cells
    from_square, to = divmod(relocation, encoding_cells)
    if from_square >= cells:
        raise ValueError(f"relocation token is outside a board with {cells} cells")
    if to >= cells:
        raise ValueError(f"relocation token is outside a board with {cells} cells")
    return Action.relocate(from_square, to)


def iter_corpus_games(corpus: Path) -> Iterator[tuple[Path, int, list[str]]]:
    for path in sorted((corpus / "games").glob("games-*.tsv")):
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 7:
                    raise ValueError(f"{path}:{line_number}: expected seven fields")
                yield path, line_number, fields


def _extract_game_records(value: Any) -> Iterator[dict[str, Any]]:
    """Yield full replay records from JSONL records and nested ladder wrappers."""

    if not isinstance(value, dict):
        return
    nested_record = value.get("record")
    if isinstance(nested_record, dict):
        yield from _extract_game_records(nested_record)
        return
    if isinstance(nested_record, str):
        try:
            decoded = json.loads(nested_record)
        except json.JSONDecodeError:
            return
        yield from _extract_game_records(decoded)
        return
    if isinstance(value.get("moves"), list):
        yield value
        return
    # Root-aware seeded sidecars retain compact actions rather than expanded
    # moves. They are still replayable and their terminal winner is derivable.
    if isinstance(value.get("actions"), str) and isinstance(value.get("initialPosition"), dict):
        yield value
        return
    games = value.get("games")
    if isinstance(games, list):
        for game in games:
            yield from _extract_game_records(game)


def iter_json_game_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    if path.suffix == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: invalid JSON: {error}") from error
        yield from ((1, record) for record in _extract_game_records(value))
        return
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            yield from ((line_number, record) for record in _extract_game_records(value))


def iter_json_sources(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        if "golden" in path.parts or "fixtures" in path.parts:
            continue
        yield path


def _record_root(record: dict[str, Any]) -> dict[str, Any] | None:
    initial = record.get("initialPosition")
    if isinstance(initial, dict):
        return initial
    provenance = record.get("provenance")
    if isinstance(provenance, dict) and isinstance(provenance.get("rootPosition"), dict):
        return provenance["rootPosition"]
    return None


def _record_config(record: dict[str, Any], root: dict[str, Any] | None) -> BoardConfig:
    raw_config = record.get("config") if isinstance(record.get("config"), dict) else {}
    root_config = root.get("config", {}) if isinstance(root, dict) and isinstance(root.get("config"), dict) else {}

    def value(name: str, default: int) -> int:
        raw = raw_config.get(name, root_config.get(name, record.get(name, default)))
        return int(raw)

    return BoardConfig(
        size=value("boardSize", BOARD_SIZE),
        reserve_per_player=value("reservePerPlayer", RESERVE),
        ply_limit=value("maxPlies", 0),
    )


def _record_actions(record: dict[str, Any], cells: int) -> list[Action]:
    if isinstance(record.get("moves"), list):
        return [action_from_record(move["action"]) for move in record["moves"]]
    action_stream = record.get("actions")
    if not isinstance(action_stream, str) or len(action_stream) % 2:
        raise ValueError("replay record has no even-length action stream")
    return [decode_action(action_stream[offset : offset + 2], cells) for offset in range(0, len(action_stream), 2)]


def _replay(
    record: dict[str, Any],
    config: BoardConfig,
    root: dict[str, Any] | None,
    actions: list[Action],
) -> GameState:
    if root is None:
        state = GameState.initial(config)
    else:
        # The Python loader already owns seeded-root validation. Copying the
        # root into the expected field lets it consume provenance.rootPosition
        # from older runs without changing the contract on disk.
        replay_record = {"initialPosition": root}
        state = initial_state_from_record(replay_record, config)
    for action in actions:
        if action not in state.legal_actions():
            raise ValueError(f"illegal action {action.short()} at replay ply {state.ply}")
        state = state.apply_legal(action)
    declared = record.get("winner")
    actual = None if state.winner is None else ("light" if state.winner is Player.LIGHT else "dark")
    if declared in {"light", "dark"} and actual != declared:
        raise ValueError(f"declared winner {declared!r} does not match replay winner {actual!r}")
    return state


def _classification(state: GameState, record: dict[str, Any] | None = None) -> str:
    if state.winner is not None:
        return "terminal-path-win"
    reason = record.get("reason") if record is not None else None
    legal = state.legal_actions()
    if reason == "no-legal-action":
        if legal:
            raise ValueError("record declares no-legal-action but replay still has legal actions")
        return "no-legal-action-draw"
    if not legal:
        return "no-legal-action-draw"
    if reason in {"threefold-repetition", "max-plies"}:
        return "history-dependent-draw"
    return "non-terminal"


def _identity(config: BoardConfig, root: dict[str, Any] | None, actions: Iterable[Action]) -> str:
    payload = {
        "size": config.size,
        "reserve": config.reserve_per_player,
        "root": root,
        "actions": [[action.kind, action.from_square, action.to] for action in actions],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _source_meta(path: Path, format_name: str) -> dict[str, Any]:
    return {
        "path": relative_path(path),
        "format": format_name,
        "bytes": path.stat().st_size,
        "sha256": rows_sha256(path),
        "records": 0,
        "uniqueReplays": 0,
        "duplicateReplays": 0,
        "terminalPathWins": 0,
        "noLegalActionDraws": 0,
        "historyDependentDraws": 0,
        "nonTerminal": 0,
        "unsupported": 0,
    }


def _table_for_config(
    config: BoardConfig,
    tables: dict[tuple[int, int], GoldenTable],
) -> GoldenTable:
    namespace = (config.size, config.reserve_per_player)
    table = tables.get(namespace)
    if table is None:
        table = GoldenTable(board_size=config.size, reserve_per_player=config.reserve_per_player)
        tables[namespace] = table
    return table


def _process_replay(
    *,
    path: Path,
    line_number: int,
    record: dict[str, Any],
    source: dict[str, Any],
    tables: dict[tuple[int, int], GoldenTable],
    replay_cache: dict[str, tuple[GameState, str]],
) -> None:
    source["records"] += 1
    root = _record_root(record)
    config = _record_config(record, root)
    if config.size < 3 or config.reserve_per_player < 1:
        source["unsupported"] += 1
        return
    table = _table_for_config(config, tables)
    actions = _record_actions(record, config.cell_count)
    if record.get("plies") is not None and int(record["plies"]) != len(actions):
        raise ValueError(f"replay plies do not match actions")
    identity = _identity(config, root, actions)
    cached = replay_cache.get(identity)
    if cached is None:
        state = _replay(record, config, root, actions)
        classification = _classification(state, record)
        replay_cache[identity] = (state, classification)
        source["uniqueReplays"] += 1
    else:
        state, classification = cached
        source["duplicateReplays"] += 1
        # A duplicate action sequence may come from a record with a different
        # metadata wrapper; still check any declared winner independently.
        declared = record.get("winner")
        actual = None if state.winner is None else ("light" if state.winner is Player.LIGHT else "dark")
        if declared in {"light", "dark"} and actual != declared:
            raise ValueError("declared winner disagrees with cached replay")

    if classification == "terminal-path-win":
        source["terminalPathWins"] += 1
        outcome = WIN if state.winner == state.turn else LOSS
        table.put(state, outcome)
    elif classification == "no-legal-action-draw":
        source["noLegalActionDraws"] += 1
        table.put(state, DRAW)
    elif classification == "history-dependent-draw":
        source["historyDependentDraws"] += 1
    else:
        source["nonTerminal"] += 1


def build_table(
    root: Path,
    corpus: Path,
    output: Path,
    manifest_path: Path,
    inventory_path: Path,
    allow_errors: bool = False,
) -> dict[str, Any]:
    tables: dict[tuple[int, int], GoldenTable] = {}
    replay_cache: dict[str, tuple[GameState, str]] = {}
    sources: list[dict[str, Any]] = []
    errors: list[str] = []

    source_by_path: dict[Path, dict[str, Any]] = {}
    corpus_game_paths = sorted((corpus / "games").glob("games-*.tsv"))
    for path in corpus_game_paths:
        source = _source_meta(path, "canonical-game-tsv")
        sources.append(source)
        source_by_path[path] = source
    try:
        for path, line_number, fields in iter_corpus_games(corpus):
            _, rules, size, reserve, _repetition, plies, action_stream = fields
            source = source_by_path[path]
            source["records"] += 1
            if rules != SUPPORTED_RULES:
                source["unsupported"] += 1
                continue
            config = BoardConfig(size=int(size), reserve_per_player=int(reserve))
            actions = [
                decode_action(action_stream[offset : offset + 2], config.cell_count)
                for offset in range(0, len(action_stream), 2)
            ]
            if len(actions) != int(plies):
                raise ValueError(f"{path}:{line_number}: action count does not match plies")
            identity = _identity(config, None, actions)
            cached = replay_cache.get(identity)
            if cached is None:
                state = _replay({}, config, None, actions)
                classification = _classification(state)
                replay_cache[identity] = (state, classification)
                source["uniqueReplays"] += 1
            else:
                state, classification = cached
                source["duplicateReplays"] += 1
            if classification == "terminal-path-win":
                source["terminalPathWins"] += 1
                _table_for_config(config, tables).put(state, WIN if state.winner == state.turn else LOSS)
            elif classification == "no-legal-action-draw":
                source["noLegalActionDraws"] += 1
                _table_for_config(config, tables).put(state, DRAW)
            elif classification == "history-dependent-draw":
                source["historyDependentDraws"] += 1
            else:
                source["nonTerminal"] += 1
    except (TypeError, ValueError, KeyError) as error:
        errors.append(str(error))

    json_paths = list(iter_json_sources(root))
    for path in json_paths:
        try:
            records = list(iter_json_game_records(path))
        except ValueError as error:
            errors.append(str(error))
            continue
        if not records:
            continue
        source = _source_meta(path, "jsonl" if path.suffix == ".jsonl" else "json")
        sources.append(source)
        for line_number, record in records:
            try:
                _process_replay(
                    path=path,
                    line_number=line_number,
                    record=record,
                    source=source,
                    tables=tables,
                    replay_cache=replay_cache,
                )
            except (TypeError, ValueError, KeyError) as error:
                errors.append(f"{path}:{line_number}: {error}")

    if errors and not allow_errors:
        preview = "\n".join(errors[:20])
        suffix = "" if len(errors) <= 20 else f"\n... and {len(errors) - 20} more"
        raise RuntimeError(f"golden ingestion found {len(errors)} replay errors:\n{preview}{suffix}")

    output.parent.mkdir(parents=True, exist_ok=True)
    shard_specs: list[dict[str, Any]] = []
    primary_namespace = (BOARD_SIZE, RESERVE)
    if primary_namespace not in tables:
        tables[primary_namespace] = GoldenTable(board_size=BOARD_SIZE, reserve_per_player=RESERVE)

    table_root = output.parent.parent
    for (board_size, reserve), table in sorted(tables.items()):
        if (board_size, reserve) == primary_namespace:
            shard_path = output
        else:
            shard_path = table_root / f"{board_size}x{board_size}-r{reserve}" / output.name
        rows = table.write(shard_path)
        shard_specs.append(
            {
                "path": relative_path(shard_path),
                "boardSize": board_size,
                "reservePerPlayer": reserve,
                "keyBytes": key_bytes_for_board_size(board_size),
                "rowBytes": key_bytes_for_board_size(board_size) + 1,
                "rows": rows,
                "bytes": shard_path.stat().st_size,
                "sha256": rows_sha256(shard_path),
                "source": "all-discoverable-replay-terminal-truths",
            }
        )
    totals = {
        "sourceFiles": len(sources),
        "recordsSeen": sum(source["records"] for source in sources),
        "uniqueReplays": len(replay_cache),
        "terminalPathWins": sum(source["terminalPathWins"] for source in sources),
        "noLegalActionDraws": sum(source["noLegalActionDraws"] for source in sources),
        "historyDependentDraws": sum(source["historyDependentDraws"] for source in sources),
        "nonTerminal": sum(source["nonTerminal"] for source in sources),
        "unsupported": sum(source["unsupported"] for source in sources),
        "replayErrors": len(errors),
        "rows": sum(spec["rows"] for spec in shard_specs),
        "namespaces": len(shard_specs),
    }
    inventory = {
        "schemaVersion": 1,
        "scope": "research replay archives excluding data/golden and data/fixtures",
        "discovery": {
            "researchRoot": relative_path(root),
            "canonicalCorpus": relative_path(corpus),
            "scannedJsonArchives": len(json_paths),
            "scannedCanonicalGameShards": len(corpus_game_paths),
            "excludedDirectories": [
                "data/golden",
                "data/fixtures",
            ],
            "includedFormats": ["json", "jsonl", "canonical-game-tsv"],
        },
        "sources": sorted(sources, key=lambda source: source["path"]),
        "totals": totals,
    }
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"] = {
        "kind": "all-discoverable-replay-terminal-truths",
        "scope": inventory["scope"],
        "corpus": relative_path(corpus),
        "selection": "all replay-bearing JSON/JSONL archives plus canonical corpus game shards; path wins and no-legal-action draws only",
        "inventory": {
            "path": relative_path(inventory_path),
            "bytes": inventory_path.stat().st_size,
            "sha256": rows_sha256(inventory_path),
        },
    }
    manifest.setdefault("key", {})["namespace"] = "boardSize+reservePerPlayer"
    manifest["counts"] = totals
    manifest["shards"] = shard_specs
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {
        **totals,
        "bytes": output.stat().st_size,
        "sha256": rows_sha256(output),
        "inventory": relative_path(inventory_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "research",
        help="research root to scan for replay-bearing JSON/JSONL archives",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=PROJECT_ROOT / "data/corpora/games-v1",
        help="canonical action-only corpus directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/golden/tables/historyless-wdl-v1/7x7-r14/shard-00.bin",
        help="persistent golden shard to write",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/golden/manifest.json",
        help="golden manifest to update",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=PROJECT_ROOT / "data/golden/source-inventory.json",
        help="source inventory to write",
    )
    parser.add_argument(
        "--allow-errors",
        action="store_true",
        help="write a table while recording replay errors in the inventory",
    )
    args = parser.parse_args()
    summary = build_table(
        root=args.root,
        corpus=args.corpus,
        output=args.output,
        manifest_path=args.manifest,
        inventory_path=args.inventory,
        allow_errors=args.allow_errors,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
