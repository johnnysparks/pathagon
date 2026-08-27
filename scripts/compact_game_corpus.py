#!/usr/bin/env python3
"""Build a content-addressed, Git-friendly corpus from legacy game archives.

The canonical game identity contains only rules/configuration and the move
sequence. Run metadata is stored as a separate observation keyed by the game.
Python-specific policy, Q-target, and search arrays stay in their source
archive and are intentionally excluded from the durable corpus.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
ALPHABET_INDEX = {character: index for index, character in enumerate(ALPHABET)}
SHARD_COUNT = 64
GAME_HEADER = "# g1-key\trules\tsize\treserve\trepetition\tplies\t2-char-actions\n"
OBSERVATION_HEADER = (
    "# g1-key\tsource\tseed64\tlight\tlight-model\tdark\tdark-model\twinner\treason\tmax-plies\n"
)
SOURCE_HEADER = "# source\tpath\tbytes\trecords\tunique-observations\n"


class RecordError(ValueError):
    """A candidate looked like a game record but could not be normalized."""


@dataclass(frozen=True)
class CanonicalGame:
    key: str
    shard: int
    rules: str
    size: int
    reserve: int
    repetition: int
    plies: int
    actions: str


@dataclass(frozen=True)
class Observation:
    game_key: str
    shard: int
    source_id: str
    seed: str
    light: str
    light_model: str
    dark: str
    dark_model: str
    winner: str
    reason: str
    max_plies: str


def encode_digest(payload: bytes, prefix: str) -> tuple[str, int]:
    digest = hashlib.sha256(payload).digest()
    encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"{prefix}_{encoded}", digest[0] >> 2


def encode_radix(value: int) -> str:
    if value < 0:
        raise RecordError("seed must be non-negative")
    if value == 0:
        return "0"
    digits: list[str] = []
    while value:
        digits.append(ALPHABET[value & 63])
        value >>= 6
    return "".join(reversed(digits))


def decode_radix(value: str) -> int:
    result = 0
    for character in value:
        try:
            digit = ALPHABET_INDEX[character]
        except KeyError as error:
            raise RecordError(f"invalid radix64 character {character!r}") from error
        result = (result << 6) | digit
    return result


def encode_action(action: object, size: int) -> str:
    if not isinstance(action, dict):
        raise RecordError("move action is not an object")
    kind = action.get("kind", action.get("type"))
    cell_count = size * size
    if kind == "place":
        destination = integer_field(action.get("to"), "place destination")
        if not 0 <= destination < cell_count:
            raise RecordError(f"place destination {destination} is outside {size}x{size}")
        code = destination
    elif kind == "relocate":
        source = integer_field(action.get("from"), "relocation source")
        destination = integer_field(action.get("to"), "relocation destination")
        if not 0 <= source < cell_count or not 0 <= destination < cell_count:
            raise RecordError(f"relocation {source}->{destination} is outside {size}x{size}")
        # The existing Pathagon p1 encoding reserves the first 49 values for
        # placements and uses a fixed 49-cell stride for every supported size.
        code = 49 + source * 49 + destination
    else:
        raise RecordError(f"unknown action kind {kind!r}")
    if code >= 4096:
        raise RecordError(f"action code {code} exceeds the 12-bit encoding")
    return ALPHABET[code >> 6] + ALPHABET[code & 63]


def integer_field(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise RecordError(f"{label} is not an integer")
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise RecordError(f"{label} is not an integer") from error
    return result


def infer_size(record: dict[str, Any], source: Path) -> int:
    config = record.get("config") if isinstance(record.get("config"), dict) else {}
    value = record.get("boardSize", config.get("boardSize"))
    if value is None:
        lowered = source.as_posix().lower()
        for size in range(3, 8):
            if f"{size}x{size}" in lowered:
                return size
        return 7
    size = integer_field(value, "board size")
    if not 1 <= size <= 7:
        raise RecordError(f"unsupported board size {size}")
    return size


def normalize_record(record: dict[str, Any], source: Path, source_id: str) -> tuple[CanonicalGame, Observation]:
    moves = record.get("moves")
    if not isinstance(moves, list):
        raise RecordError("moves is not an array")
    size = infer_size(record, source)
    config = record.get("config") if isinstance(record.get("config"), dict) else {}
    reserve = integer_field(
        record.get("reservePerPlayer", config.get("reservePerPlayer", 2 * size)),
        "reserve per player",
    )
    repetition = integer_field(config.get("repetitionLimit", record.get("repetitionLimit", 3)), "repetition limit")
    rules = safe_field(config.get("rulesVersion", record.get("rulesVersion", "pathagon-rules-v1")))
    actions = "".join(encode_action(move.get("action") if isinstance(move, dict) else None, size) for move in moves)
    identity = f"g1\0{rules}\0{size}\0{reserve}\0{repetition}\0{actions}".encode("utf-8")
    key, shard = encode_digest(identity, "g1")
    game = CanonicalGame(key, shard, rules, size, reserve, repetition, len(moves), actions)

    seed_value = record.get("seed")
    seed = "-" if seed_value is None else encode_radix(integer_field(seed_value, "seed"))
    agents = record.get("agents") if isinstance(record.get("agents"), dict) else {}
    max_plies_value = record.get("maxPlies", config.get("maxPlies"))
    max_plies = "-" if max_plies_value is None else str(integer_field(max_plies_value, "max plies"))
    observation = Observation(
        key,
        shard,
        source_id,
        seed,
        safe_field(agents.get("light", "-")),
        model_identity(record, "light"),
        safe_field(agents.get("dark", "-")),
        model_identity(record, "dark"),
        winner_code(record.get("winner")),
        reason_code(record.get("reason")),
        max_plies,
    )
    return game, observation


def normalize_p1(line: str, source: Path, source_id: str) -> tuple[CanonicalGame, Observation]:
    fields = line.rstrip("\n").split("\t")
    if len(fields) != 7 or fields[0] != "p1":
        raise RecordError("invalid p1 row")
    _, seed, light, dark, winner, reason, actions = fields
    if len(actions) % 2:
        raise RecordError("p1 action stream has odd length")
    for character in actions:
        if character not in ALPHABET_INDEX:
            raise RecordError(f"invalid p1 action character {character!r}")
    # The p1 format is defined by the existing Rust exporter only for 7x7/14.
    size, reserve, repetition, rules = 7, 14, 3, "pathagon-rules-v1"
    identity = f"g1\0{rules}\0{size}\0{reserve}\0{repetition}\0{actions}".encode("utf-8")
    key, shard = encode_digest(identity, "g1")
    game = CanonicalGame(key, shard, rules, size, reserve, repetition, len(actions) // 2, actions)
    decode_radix(seed)
    observation = Observation(
        key,
        shard,
        source_id,
        seed,
        safe_field(light),
        "-",
        safe_field(dark),
        "-",
        winner_code(winner),
        reason_code(reason),
        "-",
    )
    return game, observation


def safe_field(value: object) -> str:
    text = str(value)
    return text.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def winner_code(value: object) -> str:
    mapping = {"light": "L", "dark": "D", None: "-", "L": "L", "D": "D", "-": "-"}
    return mapping.get(value, "?")


def reason_code(value: object) -> str:
    mapping = {
        "path": "P",
        "threefold-repetition": "R",
        "max-plies": "M",
        "no-legal-action": "N",
        "P": "P",
        "R": "R",
        "M": "M",
        "N": "N",
    }
    return mapping.get(value, "?")


def model_identity(record: dict[str, Any], player: str) -> str:
    specifications = record.get("agentSpecifications")
    if not isinstance(specifications, dict):
        return "-"
    specification = specifications.get(player)
    if not isinstance(specification, dict):
        return "-"
    manifest = specification.get("manifest")
    candidates: list[object] = []
    if isinstance(manifest, dict):
        candidates.extend(
            manifest.get(field)
            for field in ("modelHash", "checkpointHash", "weightsHash", "model", "checkpoint")
        )
    candidates.extend(specification.get(field) for field in ("modelHash", "checkpointHash"))
    for candidate in candidates:
        if candidate not in (None, ""):
            return safe_field(candidate)
    return "-"


def looks_like_record(value: object) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("moves"), list):
        return False
    moves = value["moves"]
    return not moves or all(isinstance(move, dict) and "action" in move for move in moves)


def records_from_value(value: object) -> Iterator[dict[str, Any]]:
    if looks_like_record(value):
        yield value  # type: ignore[misc]
        return
    if isinstance(value, dict):
        nested = value.get("record")
        if isinstance(nested, str):
            try:
                nested = json.loads(nested)
            except json.JSONDecodeError:
                nested = None
        if nested is not None:
            yield from records_from_value(nested)
        for key, child in value.items():
            if key == "record" or key in {"moves", "actionValues", "actionVisits", "policy"}:
                continue
            if isinstance(child, (dict, list)):
                yield from records_from_value(child)
    elif isinstance(value, list):
        for child in value:
            if isinstance(child, (dict, list)):
                yield from records_from_value(child)


def json_records(path: Path) -> Iterator[dict[str, Any]]:
    if path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield from json_lines(handle, path)
    elif path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            yield from json_lines(handle, path)
    else:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        yield from records_from_value(value)


def json_lines(handle: TextIO, path: Path) -> Iterator[dict[str, Any]]:
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RecordError(f"line {line_number}: invalid JSON: {error}") from error
        yield from records_from_value(value)


def candidate_files(inputs: list[Path], output: Path) -> list[Path]:
    candidates: set[Path] = set()
    resolved_output = output.resolve()
    for input_path in inputs:
        if input_path.is_file():
            paths: Iterable[Path] = (input_path,)
        elif input_path.is_dir():
            paths = input_path.rglob("*")
        else:
            raise FileNotFoundError(input_path)
        for path in paths:
            if not path.is_file():
                continue
            try:
                path.resolve().relative_to(resolved_output)
                continue
            except ValueError:
                pass
            if path.name == "games.tsv" or path.suffix == ".p1" or path.suffix in {".json", ".jsonl"} or path.name.endswith(".jsonl.gz"):
                candidates.add(path)
    return sorted(candidates, key=lambda path: path.as_posix())


def display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def source_identifier(path_text: str) -> str:
    key, _ = encode_digest(f"source-v1\0{path_text}".encode("utf-8"), "s1")
    return key


def initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE games (
            key TEXT PRIMARY KEY, shard INTEGER NOT NULL, rules TEXT NOT NULL,
            size INTEGER NOT NULL, reserve INTEGER NOT NULL, repetition INTEGER NOT NULL,
            plies INTEGER NOT NULL, actions TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE observations (
            game_key TEXT NOT NULL, shard INTEGER NOT NULL, source_id TEXT NOT NULL,
            seed TEXT NOT NULL, light TEXT NOT NULL, light_model TEXT NOT NULL,
            dark TEXT NOT NULL, dark_model TEXT NOT NULL, winner TEXT NOT NULL,
            reason TEXT NOT NULL, max_plies TEXT NOT NULL,
            PRIMARY KEY (game_key, source_id, seed, light, light_model, dark, dark_model, winner, reason, max_plies)
        ) WITHOUT ROWID;
        CREATE TABLE sources (
            source_id TEXT PRIMARY KEY, path TEXT NOT NULL, bytes INTEGER NOT NULL,
            records INTEGER NOT NULL, observations INTEGER NOT NULL
        ) WITHOUT ROWID;
        CREATE TABLE errors (
            path TEXT NOT NULL, message TEXT NOT NULL,
            PRIMARY KEY (path, message)
        ) WITHOUT ROWID;
        """
    )


def game_shard(key: str) -> int:
    if not key.startswith("g1_"):
        raise RecordError(f"invalid canonical game key {key!r}")
    encoded = key.removeprefix("g1_")
    try:
        digest = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
    except (ValueError, TypeError) as error:
        raise RecordError(f"invalid canonical game key {key!r}") from error
    if len(digest) != hashlib.sha256().digest_size:
        raise RecordError(f"invalid canonical game key {key!r}")
    return digest[0] >> 2


def load_existing_corpus(connection: sqlite3.Connection, corpus: Path) -> dict[str, int]:
    """Seed an update from the durable corpus before reading disposable inputs."""
    games = 0
    observations = 0
    sources = 0
    games_dir = corpus / "games"
    observations_dir = corpus / "observations"
    if not games_dir.is_dir() or not observations_dir.is_dir() or not (corpus / "sources.tsv").is_file():
        raise FileNotFoundError(f"existing corpus is incomplete: {corpus}")
    for shard_path in sorted(games_dir.glob("games-*.tsv")):
        with shard_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 7:
                    raise RecordError(f"{shard_path}:{line_number}: invalid canonical game row")
                key, rules, size, reserve, repetition, plies, actions = fields
                connection.execute(
                    "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        key,
                        game_shard(key),
                        rules,
                        integer_field(size, "board size"),
                        integer_field(reserve, "reserve"),
                        integer_field(repetition, "repetition"),
                        integer_field(plies, "plies"),
                        actions,
                    ),
                )
                games += 1
    for shard_path in sorted(observations_dir.glob("observations-*.tsv")):
        with shard_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 10:
                    raise RecordError(f"{shard_path}:{line_number}: invalid canonical observation row")
                key = fields[0]
                connection.execute(
                    "INSERT INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (key, game_shard(key), *fields[1:]),
                )
                observations += 1
    with (corpus / "sources.tsv").open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise RecordError(f"{corpus / 'sources.tsv'}:{line_number}: invalid source row")
            source_id, path, size, records, source_observations = fields
            connection.execute(
                "INSERT INTO sources VALUES (?, ?, ?, ?, ?)",
                (
                    source_id,
                    path,
                    integer_field(size, "source bytes"),
                    integer_field(records, "source records"),
                    integer_field(source_observations, "source observations"),
                ),
            )
            sources += 1
    connection.commit()
    return {"games": games, "observations": observations, "sources": sources}


def insert_pair(connection: sqlite3.Connection, game: CanonicalGame, observation: Observation) -> bool:
    connection.execute(
        "INSERT OR IGNORE INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (game.key, game.shard, game.rules, game.size, game.reserve, game.repetition, game.plies, game.actions),
    )
    cursor = connection.execute(
        "INSERT OR IGNORE INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            observation.game_key,
            observation.shard,
            observation.source_id,
            observation.seed,
            observation.light,
            observation.light_model,
            observation.dark,
            observation.dark_model,
            observation.winner,
            observation.reason,
            observation.max_plies,
        ),
    )
    return cursor.rowcount > 0


def ingest_source(connection: sqlite3.Connection, path: Path, root: Path) -> tuple[int, int]:
    path_text = display_path(path, root)
    source_id = source_identifier(path_text)
    records = 0
    observations = 0
    try:
        if path.name == "games.tsv" or path.suffix == ".p1":
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip() or line.startswith("#"):
                        continue
                    try:
                        game, observation = normalize_p1(line, path, source_id)
                    except RecordError as error:
                        connection.execute(
                            "INSERT OR IGNORE INTO errors VALUES (?, ?)",
                            (path_text, f"line {line_number}: {error}"),
                        )
                        continue
                    observations += int(insert_pair(connection, game, observation))
                    records += 1
        else:
            for record_number, record in enumerate(json_records(path), start=1):
                try:
                    game, observation = normalize_record(record, path, source_id)
                except RecordError as error:
                    connection.execute(
                        "INSERT OR IGNORE INTO errors VALUES (?, ?)",
                        (path_text, f"record {record_number}: {error}"),
                    )
                    continue
                observations += int(insert_pair(connection, game, observation))
                records += 1
    except (OSError, UnicodeError, json.JSONDecodeError, RecordError) as error:
        connection.execute("INSERT OR IGNORE INTO errors VALUES (?, ?)", (path_text, str(error)))
    if records:
        connection.execute(
            "INSERT OR REPLACE INTO sources VALUES (?, ?, ?, ?, ?)",
            (source_id, path_text, path.stat().st_size, records, observations),
        )
    connection.commit()
    return records, observations


def write_outputs(connection: sqlite3.Connection, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    games_dir = output / "games"
    observations_dir = output / "observations"
    games_dir.mkdir()
    observations_dir.mkdir()
    for shard in range(SHARD_COUNT):
        with (games_dir / f"games-{shard:02x}.tsv").open("w", encoding="utf-8") as handle:
            handle.write(GAME_HEADER)
            for row in connection.execute(
                "SELECT key, rules, size, reserve, repetition, plies, actions FROM games WHERE shard = ? ORDER BY key",
                (shard,),
            ):
                handle.write("\t".join(map(str, row)) + "\n")
        with (observations_dir / f"observations-{shard:02x}.tsv").open("w", encoding="utf-8") as handle:
            handle.write(OBSERVATION_HEADER)
            for row in connection.execute(
                """SELECT game_key, source_id, seed, light, light_model, dark, dark_model, winner, reason, max_plies
                   FROM observations WHERE shard = ? ORDER BY game_key, source_id, seed, light, light_model, dark, dark_model""",
                (shard,),
            ):
                handle.write("\t".join(map(str, row)) + "\n")

    with (output / "sources.tsv").open("w", encoding="utf-8") as handle:
        handle.write(SOURCE_HEADER)
        for row in connection.execute("SELECT source_id, path, bytes, records, observations FROM sources ORDER BY path"):
            handle.write("\t".join(map(str, row)) + "\n")

    errors = [{"path": path, "message": message} for path, message in connection.execute("SELECT path, message FROM errors ORDER BY path")]
    if errors:
        with (output / "errors.jsonl").open("w", encoding="utf-8") as handle:
            for error in errors:
                handle.write(json.dumps(error, sort_keys=True, separators=(",", ":")) + "\n")

    game_count = connection.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    observation_count = connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    source_count = connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    record_count = connection.execute("SELECT COALESCE(SUM(records), 0) FROM sources").fetchone()[0]
    sizes = {str(size): count for size, count in connection.execute("SELECT size, COUNT(*) FROM games GROUP BY size ORDER BY size")}
    manifest = {
        "schemaVersion": 1,
        "gameKey": "sha256-base64url(rules,size,reserve,repetition,actions)",
        "actionEncoding": "base64url-12bit-fixed-49-stride",
        "shards": SHARD_COUNT,
        "games": game_count,
        "observations": observation_count,
        "recordsFound": record_count,
        "duplicateRecords": record_count - game_count,
        "sourceFilesWithGames": source_count,
        "errors": len(errors),
        "gamesByBoardSize": sizes,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "README.md").write_text(readme_text(), encoding="utf-8")
    return manifest


def readme_text() -> str:
    return """# Unified compact game corpus v1

This corpus is generated from historical Pathagon archives by
`scripts/compact_game_corpus.py`. It is deterministic, content-addressed, and
split into small shards for reviewable Git diffs.

## Identity

A `g1_...` key is the SHA-256 digest of the rules version, board size, reserve,
repetition limit, and 12-bit encoded action sequence. Seeds, agents, engines,
outcomes, source paths, and search/training annotations do not affect identity.

`games/` stores each unique game once. `observations/` associates source and
run metadata with a game key. `sources.tsv` maps compact source IDs back to the
original archive paths.

Observations retain the agent ID and model/checkpoint hash for each color when
available, plus outcome and minimal provenance. Policy tensors, Q arrays,
visits, and search scores are not part of the game table. Useful universal
targets may be promoted separately into versioned, game-keyed corpus sidecars.
Every game state can be reconstructed by replaying the canonical action
sequence in Rust.

## Rebuild

```bash
python3 scripts/compact_game_corpus.py \
  --input research/runs \
  --output research/corpora/games-v1
```

The command refuses to replace an existing output directory unless
`--replace` is provided. When the output exists, it is loaded as the durable
base before new inputs are scanned; use `--no-base` only for an intentional
from-scratch rebuild. Parse or normalization errors produce `errors.jsonl` and
a nonzero exit unless `--allow-errors` is explicitly selected.
"""


def replace_output(staged: Path, output: Path, replace: bool) -> None:
    if output.exists():
        if not replace:
            raise FileExistsError(f"output already exists: {output}; pass --replace to rebuild")
        if output.is_symlink() or not output.is_dir():
            raise ValueError(f"refusing to replace non-directory output: {output}")
        shutil.rmtree(output)
    os.replace(staged, output)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, default=[Path("research/runs")])
    parser.add_argument("--output", type=Path, default=Path("research/corpora/games-v1"))
    parser.add_argument("--replace", action="store_true", help="replace an existing output after a successful rebuild")
    parser.add_argument("--no-base", action="store_true", help="ignore an existing output and rebuild only from inputs")
    parser.add_argument("--allow-errors", action="store_true", help="write the partial corpus even if sources fail to parse")
    parser.add_argument("--progress-every", type=int, default=500, help="report progress after this many candidate files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path.cwd()
    inputs = [path.resolve() for path in args.input]
    output = args.output.resolve()
    files = candidate_files(inputs, output)
    print(json.dumps({"phase": "scan", "candidateFiles": len(files)}, sort_keys=True), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="compact-games-", dir=output.parent) as temporary:
        temporary_path = Path(temporary)
        connection = sqlite3.connect(temporary_path / "index.sqlite3")
        initialize_database(connection)
        if output.exists() and not args.no_base:
            seeded = load_existing_corpus(connection, output)
            print(json.dumps({"phase": "base", **seeded}, sort_keys=True), flush=True)
        total_records = 0
        for index, path in enumerate(files, start=1):
            records, _observations = ingest_source(connection, path, root)
            total_records += records
            if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(files)):
                print(
                    json.dumps(
                        {"phase": "ingest", "files": index, "candidateFiles": len(files), "records": total_records},
                        sort_keys=True,
                    ),
                    flush=True,
                )
        error_count = connection.execute("SELECT COUNT(*) FROM errors").fetchone()[0]
        staged = temporary_path / "output"
        manifest = write_outputs(connection, staged)
        connection.close()
        if error_count and not args.allow_errors:
            errors_path = staged / "errors.jsonl"
            for line in errors_path.read_text(encoding="utf-8").splitlines()[:20]:
                print(line, file=sys.stderr)
            print(f"conversion found {error_count} source errors; rerun with --allow-errors only after review", file=sys.stderr)
            return 2
        replace_output(staged, output, args.replace)
    print(json.dumps({"phase": "complete", "output": str(output), **manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
