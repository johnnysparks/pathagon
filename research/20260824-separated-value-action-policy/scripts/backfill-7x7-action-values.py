#!/usr/bin/env python3
"""Backfill root-Q/action-value targets onto existing 7x7 replay games.

The source games are replayed exactly; only the per-position search targets
are added. The checkpoint is selected from each record's model hash, and the
source JSONL files are never modified. Outputs are gzip-compressed JSONL so a
large backfill remains practical to archive and move between machines.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import io
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from research.gnn.contract import ROOT_Q_SOURCE
from research.gnn.game import BoardConfig, GameState, Player, action_from_record, repetition_key
from research.gnn.mcts import PUCTSearch
from research.gnn.train import choose_device, load_model, model_state_hash


DEFAULT_CHECKPOINTS = (
    REPO_ROOT / "research/runs/gnn/benchmark-7x7/small-gnn-warmstart.pt",
    REPO_ROOT / "research/runs/gnn/benchmark-7x7/gnn-warmstart.pt",
    REPO_ROOT / "research/runs/gnn/benchmark-7x7/cnn-warmstart.pt",
    REPO_ROOT / "research/runs/gnn/benchmark-7x7/generated/batch-20260824-neural-reval-20260824/reval-gnn-30k.pt",
    REPO_ROOT / "research/runs/gnn/benchmark-7x7/generated/batch-20260824-neural-reval-20260824/reval-cnn-30k.pt",
)
_WORKER_CHECKPOINTS = None
_WORKER_SIMULATIONS = 0
_WORKER_ROOT_NOISE = False


def open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def jsonl_paths(input_dirs: list[Path], input_files: list[Path]) -> list[Path]:
    paths = list(input_files)
    for directory in input_dirs:
        for pattern in ("*.jsonl", "*.jsonl.gz"):
            paths.extend(sorted(directory.glob(pattern)))
    unique = {path.resolve(): path for path in paths}
    return sorted(unique.values())


def record_lines(path: Path):
    with open_text(path, "rt") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict) or not isinstance(record.get("moves"), list):
                raise RuntimeError(f"{path}:{line_number}: expected a replay record with moves")
            yield line_number, record


def model_hashes(record: dict) -> set[str]:
    hashes = set()
    for player in ("light", "dark"):
        manifest = record.get("agentSpecifications", {}).get(player, {}).get("manifest", {})
        model_hash = manifest.get("modelHash")
        if model_hash:
            hashes.add(model_hash)
    return hashes


def complete_q_target(move: dict) -> bool:
    values = move.get("actionValues")
    visits = move.get("actionVisits")
    return (
        isinstance(values, list)
        and isinstance(visits, list)
        and move.get("actionValueSource") == ROOT_Q_SOURCE
        and len(values) == len(visits)
        and len(values) > 0
    )


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_checkpoint_map(paths: list[Path], device: str) -> dict[str, tuple[Path, object]]:
    selected_device = choose_device(device)
    checkpoints: dict[str, tuple[Path, object]] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint: {path}")
        model = load_model(path, selected_device)
        model.eval()
        model_hash = model_state_hash(model)
        checkpoints[model_hash] = (path, model)
        print(f"checkpoint: {relative_path(path)} -> {model_hash}", flush=True)
    return checkpoints


def initialize_worker(checkpoint_paths: list[Path], device: str, simulations: int, root_noise: bool) -> None:
    global _WORKER_CHECKPOINTS, _WORKER_SIMULATIONS, _WORKER_ROOT_NOISE
    import torch

    torch.set_num_threads(1)
    _WORKER_CHECKPOINTS = load_checkpoint_map(checkpoint_paths, device)
    _WORKER_SIMULATIONS = simulations
    _WORKER_ROOT_NOISE = root_noise


def backfill_worker(record: dict) -> tuple[dict, int, int, str]:
    if _WORKER_CHECKPOINTS is None:
        raise RuntimeError("backfill worker was not initialized")
    return backfill_record(record, _WORKER_CHECKPOINTS, _WORKER_SIMULATIONS, _WORKER_ROOT_NOISE)


def backfill_record(
    record: dict,
    checkpoints: dict[str, tuple[Path, object]],
    simulations: int,
    root_noise: bool,
) -> tuple[dict, int, int, str]:
    hashes = model_hashes(record)
    if len(hashes) != 1:
        raise RuntimeError(f"seed {record.get('seed')}: expected one model hash, found {sorted(hashes)}")
    model_hash = next(iter(hashes))
    if model_hash not in checkpoints:
        raise RuntimeError(f"seed {record.get('seed')}: no checkpoint loaded for {model_hash}")
    checkpoint_path, model = checkpoints[model_hash]
    config_value = record.get("config", {})
    config = BoardConfig(
        size=int(config_value.get("boardSize", 7)),
        reserve_per_player=int(config_value.get("reservePerPlayer", 14)),
        ply_limit=int(config_value.get("maxPlies", 196)),
    )
    if config.size != 7 or config.reserve_per_player != 14:
        raise RuntimeError(f"seed {record.get('seed')}: expected a 7x7 reserve-14 replay")

    state = GameState.initial(config)
    repetitions: dict[tuple, int] = {}
    search = PUCTSearch(model, simulations=simulations)
    added_positions = 0
    existing_positions = 0
    for move in record["moves"]:
        position = repetition_key(state)
        repetitions[position] = repetitions.get(position, 0) + 1
        action = action_from_record(move["action"])
        actions = tuple(state.legal_actions())
        if action not in actions:
            raise RuntimeError(f"seed {record.get('seed')}: illegal action {action.short()} at ply {state.ply}")
        if complete_q_target(move):
            existing_positions += 1
        else:
            root, search_actions, _ = search.run(
                state,
                add_root_noise=root_noise,
                history=set(repetitions),
            )
            if tuple(search_actions) != actions:
                raise RuntimeError(f"seed {record.get('seed')}: search action order diverged at ply {state.ply}")
            values, visits = search.root_action_values(root, list(actions))
            move["actionValues"] = values
            move["actionVisits"] = visits
            move["actionValueSource"] = ROOT_Q_SOURCE
            added_positions += 1
        state = state.apply_legal(action)

    actual_winner = None if state.winner is None else ("light" if state.winner is Player.LIGHT else "dark")
    if actual_winner != record.get("winner"):
        raise RuntimeError(
            f"seed {record.get('seed')}: winner mismatch after replay ({actual_winner!r} != {record.get('winner')!r})"
        )
    record["qBackfill"] = {
        "source": ROOT_Q_SOURCE,
        "simulations": simulations,
        "rootNoise": root_noise,
        "checkpoint": relative_path(checkpoint_path),
        "modelHash": model_hash,
    }
    return record, added_positions, existing_positions, model_hash


def write_gzip_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")


def backfill_file(
    source: Path,
    output: Path,
    checkpoints: dict[str, tuple[Path, object]],
    simulations: int,
    root_noise: bool,
    progress_every: int,
    limit_games: int,
    workers: int,
    checkpoint_paths: list[Path],
) -> dict:
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    games = positions = added_positions = existing_positions = 0
    models = Counter()

    source_records = [record for _, record in record_lines(source)]
    if limit_games:
        source_records = source_records[:limit_games]

    def results():
        if workers == 1:
            for record in source_records:
                yield backfill_record(record, checkpoints, simulations, root_noise)
            return
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=initialize_worker,
            initargs=(checkpoint_paths, str(choose_device("cpu")), simulations, root_noise),
        ) as executor:
            yield from executor.map(backfill_worker, source_records, chunksize=1)

    def records():
        nonlocal games, positions, added_positions, existing_positions
        for record, added, existing, model_hash in results():
            games += 1
            positions += len(record["moves"])
            added_positions += added
            existing_positions += existing
            models[model_hash] += 1
            if games == 1 or games % progress_every == 0:
                elapsed = max(0.001, time.monotonic() - started)
                print(
                    f"{source.name}: {games} games / {positions} positions "
                    f"({positions / elapsed:.1f} positions/s)",
                    flush=True,
                )
            yield record

    write_gzip_jsonl(output, records())
    return {
        "input": relative_path(source),
        "output": relative_path(output),
        "games": games,
        "positions": positions,
        "qPositionsAdded": added_positions,
        "qPositionsAlreadyPresent": existing_positions,
        "modelHashes": dict(sorted(models.items())),
        "elapsedSeconds": round(time.monotonic() - started, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path, default=[])
    parser.add_argument("--input", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", type=Path, default=[])
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--root-noise", action="store_true", help="include Dirichlet noise in the root search")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--limit-games", type=int, default=0, help="process only the first N games per input (for smoke tests)")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        raise SystemExit("provide at least one --input or --input-dir")
    if args.simulations < 1 or args.progress_every < 1 or args.limit_games < 0 or args.workers < 1:
        raise SystemExit("simulations, progress interval, and workers must be positive; game limit cannot be negative")
    sources = jsonl_paths(args.input_dir, args.input)
    if not sources:
        raise SystemExit("no JSONL input files found")
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    checkpoint_paths = args.checkpoint or list(DEFAULT_CHECKPOINTS)
    checkpoint_paths = [path if path.is_absolute() else REPO_ROOT / path for path in checkpoint_paths]
    checkpoints = load_checkpoint_map(checkpoint_paths, args.device)

    results = []
    for source in sources:
        base_name = source.name.removesuffix(".jsonl.gz") if source.name.endswith(".jsonl.gz") else source.stem
        output = output_dir / f"{base_name}.jsonl.gz"
        if output.exists():
            raise SystemExit(f"refusing to overwrite existing output: {output}")
        results.append(
            backfill_file(
                source,
                output,
                checkpoints,
                args.simulations,
                args.root_noise,
                args.progress_every,
                args.limit_games,
                args.workers,
                checkpoint_paths,
            )
        )

    manifest = {
        "schemaVersion": 1,
        "status": "complete",
        "boardSize": 7,
        "reservePerPlayer": 14,
        "actionValueSource": ROOT_Q_SOURCE,
        "simulations": args.simulations,
        "rootNoise": args.root_noise,
        "device": str(choose_device(args.device)),
        "files": results,
        "games": sum(result["games"] for result in results),
        "positions": sum(result["positions"] for result in results),
        "qPositionsAdded": sum(result["qPositionsAdded"] for result in results),
        "qPositionsAlreadyPresent": sum(result["qPositionsAlreadyPresent"] for result in results),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    main()
