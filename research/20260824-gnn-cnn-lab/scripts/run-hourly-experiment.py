#!/usr/bin/env python3
"""Run one bounded, reproducible Pathagon learning-lab round.

Each round is deliberately isolated and append-only:

1. generate a fresh, provenance-stamped 7x7 policy/value self-play slice;
2. generate a separate higher-budget root-Q/action-value slice;
3. rebuild a deduplicated train/held-out corpus without consuming prior hourly
   reports or league games as training data;
4. clean-train several policy/value architecture lanes and a Q/Advantage
   checkpoint from their separate corpora;
5. score every candidate on the held-out split; and
6. run the current league roster plus the new candidates through a
   color-rotated round-robin.

The runner never overwrites a checkpoint or promotes a model to the browser.
It is safe to invoke from launchd, cron, or manually. A directory lock makes
an overlapping hourly invocation a no-op.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
DEFAULT_RUN_ROOT = REPO_ROOT / "research/runs/gnn/hourly"
DEFAULT_VENV_PYTHON = REPO_ROOT / ".venv-pathagon-gnn/bin/python"
PLAYERS = ("scout", "learner", "cnn")
ROOT_Q_SOURCE = "mcts-root-q-v1"
DEFAULT_QADV_CHECKPOINT = REPO_ROOT / "research/runs/gnn/benchmark-7x7/generated/batch-20260824-qadv-128-pilot-20260824/qadv-arbiter-7x7-v0.1.0.pt"

ARCHITECTURES = {
    "full-gnn": {
        "architecture": "gnn",
        "hidden": 64,
        "layers": 8,
        "cnn_blocks": None,
        "description": "full residual mean-message-passing GNN",
    },
    "compact-gnn": {
        "architecture": "gnn",
        "hidden": 32,
        "layers": 4,
        "cnn_blocks": None,
        "description": "compact GNN capacity ablation",
    },
    "cnn": {
        "architecture": "cnn",
        "hidden": 64,
        "layers": None,
        "cnn_blocks": 4,
        "description": "fixed-7x7 residual CNN",
    },
    "compact-cnn": {
        "architecture": "cnn",
        "hidden": 32,
        "layers": None,
        "cnn_blocks": 2,
        "description": "compact CNN capacity ablation",
    },
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp(value: datetime | None = None) -> str:
    return (value or utc_now()).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON in {path}: {error}") from error


@contextmanager
def run_lock(path: Path) -> Iterator[bool]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError:
        yield False
        return
    try:
        write_json(path / "owner.json", {"pid": os.getpid(), "startedAtUtc": utc_now().isoformat()})
        yield True
    finally:
        shutil.rmtree(path, ignore_errors=True)


def choose_python(explicit: str | None) -> str:
    if explicit:
        return explicit
    if DEFAULT_VENV_PYTHON.is_file():
        return str(DEFAULT_VENV_PYTHON)
    return sys.executable


def run_command(
    command: list[str],
    cwd: Path,
    log_path: Path,
    timeout_seconds: int,
) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    display = " ".join(subprocess.list2cmdline([item]) for item in command)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {display}\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=cwd,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        log.write(completed.stdout)
        log.write(f"\n[exit={completed.returncode} elapsed={time.monotonic() - started:.1f}s]\n")
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-30:])
        raise RuntimeError(f"command failed with exit code {completed.returncode}:\n{tail}")
    return completed.stdout


def last_json_line(output: str) -> dict:
    for line in reversed(output.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("command completed without a JSON result line")


def inspect_qadv_batch(path: Path) -> dict:
    """Require every generated game to carry complete root-Q targets."""
    paths = sorted(path.glob("*.jsonl"))
    if not paths:
        raise RuntimeError(f"Q/Advantage generation produced no JSONL files under {path}")
    games = positions = q_positions = 0
    incomplete_games = 0
    for source in paths:
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                moves = record.get("moves", [])
                games += 1
                positions += len(moves)
                complete = all(
                    isinstance(move, dict)
                    and isinstance(move.get("actionValues"), list)
                    and isinstance(move.get("actionVisits"), list)
                    and move.get("actionValueSource") == ROOT_Q_SOURCE
                    and len(move["actionValues"]) == len(move["actionVisits"])
                    and len(move["actionValues"]) > 0
                    for move in moves
                )
                if complete:
                    q_positions += len(moves)
                else:
                    incomplete_games += 1
    if incomplete_games:
        raise RuntimeError(
            f"Q/Advantage generation produced {incomplete_games} games without complete {ROOT_Q_SOURCE} targets"
        )
    return {
        "files": len(paths),
        "games": games,
        "positions": positions,
        "qPositions": q_positions,
        "actionValueSource": ROOT_Q_SOURCE,
    }


def parse_names(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in names if item not in allowed]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown {label}: {', '.join(unknown)}")
    if not names:
        raise argparse.ArgumentTypeError(f"at least one {label} is required")
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError(f"{label} must not contain duplicates")
    return names


def seed_for_round(round_number: int) -> int:
    # Keep hourly seeds disjoint while staying inside the game's u32 seed range.
    return 2_026_000_000 + round_number * 10_000


def checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def train_qadv_candidate(
    *,
    python: str,
    data_path: Path,
    resume_path: Path,
    output_path: Path,
    steps: int,
    learning_rate: float,
    seed: int,
    heldout_fraction: float,
    device: str,
    log_path: Path,
    timeout_seconds: int,
) -> dict:
    if not resume_path.is_file():
        raise RuntimeError(f"missing Q/Advantage warm-start checkpoint: {resume_path}")
    command = [
        python,
        "-m",
        "research.gnn.train",
        "qadv",
        "--data",
        str(data_path),
        "--resume",
        str(resume_path),
        "--out",
        str(output_path),
        "--size",
        "7",
        "--steps",
        str(steps),
        "--learning-rate",
        str(learning_rate),
        "--heldout-fraction",
        str(heldout_fraction),
        "--seed",
        str(seed),
        "--device",
        device,
        "--agent-id",
        "qadv-arbiter-7x7-hourly",
        "--agent-name",
        "The Q-Arbiter · Hourly Q/Advantage",
    ]
    result = last_json_line(run_command(command, REPO_ROOT, log_path, timeout_seconds))
    result.update(
        {
            "checkpoint": str(output_path.relative_to(REPO_ROOT)),
            "modelHash": checkpoint_hash(output_path),
            "warmStart": str(resume_path.relative_to(REPO_ROOT)),
        }
    )
    return result


def train_candidate(
    *,
    python: str,
    architecture_name: str,
    config: dict,
    data_path: Path,
    output_path: Path,
    steps: int,
    seed: int,
    device: str,
    log_path: Path,
    timeout_seconds: int,
) -> dict:
    command = [
        python,
        "-m",
        "research.gnn.train",
        "warmstart",
        "--data",
        str(data_path),
        "--out",
        str(output_path),
        "--architecture",
        str(config["architecture"]),
        "--size",
        "7",
        "--reserve",
        "14",
        "--hidden",
        str(config["hidden"]),
        "--steps",
        str(steps),
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    if config["layers"] is not None:
        command.extend(["--layers", str(config["layers"])])
    if config["cnn_blocks"] is not None:
        command.extend(["--cnn-blocks", str(config["cnn_blocks"])])
    result = last_json_line(run_command(command, REPO_ROOT, log_path, timeout_seconds))
    result.update(
        {
            "name": architecture_name,
            "description": config["description"],
            "checkpoint": str(output_path.relative_to(REPO_ROOT)),
            "modelHash": checkpoint_hash(output_path),
        }
    )
    return result


def evaluate_candidate(
    *,
    python: str,
    candidate: dict,
    heldout_path: Path,
    seed: int,
    device: str,
    max_examples: int,
    log_path: Path,
    timeout_seconds: int,
) -> dict:
    command = [
        python,
        "scripts/evaluate-7x7-checkpoint.py",
        "--checkpoint",
        str(REPO_ROOT / candidate["checkpoint"]),
        "--data",
        str(heldout_path),
        "--size",
        "7",
        "--reserve",
        "14",
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    if max_examples:
        command.extend(["--max-examples", str(max_examples)])
    result = last_json_line(run_command(command, REPO_ROOT, log_path, timeout_seconds))
    return {"candidate": candidate["name"], "checkpoint": candidate["checkpoint"], **result}


def build_candidate_agent(path: Path, label: str, simulations: int, device):
    # Imports are delayed so corpus-only failures still leave a useful report.
    from research.gnn.contract import agent_manifest
    from research.gnn.league import AgentSpec, GNNAgent, checkpoint_hash as league_checkpoint_hash, load_model

    model = load_model(path, device)
    model.eval()
    model_hash = league_checkpoint_hash(path)
    return AgentSpec(
        id=f"hourly-{path.stem}",
        label=label,
        kind="gnn",
        choose=GNNAgent(model, simulations=simulations),
        manifest=agent_manifest(runtime="python", node_budget=simulations, model_hash=model_hash),
    )


def run_league(
    *,
    round_number: int,
    seed: int,
    candidate_paths: list[tuple[str, Path]],
    output_path: Path,
    games_per_match: int,
    simulations: int,
    device_name: str,
    max_pairings: int,
) -> dict:
    import torch

    from research.gnn.game import BoardConfig
    from research.gnn.league import build_roster, play_game, summarize, update_elo
    from research.gnn.train import choose_device

    device = choose_device(device_name)
    board = BoardConfig(7, 14)
    roster = build_roster(7, 14, simulations, device)
    for name, path in candidate_paths:
        roster.append(build_candidate_agent(path, f"Hourly candidate · {name}", simulations, device))

    ratings = {agent.id: 1_000.0 for agent in roster}
    records: list[dict] = []
    head_to_head: list[dict] = []
    pairings = [
        (left_index, right_index)
        for left_index in range(len(roster))
        for right_index in range(left_index + 1, len(roster))
    ]
    if max_pairings:
        pairings = pairings[:max_pairings]
    for pair_index, (left_index, right_index) in enumerate(pairings):
        left = roster[left_index]
        right = roster[right_index]
        matchup: list[dict] = []
        for game_index in range(games_per_match):
            # The round number rotates colors between hourly runs, so a
            # one-game-per-pair budget still balances over time.
            left_is_light = (round_number + pair_index + game_index) % 2 == 0
            light, dark = (left, right) if left_is_light else (right, left)
            record = play_game(light, dark, board, seed + pair_index * 100 + game_index)
            matchup.append(record)
            records.append(record)
            update_elo(ratings, record)
        head_to_head.append(
            {
                "left": left.id,
                "right": right.id,
                "games": len(matchup),
                "leftSummary": summarize(matchup, left.id),
                "rightSummary": summarize(matchup, right.id),
            }
        )

    standings = []
    for agent in roster:
        agent_records = [record for record in records if agent.id in record["agents"].values()]
        standings.append(
            {
                "id": agent.id,
                "label": agent.label,
                "kind": agent.kind,
                "rating": round(ratings[agent.id]),
                **summarize(agent_records, agent.id),
            }
        )
    standings.sort(key=lambda entry: (-entry["rating"], -entry["points"], entry["id"]))
    result = {
        "schemaVersion": 1,
        "mode": "hourly-gnn-cnn-league",
        "boardSize": 7,
        "reservePerPlayer": 14,
        "round": round_number,
        "seed": seed,
        "gamesPerMatch": games_per_match,
        "simulations": simulations,
        "device": str(torch.device(device)),
        "rosterSize": len(roster),
        "pairings": len(pairings),
        "standings": standings,
        "headToHead": head_to_head,
        "games": records,
    }
    write_json(output_path, result)
    return {"path": str(output_path.relative_to(REPO_ROOT)), "rosterSize": len(roster), "pairings": len(pairings), "standings": standings}


def run_round(args: argparse.Namespace, round_number: int, run_dir: Path, python: str) -> dict:
    started = utc_now()
    seed = seed_for_round(round_number)
    log_path = run_dir / "run.log"
    data_dir = run_dir / "data"
    policy_data_dir = data_dir / "policy"
    qadv_data_dir = data_dir / "qadv"
    benchmark_dir = run_dir / "benchmark-7x7"
    qadv_benchmark_dir = run_dir / "benchmark-qadv"
    models_dir = run_dir / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    policy_generation_command = [
        python,
        "scripts/generate-7x7-selfplay.py",
        "--games-per-player",
        str(args.games_per_player),
        "--players",
        ",".join(args.players),
        "--seed",
        str(seed),
        "--workers",
        str(args.workers),
        "--simulations",
        str(args.selfplay_simulations),
        "--temperature-moves",
        str(args.temperature_moves),
        "--max-plies",
        "196",
        "--selfplay-device",
        args.selfplay_device,
        "--device",
        args.device,
        "--output-dir",
        str(policy_data_dir),
    ]
    run_command(policy_generation_command, REPO_ROOT, log_path, args.command_timeout)

    qadv_generation_command = [
        python,
        "scripts/generate-7x7-selfplay.py",
        "--games-per-player",
        str(args.qadv_games_per_player),
        "--players",
        ",".join(args.players),
        "--seed",
        str(seed + 5_000),
        "--workers",
        str(args.workers),
        "--simulations",
        str(args.qadv_simulations),
        "--temperature-moves",
        str(args.qadv_temperature_moves),
        "--max-plies",
        "196",
        "--selfplay-device",
        args.selfplay_device,
        "--device",
        args.device,
        "--output-dir",
        str(qadv_data_dir),
    ]
    run_command(qadv_generation_command, REPO_ROOT, log_path, args.command_timeout)
    qadv_generation = inspect_qadv_batch(qadv_data_dir)

    benchmark_command = [
        python,
        "scripts/build-7x7-benchmark.py",
        "--root",
        "research/runs/gnn",
        "--output",
        str(benchmark_dir),
        "--heldout-fraction",
        str(args.heldout_fraction),
        "--seed",
        str(seed + 1),
        "--exclude-path",
        "hourly/*/benchmark-7x7/*",
        "--exclude-path",
        "hourly/*/data/qadv/*.jsonl",
        "--exclude-path",
        "hourly/*/league.json",
        "--exclude-path",
        "hourly/*/report.json",
        "--exclude-path",
        "hourly/*/failure.json",
        "--exclude-path",
        "hourly/latest.json",
    ]
    benchmark_result = last_json_line(run_command(benchmark_command, REPO_ROOT, log_path, args.command_timeout))

    qadv_benchmark_command = [
        python,
        "scripts/build-7x7-benchmark.py",
        "--root",
        "research/runs/gnn",
        "--output",
        str(qadv_benchmark_dir),
        "--heldout-fraction",
        str(args.heldout_fraction),
        "--seed",
        str(seed + 2),
        "--require-action-values",
        "--exclude-path",
        "hourly/*/benchmark-7x7/*",
        "--exclude-path",
        "hourly/*/benchmark-qadv/*",
        "--exclude-path",
        "hourly/*/league.json",
        "--exclude-path",
        "hourly/*/report.json",
        "--exclude-path",
        "hourly/*/failure.json",
        "--exclude-path",
        "hourly/latest.json",
    ]
    qadv_benchmark_result = last_json_line(
        run_command(qadv_benchmark_command, REPO_ROOT, log_path, args.command_timeout)
    )
    qadv_checkpoint = Path(args.qadv_checkpoint)
    if not qadv_checkpoint.is_absolute():
        qadv_checkpoint = REPO_ROOT / qadv_checkpoint
    qadv_candidate = train_qadv_candidate(
        python=python,
        data_path=qadv_benchmark_dir / "all.jsonl",
        resume_path=qadv_checkpoint,
        output_path=models_dir / "qadv-arbiter.pt",
        steps=args.qadv_training_steps,
        learning_rate=args.qadv_learning_rate,
        seed=seed + 300,
        heldout_fraction=args.heldout_fraction,
        device=args.device,
        log_path=log_path,
        timeout_seconds=args.command_timeout,
    )

    candidates: list[dict] = []
    evaluations: list[dict] = []
    failed_candidates: list[dict] = []
    for index, name in enumerate(args.architectures):
        config = ARCHITECTURES[name]
        checkpoint = models_dir / f"{name}.pt"
        try:
            candidate = train_candidate(
                python=python,
                architecture_name=name,
                config=config,
                data_path=benchmark_dir / "train.jsonl",
                output_path=checkpoint,
                steps=args.training_steps,
                seed=seed + 100 + index,
                device=args.device,
                log_path=log_path,
                timeout_seconds=args.command_timeout,
            )
            candidates.append(candidate)
            evaluations.append(
                evaluate_candidate(
                    python=python,
                    candidate=candidate,
                    heldout_path=benchmark_dir / "heldout.jsonl",
                    seed=seed + 500 + index,
                    device=args.device,
                    max_examples=args.eval_max_examples,
                    log_path=log_path,
                    timeout_seconds=args.command_timeout,
                )
            )
        except Exception as error:  # keep the remaining architecture lanes moving
            failed_candidates.append({"name": name, "error": str(error)})

    league_info = None
    if candidates:
        league_info = run_league(
            round_number=round_number,
            seed=seed + 1_000,
            candidate_paths=[(candidate["name"], REPO_ROOT / candidate["checkpoint"]) for candidate in candidates],
            output_path=run_dir / "league.json",
            games_per_match=args.games_per_match,
            simulations=args.league_simulations,
            device_name=args.device,
            max_pairings=args.max_league_pairings,
        )

    finished = utc_now()
    report = {
        "schemaVersion": 1,
        "kind": "pathagon-hourly-ai-experiment",
        "status": "complete" if not failed_candidates else "partial",
        "round": round_number,
        "startedAtUtc": started.isoformat(),
        "finishedAtUtc": finished.isoformat(),
        "seed": seed,
        "boardSize": 7,
        "reservePerPlayer": 14,
        "dataExpansion": {
            "players": args.players,
            "gamesPerPlayer": args.games_per_player,
            "outputDir": str(data_dir.relative_to(REPO_ROOT)),
            "policyOutputDir": str(policy_data_dir.relative_to(REPO_ROOT)),
            "benchmark": benchmark_result,
        },
        "qadvantage": {
            "players": args.players,
            "gamesPerPlayer": args.qadv_games_per_player,
            "simulations": args.qadv_simulations,
            "temperatureMoves": args.qadv_temperature_moves,
            "outputDir": str(qadv_data_dir.relative_to(REPO_ROOT)),
            "generation": qadv_generation,
            "benchmark": qadv_benchmark_result,
            "candidate": qadv_candidate,
        },
        "cleanRetraining": {
            "steps": args.training_steps,
            "architectures": [ARCHITECTURES[name] | {"name": name} for name in args.architectures],
            "candidates": candidates,
            "evaluations": evaluations,
            "failed": failed_candidates,
        },
        "league": league_info,
        "architectureIdeas": [
            "separated transition/Q-ranking head using action-value targets",
            "policy/value uncertainty head for search calibration",
            "shared trunk with board-size-specific adapters",
            "learned move filters plus a calibrated heuristic residual",
        ],
        "promotion": "diagnostic only; no browser promotion or checkpoint overwrite",
    }
    write_json(run_dir / "report.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    result.add_argument("--python", help="Python interpreter; defaults to the project research venv")
    result.add_argument("--games-per-player", type=int, default=12)
    result.add_argument("--players", type=lambda value: parse_names(value, PLAYERS, "players"), default=list(PLAYERS))
    result.add_argument(
        "--architectures",
        type=lambda value: parse_names(value, tuple(ARCHITECTURES), "architectures"),
        default=list(ARCHITECTURES),
    )
    result.add_argument("--training-steps", type=int, default=250)
    result.add_argument("--workers", type=int, default=2)
    result.add_argument("--selfplay-simulations", type=int, default=4)
    result.add_argument("--qadv-games-per-player", type=int, default=8, help="fresh games per player for the root-Q target lane")
    result.add_argument("--qadv-simulations", type=int, default=32, help="MCTS simulations per move for the root-Q target lane")
    result.add_argument("--qadv-temperature-moves", type=int, default=16)
    result.add_argument("--qadv-training-steps", type=int, default=1_000)
    result.add_argument("--qadv-learning-rate", type=float, default=3e-4)
    result.add_argument("--qadv-checkpoint", default=str(DEFAULT_QADV_CHECKPOINT))
    result.add_argument("--league-simulations", type=int, default=2)
    result.add_argument("--games-per-match", type=int, default=1)
    result.add_argument("--max-league-pairings", type=int, default=0, help="cap league pairings; 0 battles the full active roster")
    result.add_argument("--temperature-moves", type=int, default=16)
    result.add_argument("--heldout-fraction", type=float, default=0.2)
    result.add_argument("--eval-max-examples", type=int, default=0, help="cap held-out scoring examples; 0 scores the full split")
    result.add_argument("--device", default="auto")
    result.add_argument("--selfplay-device", default="cpu")
    result.add_argument("--command-timeout", type=int, default=3_000)
    result.add_argument("--smoke", action="store_true", help="run one tiny local round for validation")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.smoke:
        args.games_per_player = 1
        args.qadv_games_per_player = 1
        args.training_steps = 2
        args.qadv_training_steps = 2
        args.qadv_simulations = 2
        args.qadv_temperature_moves = 8
        args.architectures = ["compact-gnn"]
        args.games_per_match = 1
        args.league_simulations = 1
        args.max_league_pairings = 4
        args.workers = 1
        args.command_timeout = min(args.command_timeout, 300)
        args.eval_max_examples = 256
    if any(
        value < 1
        for value in (
            args.games_per_player,
            args.qadv_games_per_player,
            args.training_steps,
            args.qadv_training_steps,
            args.qadv_simulations,
            args.workers,
        )
    ):
        raise SystemExit("games, simulations, training steps, and workers must be positive")
    if args.qadv_learning_rate <= 0:
        raise SystemExit("Q/Advantage learning rate must be positive")
    if not 0.0 < args.heldout_fraction < 1.0:
        raise SystemExit("held-out fraction must be between 0 and 1")
    if args.games_per_match < 1 or args.league_simulations < 1:
        raise SystemExit("league games and simulations must be positive")

    run_root = args.run_root if args.run_root.is_absolute() else REPO_ROOT / args.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / ".lock"
    state_path = run_root / "state.json"
    python = choose_python(args.python)

    with run_lock(lock_path) as acquired:
        if not acquired:
            print(json.dumps({"status": "skipped", "reason": "another hourly experiment is already running"}))
            return

        state = read_json(state_path, {})
        if not isinstance(state, dict):
            raise RuntimeError(f"expected an object in {state_path}")
        round_number = int(state.get("lastRound", 0)) + 1
        run_dir = run_root / f"round-{round_number:06d}-{timestamp()}"
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json(
            state_path,
            {
                **state,
                "status": "running",
                "lastRound": round_number,
                "lastStartedAtUtc": utc_now().isoformat(),
                "currentRun": str(run_dir.relative_to(REPO_ROOT)),
            },
        )
        try:
            report = run_round(args, round_number, run_dir, python)
            write_json(
                state_path,
                {
                    "status": report["status"],
                    "lastRound": round_number,
                    "lastFinishedAtUtc": report["finishedAtUtc"],
                    "lastReport": str((run_dir / "report.json").relative_to(REPO_ROOT)),
                },
            )
            write_json(run_root / "latest.json", report)
            print(json.dumps({"status": report["status"], "round": round_number, "report": str(run_dir / "report.json")}))
        except BaseException as error:
            failure = {
                "status": "failed",
                "round": round_number,
                "failedAtUtc": utc_now().isoformat(),
                "error": str(error),
                "run": str(run_dir.relative_to(REPO_ROOT)),
            }
            write_json(run_dir / "failure.json", failure)
            write_json(
                state_path,
                {
                    "status": "failed",
                    "lastRound": round_number,
                    "lastFailedAtUtc": failure["failedAtUtc"],
                    "lastFailure": str((run_dir / "failure.json").relative_to(REPO_ROOT)),
                },
            )
            print(json.dumps(failure), file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
