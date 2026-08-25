#!/usr/bin/env python3
"""Run iterative QAdv battle batches and keep the live cross-play ladder fed."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "research/runs/gnn/benchmark-7x7/generated/qadv-live-tournament-20260825"
DEFAULT_CHECKPOINT = REPO_ROOT / "research/runs/gnn/benchmark-7x7/generated/batch-20260824-qadv-128-pilot-20260824/qadv-arbiter-7x7-v0.1.0.pt"
DEFAULT_BASE_DATA = DEFAULT_CHECKPOINT.parent
GUIDED_ID = "qadv-arbiter-guided-7x7-v0.2.0"
GUIDED_LABEL = "The Q-Arbiter · Guided Search"
SITE_URL = "https://pathagon-game.sparks-house-6466.chatgpt.site"


def run(command: list[str]) -> None:
    print("$ " + " ".join(command[:3]) + (" …" if len(command) > 3 else ""), file=sys.stderr, flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def initialize_corpus(base_data: Path, corpus: Path) -> None:
    if corpus.exists():
        return
    corpus.parent.mkdir(parents=True, exist_ok=True)
    with corpus.open("w", encoding="utf-8") as output:
        for source in sorted(base_data.glob("*.jsonl")):
            with source.open(encoding="utf-8") as input_file:
                shutil.copyfileobj(input_file, output)


def append_games(corpus: Path, report: Path) -> int:
    payload = json.loads(report.read_text(encoding="utf-8"))
    games = payload.get("games", [])
    with corpus.open("a", encoding="utf-8") as output:
        for game in games:
            output.write(json.dumps(game, sort_keys=True) + "\n")
    return len(games)


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-data", type=Path, default=DEFAULT_BASE_DATA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--site-url", default=SITE_URL)
    parser.add_argument("--batches", type=int, default=10, help="number of 100-game battle/retrain cycles")
    parser.add_argument("--games-per-batch", type=int, default=100)
    parser.add_argument("--train-steps", type=int, default=1_000)
    parser.add_argument("--opponents", default="pathfinder")
    parser.add_argument("--qadv-top-k", type=int, default=4)
    parser.add_argument("--qadv-reply-k", type=int, default=3)
    parser.add_argument("--baseline-simulations", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=196)
    parser.add_argument("--seed", type=int, default=2026084000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--archive-token", default=os.environ.get("PATHAGON_ARCHIVE_TOKEN"))
    args = parser.parse_args()
    if args.batches < 1 or args.games_per_batch < 2 or args.games_per_batch % 2:
        raise SystemExit("--batches must be positive and --games-per-batch must be an even number >= 2")
    if not args.archive_token:
        raise SystemExit("live tournament requires PATHAGON_ARCHIVE_TOKEN or --archive-token")
    if not args.checkpoint.is_absolute():
        args.checkpoint = REPO_ROOT / args.checkpoint
    if not args.base_data.is_absolute():
        args.base_data = REPO_ROOT / args.base_data
    if not args.out_dir.is_absolute():
        args.out_dir = REPO_ROOT / args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    corpus = args.out_dir / "qadv-live-corpus.jsonl"
    manifest_path = args.out_dir / "tournament-manifest.json"
    initialize_corpus(args.base_data, corpus)

    manifest = {
        "schemaVersion": 1,
        "mode": "qadv-live-tournament",
        "agentId": GUIDED_ID,
        "agentLabel": GUIDED_LABEL,
        "gamesPerBatch": args.games_per_batch,
        "batchesRequested": args.batches,
        "trainStepsPerBatch": args.train_steps,
        "opponents": [item.strip() for item in args.opponents.split(",") if item.strip()],
        "selector": "guided",
        "qadvTopK": args.qadv_top_k,
        "qadvReplyK": args.qadv_reply_k,
        "checkpoint": str(args.checkpoint),
        "batches": [],
    }
    write_manifest(manifest_path, manifest)

    current_checkpoint = args.checkpoint
    for batch_index in range(args.batches):
        batch_number = batch_index + 1
        batch_name = f"batch-{batch_number:04d}"
        report = args.out_dir / f"{batch_name}.json"
        run_id = f"qadv-live-tournament-20260825-{batch_name}"
        batch_seed = args.seed + batch_index * 100_000
        run([
            sys.executable,
            str(REPO_ROOT / "scripts/run-qadv-arena.py"),
            "--checkpoint", str(current_checkpoint),
            "--selector", "guided",
            "--qadv-top-k", str(args.qadv_top_k),
            "--qadv-reply-k", str(args.qadv_reply_k),
            "--opponents", args.opponents,
            "--games-per-match", str(args.games_per_batch),
            "--baseline-simulations", str(args.baseline_simulations),
            "--max-plies", str(args.max_plies),
            "--seed", str(batch_seed),
            "--device", args.device,
            "--out", str(report),
        ])
        games = append_games(corpus, report)
        run([
            "node",
            "--experimental-strip-types",
            str(REPO_ROOT / "scripts/archive-selfplay.ts"),
            "--file", str(report),
            "--url", args.site_url,
            "--engine", "python",
            "--mode", "cross-play",
            "--run-id", run_id,
            "--token", args.archive_token,
        ])

        next_checkpoint = args.out_dir / f"{GUIDED_ID}-iter-{batch_number:04d}.pt"
        run([
            sys.executable,
            str(REPO_ROOT / "research/gnn/train.py"),
            "qadv",
            "--data", str(corpus),
            "--resume", str(current_checkpoint),
            "--out", str(next_checkpoint),
            "--steps", str(args.train_steps),
            "--architecture", "gnn",
            "--agent-id", GUIDED_ID,
            "--agent-name", GUIDED_LABEL,
            "--agent-version", "0.2.0",
            "--agent-kind", "learned",
            "--agent-engine", "python-gnn",
            "--seed", str(batch_seed + 50_000),
            "--device", args.device,
        ])
        payload = json.loads(report.read_text(encoding="utf-8"))
        qadv_summary = payload["headToHead"][0]["qadvSummary"] if payload.get("headToHead") else None
        batch_entry = {
            "batch": batch_number,
            "runId": run_id,
            "seed": batch_seed,
            "games": games,
            "battleSummary": qadv_summary,
            "checkpointBefore": str(current_checkpoint),
            "checkpointAfter": str(next_checkpoint),
        }
        manifest["batches"].append(batch_entry)
        manifest["checkpoint"] = str(next_checkpoint)
        write_manifest(manifest_path, manifest)
        current_checkpoint = next_checkpoint
        print(json.dumps(batch_entry, sort_keys=True), flush=True)

    print(json.dumps({"manifest": str(manifest_path), "batches": len(manifest["batches"]), "checkpoint": str(current_checkpoint)}, sort_keys=True))


if __name__ == "__main__":
    main()
