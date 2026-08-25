#!/usr/bin/env python3
"""Run a fresh 7x7 GNN Learner vs Surveyor match and optionally stream replays."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from research.gnn.contract import agent_manifest
from research.gnn.game import BoardConfig
from research.gnn.league import (
    AgentSpec,
    GNNAgent,
    HeuristicAgent,
    checkpoint_hash,
    load_model,
    play_game,
    summarize,
    update_elo,
)
from research.gnn.train import choose_device


GNN_ID = "gnn-warmstart-7x7"
SURVEYOR_ID = "surveyor-v0.2.0"
DEFAULT_CHECKPOINT = REPO_ROOT / "research/runs/gnn/benchmark-7x7/gnn-warmstart.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=4, help="number of games; colors alternate")
    parser.add_argument("--seed", type=int, default=2026082400)
    parser.add_argument("--simulations", type=int, default=4)
    parser.add_argument("--survey-depth", type=int, default=2, help="Surveyor search depth")
    parser.add_argument("--survey-beam", type=int, default=64, help="Surveyor candidate beam")
    parser.add_argument("--survey-max-nodes", type=int, default=12_000, help="Surveyor node budget")
    parser.add_argument("--max-plies", type=int, default=180)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out", type=Path, default=Path("research/runs/gnn/league/gnn-learner-vs-surveyor-fresh.json"))
    parser.add_argument("--upload-url", help="site URL or /api/selfplay endpoint; upload one game after each result")
    parser.add_argument("--run-id", default="gnn-surveyor-fresh-20260824")
    parser.add_argument("--bearer-token", default=os.environ.get("PATHAGON_BRIDGE_TOKEN"))
    args = parser.parse_args()
    if args.games < 1 or args.games > 100:
        parser.error("--games must be between 1 and 100")
    if args.seed < 0 or args.seed + args.games > 4_294_967_296:
        parser.error("seed range must fit in an unsigned 32-bit integer")
    if args.upload_url and not args.bearer_token:
        parser.error("--upload-url requires --bearer-token or PATHAGON_BRIDGE_TOKEN")
    return args


def build_agents(args: argparse.Namespace) -> tuple[AgentSpec, AgentSpec]:
    device = choose_device(args.device)
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing GNN checkpoint: {checkpoint}")
    model = load_model(checkpoint, device)
    model.eval()
    gnn = AgentSpec(
        GNN_ID,
        "GNN Learner · 7x7 warm start",
        "gnn",
        GNNAgent(model, args.simulations),
        agent_manifest(runtime="python", node_budget=args.simulations, model_hash=checkpoint_hash(checkpoint)),
    )
    surveyor = HeuristicAgent(depth=args.survey_depth, beam_width=args.survey_beam, max_nodes=args.survey_max_nodes)
    surveyor_spec = AgentSpec(
        SURVEYOR_ID,
        "The Surveyor",
        "heuristic",
        surveyor,
        agent_manifest(runtime="python", depth=args.survey_depth, beam=args.survey_beam, node_budget=args.survey_max_nodes),
    )
    return gnn, surveyor_spec


def upload_record(endpoint: str, bearer_token: str, run_id: str, sequence: int, record: dict) -> None:
    entry = {
        "id": f"{run_id}-{sequence}",
        "engine": "python-gnn-bridge",
        "mode": "cross-play",
        "runId": run_id,
        "record": record,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps({"games": [entry]}).encode("utf-8"),
        headers={"OAI-Sites-Authorization": f"Bearer {bearer_token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if response.status >= 300 or not payload.get("accepted"):
        raise RuntimeError(f"bridge upload rejected game {sequence}: {payload}")


def normalize_endpoint(value: str | None) -> str | None:
    if not value:
        return None
    endpoint = value.rstrip("/")
    return endpoint if endpoint.endswith("/api/selfplay") else f"{endpoint}/api/selfplay"


def main() -> None:
    args = parse_args()
    gnn, surveyor = build_agents(args)
    config = BoardConfig(size=7, reserve_per_player=14, ply_limit=args.max_plies)
    endpoint = normalize_endpoint(args.upload_url)
    records = []
    ratings = {GNN_ID: 957.0, SURVEYOR_ID: 1_085.0}

    for sequence in range(args.games):
        gnn_is_light = sequence % 2 == 0
        light, dark = (gnn, surveyor) if gnn_is_light else (surveyor, gnn)
        record = play_game(light, dark, config, args.seed + sequence)
        records.append(record)
        update_elo(ratings, record)
        winner = "draw" if record["winner"] is None else record["agents"][record["winner"]]
        print(f"game {sequence + 1}/{args.games}: {record['agents']['light']} vs {record['agents']['dark']} → {winner} ({record['reason']}, {record['plies']} plies)", flush=True)
        if endpoint:
            upload_record(endpoint, args.bearer_token, args.run_id, sequence, record)
            print(f"  uploaded to {args.run_id}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.out}")
    output = {
        "schemaVersion": 1,
        "mode": "gnn-surveyor-arena",
        "boardSize": 7,
        "reservePerPlayer": 14,
        "seed": args.seed,
        "gamesPerMatch": args.games,
        "simulations": args.simulations,
        "surveySearch": {"depth": args.survey_depth, "beam": args.survey_beam, "maxNodes": args.survey_max_nodes},
        "checkpoint": str(args.checkpoint),
        "runId": args.run_id,
        "standings": [
            {"id": GNN_ID, "label": "GNN Learner · 7x7 warm start", "rating": round(ratings[GNN_ID]), **summarize(records, GNN_ID)},
            {"id": SURVEYOR_ID, "label": "The Surveyor", "rating": round(ratings[SURVEYOR_ID]), **summarize(records, SURVEYOR_ID)},
        ],
        "games": records,
    }
    args.out.write_text(f"{json.dumps(output, indent=2)}\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "standings": output["standings"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
