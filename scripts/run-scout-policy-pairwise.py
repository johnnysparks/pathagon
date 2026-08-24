#!/usr/bin/env python3
"""Run color-balanced pairwise games for Scout and high-budget search candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from learning.gnn.contract import agent_manifest
from learning.gnn.game import BoardConfig
from learning.gnn.league import (
    AgentSpec,
    GNNAgent,
    HeuristicAgent,
    LunaticAgent,
    PolicyBeamAgent,
    RandomAgent,
    checkpoint_hash,
    load_model,
    play_game,
    summarize,
    update_elo,
)
from learning.gnn.train import choose_device


NEW_AGENT_NAMES = ("puct", "beam", "hybrid", "pathfinder10k", "scout10k")
DEFAULT_OPPONENTS = ("pathfinder", "surveyor", "lunatic", "learner", "cnn", "scout")
OPPONENT_NAMES = DEFAULT_OPPONENTS + ("coin-flip", "puct", "beam", "hybrid", "pathfinder10k", "scout10k")
CHECKPOINTS = {
    "scout": REPO_ROOT / "training/gnn/benchmark-7x7/small-gnn-warmstart.pt",
    "learner": REPO_ROOT / "training/gnn/benchmark-7x7/gnn-warmstart.pt",
    "cnn": REPO_ROOT / "training/gnn/benchmark-7x7/cnn-warmstart.pt",
}


def parse_csv(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in values if item not in allowed]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown {label}: {', '.join(unknown)}; choose from {', '.join(allowed)}")
    if not values:
        raise argparse.ArgumentTypeError(f"at least one {label} is required")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"{label} must be listed once each")
    return values


def build_agents(device_name: str) -> dict[str, AgentSpec]:
    device = choose_device(device_name)
    for checkpoint in CHECKPOINTS.values():
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    scout_checkpoint = CHECKPOINTS["scout"]
    scout_model = load_model(scout_checkpoint, device)
    scout_model.eval()
    scout_hash = checkpoint_hash(scout_checkpoint)

    learner_checkpoint = CHECKPOINTS["learner"]
    learner_model = load_model(learner_checkpoint, device)
    learner_model.eval()
    cnn_checkpoint = CHECKPOINTS["cnn"]
    cnn_model = load_model(cnn_checkpoint, device)
    cnn_model.eval()

    agents = {
        "puct": AgentSpec(
            "gnn-scout-puct32-7x7",
            "Scout + PUCT",
            "gnn",
            GNNAgent(scout_model, simulations=32),
            agent_manifest(runtime="python", node_budget=32, model_hash=scout_hash),
        ),
        "beam": AgentSpec(
            "gnn-scout-beam-7x7",
            "Scout + Neural Beam",
            "search",
            PolicyBeamAgent(scout_model, depth=4, beam_width=8, max_nodes=1_000),
            agent_manifest(runtime="python", depth=4, node_budget=1_000, beam=8, model_hash=scout_hash),
        ),
        "hybrid": AgentSpec(
            "gnn-scout-hybrid-beam-7x7",
            "Scout + Hybrid Beam",
            "search",
            PolicyBeamAgent(scout_model, depth=4, beam_width=8, max_nodes=1_000, heuristic_blend=0.35),
            agent_manifest(runtime="python", depth=4, node_budget=1_000, beam=8, model_hash=scout_hash),
        ),
        "pathfinder10k": AgentSpec(
            "pathfinder-deep-10k-7x7",
            "Pathfinder + Deep Search",
            "heuristic",
            HeuristicAgent(depth=4, beam_width=16, max_nodes=10_000),
            agent_manifest(runtime="python", depth=4, node_budget=10_000, beam=16),
        ),
        "scout10k": AgentSpec(
            "gnn-scout-beam10k-7x7",
            "Scout + 10k Beam",
            "search",
            PolicyBeamAgent(scout_model, depth=5, beam_width=16, max_nodes=10_000),
            agent_manifest(runtime="python", depth=5, node_budget=10_000, beam=16, model_hash=scout_hash),
        ),
        "pathfinder": AgentSpec(
            "pathfinder-v0.3.0",
            "The Pathfinder",
            "heuristic",
            HeuristicAgent(depth=2, beam_width=8, max_nodes=1_000),
            agent_manifest(runtime="python", depth=2, node_budget=1_000, beam=8),
        ),
        "surveyor": AgentSpec(
            "surveyor-v0.2.0",
            "The Surveyor",
            "heuristic",
            HeuristicAgent(depth=1, beam_width=12, max_nodes=500),
            agent_manifest(runtime="python", depth=1, node_budget=500, beam=12),
        ),
        "lunatic": AgentSpec(
            "lunatic-v0.1.0",
            "Lunatic",
            "heuristic",
            LunaticAgent(),
            agent_manifest(runtime="python", depth=1),
        ),
        "coin-flip": AgentSpec(
            "coin-flip-v0.0.1",
            "Coin Flip",
            "random",
            RandomAgent(),
            agent_manifest(runtime="python"),
        ),
        "learner": AgentSpec(
            "gnn-warmstart-7x7",
            "GNN Learner",
            "gnn",
            GNNAgent(learner_model, simulations=4),
            agent_manifest(runtime="python", node_budget=4, model_hash=checkpoint_hash(learner_checkpoint)),
        ),
        "cnn": AgentSpec(
            "cnn-baseline-7x7",
            "CNN baseline",
            "gnn",
            GNNAgent(cnn_model, simulations=4),
            agent_manifest(runtime="python", node_budget=4, model_hash=checkpoint_hash(cnn_checkpoint)),
        ),
        "scout": AgentSpec(
            "gnn-scout-7x7",
            "GNN Scout",
            "gnn",
            GNNAgent(scout_model, simulations=4),
            agent_manifest(runtime="python", node_budget=4, model_hash=scout_hash),
        ),
    }
    return agents


def normalize_endpoint(value: str) -> str:
    endpoint = value.rstrip("/")
    return endpoint if endpoint.endswith("/api/selfplay") else f"{endpoint}/api/selfplay"


def upload_record(endpoint: str, bearer_token: str, run_id: str, sequence: int, record: dict) -> None:
    entry = {
        "id": f"{run_id}-{sequence:04d}",
        "engine": "python-gnn-scout-policy",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-agents", type=lambda value: parse_csv(value, NEW_AGENT_NAMES, "new agents"), default=list(NEW_AGENT_NAMES))
    parser.add_argument("--opponents", type=lambda value: parse_csv(value, OPPONENT_NAMES, "opponents"), default=list(DEFAULT_OPPONENTS))
    parser.add_argument("--games-per-match", type=int, default=2, help="games per pairing; 2 gives one Light and one Dark start")
    parser.add_argument("--seed", type=int, default=2026082400)
    parser.add_argument("--max-plies", type=int, default=160)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--upload-url", required=True, help="site URL or /api/selfplay endpoint")
    parser.add_argument("--bearer-token", default=os.environ.get("PATHAGON_BRIDGE_TOKEN"))
    parser.add_argument("--run-id", default="offline-scout-policy-pairwise-20260824")
    parser.add_argument("--out", type=Path, default=Path("training/gnn/league/scout-policy-pairwise-20260824.json"))
    args = parser.parse_args()
    if args.games_per_match < 1 or args.games_per_match > 10:
        parser.error("--games-per-match must be between 1 and 10")
    if args.max_plies < 1 or args.max_plies > 196:
        parser.error("--max-plies must be between 1 and 196")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if not args.bearer_token:
        parser.error("--upload-url requires --bearer-token or PATHAGON_BRIDGE_TOKEN")
    if args.out.exists():
        parser.error(f"refusing to overwrite existing output: {args.out}")
    return args


def main() -> None:
    args = parse_args()
    agents = build_agents(args.device)
    config = BoardConfig(size=7, reserve_per_player=14, ply_limit=args.max_plies)
    endpoint = normalize_endpoint(args.upload_url)
    pairings = [(new_name, opponent) for new_name in args.new_agents for opponent in args.opponents]
    pairings.extend(
        (args.new_agents[left], args.new_agents[right])
        for left in range(len(args.new_agents))
        for right in range(left + 1, len(args.new_agents))
    )
    total_games = len(pairings) * args.games_per_match
    records = []
    ratings = {agent.id: 1_000.0 for agent in agents.values()}
    sequence = 0
    for pairing_index, (left_name, right_name) in enumerate(pairings):
        left = agents[left_name]
        right = agents[right_name]
        for game_index in range(args.games_per_match):
            light, dark = (left, right) if game_index % 2 == 0 else (right, left)
            seed = args.seed + pairing_index * 100 + game_index
            record = play_game(light, dark, config, seed)
            records.append(record)
            update_elo(ratings, record)
            winner = "draw" if record["winner"] is None else record["agents"][record["winner"]]
            print(
                f"game {sequence + 1}/{total_games}: {record['agents']['light']} vs {record['agents']['dark']} "
                f"→ {winner} ({record['reason']}, {record['plies']} plies)",
                flush=True,
            )
            upload_record(endpoint, args.bearer_token, args.run_id, sequence, record)
            print(f"  uploaded {args.run_id}-{sequence:04d}", flush=True)
            sequence += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "schemaVersion": 1,
        "mode": "scout-policy-pairwise",
        "boardSize": 7,
        "reservePerPlayer": 14,
        "gamesPerMatch": args.games_per_match,
        "maxPlies": args.max_plies,
        "seed": args.seed,
        "runId": args.run_id,
        "pairings": pairings,
        "standings": [
            {"id": agent.id, "label": agent.label, "rating": round(ratings[agent.id]), **summarize(records, agent.id)}
            for agent in agents.values()
            if any(agent.id in record["agents"].values() for record in records)
        ],
        "games": records,
    }
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runId": args.run_id, "games": len(records), "pairings": len(pairings), "output": str(args.out)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
