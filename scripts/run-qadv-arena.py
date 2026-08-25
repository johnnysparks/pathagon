#!/usr/bin/env python3
"""Run color-balanced 7x7 head-to-head games for The Q-Arbiter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from research.gnn.contract import agent_manifest
from research.gnn.game import BoardConfig
from research.gnn.league import (
    AgentSpec,
    HeuristicAgent,
    LunaticAgent,
    QAdvAgent,
    RandomAgent,
    checkpoint_hash,
    play_game,
    summarize,
    update_elo,
)
from research.gnn.train import choose_device, load_model


DEFAULT_QADV_ID = "qadv-arbiter-7x7-v0.1.0"
DEFAULT_QADV_LABEL = "The Q-Arbiter"
DEFAULT_GNN = REPO_ROOT / "research/runs/gnn/benchmark-7x7/generated/batch-20260824-neural-reval-20260824/reval-gnn-30k.pt"
DEFAULT_CNN = REPO_ROOT / "research/runs/gnn/benchmark-7x7/generated/batch-20260824-neural-reval-20260824/reval-cnn-30k.pt"


def build_roster(args: argparse.Namespace) -> dict[str, AgentSpec]:
    device = choose_device(args.device)
    qadv_path = Path(args.checkpoint)
    qadv_model = load_model(qadv_path, device, qadv=True)
    qadv_model.eval()
    roster: dict[str, AgentSpec] = {
        "qadv": AgentSpec(
            DEFAULT_QADV_ID,
            DEFAULT_QADV_LABEL,
            "learned",
            QAdvAgent(qadv_model),
            agent_manifest(runtime="python", model_hash=checkpoint_hash(qadv_path)),
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
    }
    for key, label, path in (("gnn", "Re-evaluated GNN 30k", Path(args.gnn_checkpoint)), ("cnn", "Re-evaluated CNN 30k", Path(args.cnn_checkpoint))):
        model = load_model(path, device)
        model.eval()
        from research.gnn.league import GNNAgent

        roster[key] = AgentSpec(
            f"{key}-reval30k-7x7",
            label,
            "puct",
            GNNAgent(model, simulations=args.baseline_simulations),
            agent_manifest(runtime="python", node_budget=args.baseline_simulations, model_hash=checkpoint_hash(path)),
        )
    return roster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="trained qadv checkpoint")
    parser.add_argument("--gnn-checkpoint", default=str(DEFAULT_GNN))
    parser.add_argument("--cnn-checkpoint", default=str(DEFAULT_CNN))
    parser.add_argument("--opponents", default="pathfinder,surveyor,gnn,cnn", help="comma-separated roster keys")
    parser.add_argument("--games-per-match", type=int, default=4, help="even count alternates Light/Dark assignments")
    parser.add_argument("--baseline-simulations", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=196)
    parser.add_argument("--seed", type=int, default=2026082500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.games_per_match < 2 or args.games_per_match % 2:
        raise SystemExit("--games-per-match must be an even number >= 2")
    roster = build_roster(args)
    opponent_keys = [item.strip() for item in args.opponents.split(",") if item.strip()]
    unknown = [key for key in opponent_keys if key not in roster or key == "qadv"]
    if unknown:
        raise SystemExit(f"unknown opponents: {', '.join(unknown)}")
    qadv = roster["qadv"]
    config = BoardConfig(7, 14, args.max_plies)
    ratings = {agent.id: 1_000.0 for agent in roster.values()}
    records: list[dict] = []
    head_to_head = []
    for opponent_key in opponent_keys:
        opponent = roster[opponent_key]
        matchup = []
        for game_index in range(args.games_per_match):
            light, dark = (qadv, opponent) if game_index % 2 == 0 else (opponent, qadv)
            record = play_game(light, dark, config, args.seed + len(records))
            records.append(record)
            matchup.append(record)
            update_elo(ratings, record)
        head_to_head.append({
            "qadv": qadv.id,
            "opponent": opponent.id,
            "games": len(matchup),
            "qadvSummary": summarize(matchup, qadv.id),
            "opponentSummary": summarize(matchup, opponent.id),
        })
    standings = []
    for agent in [qadv, *(roster[key] for key in opponent_keys)]:
        agent_records = [record for record in records if agent.id in record["agents"].values()]
        standings.append({"id": agent.id, "label": agent.label, "rating": round(ratings[agent.id]), **summarize(agent_records, agent.id)})
    standings.sort(key=lambda entry: (-entry["rating"], -entry["points"], entry["id"]))
    report = {
        "schemaVersion": 1,
        "mode": "qadv-arena",
        "agentId": qadv.id,
        "agentLabel": qadv.label,
        "checkpoint": str(Path(args.checkpoint)),
        "boardSize": 7,
        "reservePerPlayer": 14,
        "maxPlies": args.max_plies,
        "gamesPerMatch": args.games_per_match,
        "baselineSimulations": args.baseline_simulations,
        "seed": args.seed,
        "headToHead": head_to_head,
        "standings": standings,
        "games": records,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "games": len(records), "standings": standings}, sort_keys=True))


if __name__ == "__main__":
    main()
