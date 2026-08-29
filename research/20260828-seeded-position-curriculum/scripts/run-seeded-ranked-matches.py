#!/usr/bin/env python3
"""Run a color-balanced ranked ladder for the seeded-position candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.gnn.game import BoardConfig
from research.gnn.league import (
    AgentSpec,
    GNNAgent,
    HeuristicAgent,
    LunaticAgent,
    QAdvAgent,
    RandomAgent,
    agent_manifest,
    agent_version,
    checkpoint_hash,
    play_game,
    records_for_agent,
    summarize,
    update_elo,
)
from research.gnn.train import choose_device, load_model


def candidate_roster(root: Path, device: torch.device, simulations: int) -> list[AgentSpec]:
    roster: list[AgentSpec] = []
    parent = Path("research/20260827-pathfinder-rust-sorter/native-soft-sorter-depth5-small-400.pt")
    checkpoints = [("parent-policy-value", "Parent policy/value", parent, False)]
    for condition in ("0pct", "25pct", "50pct"):
        base = root / f"condition-{condition}"
        checkpoints.extend([
            (f"seeded-policy-value-{condition}", f"Seeded policy/value {condition}", base / "policy-value.pt", False),
            (f"seeded-qadv-{condition}", f"Seeded QAdv {condition}", base / "qadv.pt", True),
        ])
    for agent_id, label, path, qadv in checkpoints:
        model = load_model(path, device, qadv=True if qadv else False)
        model.eval()
        chooser = QAdvAgent(model) if qadv else GNNAgent(model, simulations)
        roster.append(AgentSpec(
            agent_id,
            label,
            "learned",
            chooser,
            agent_manifest(runtime="python", node_budget=simulations, model_hash=checkpoint_hash(path)),
        ))
    pathfinder = HeuristicAgent(depth=2, beam_width=8, max_nodes=1_000)
    surveyor = HeuristicAgent(depth=1, beam_width=12, max_nodes=500)
    roster.extend([
        AgentSpec("pathfinder-v0.3.0", "The Pathfinder", "heuristic", pathfinder, agent_manifest(runtime="python", depth=2, beam=8, node_budget=1_000)),
        AgentSpec("surveyor-v0.2.0", "The Surveyor", "heuristic", surveyor, agent_manifest(runtime="python", depth=1, beam=12, node_budget=500)),
        AgentSpec("lunatic-v0.1.0", "Lunatic", "heuristic", LunaticAgent(), agent_manifest(runtime="python", depth=1)),
        AgentSpec("coin-flip-v0.0.1", "Coin Flip", "random", RandomAgent(), agent_manifest(runtime="python")),
    ])
    return roster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--games-per-match", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026083100)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.games_per_match < 2 or args.games_per_match % 2:
        raise SystemExit("games-per-match must be even and at least 2 for color balance")
    device = choose_device(args.device)
    config = BoardConfig(7, 14)
    roster = candidate_roster(args.root, device, args.simulations)
    ratings = {agent.id: 1_000.0 for agent in roster}
    records = []
    head_to_head = []
    for left_index, left in enumerate(roster):
        for right_index in range(left_index + 1, len(roster)):
            right = roster[right_index]
            matchup = []
            for game_index in range(args.games_per_match):
                left_is_light = game_index % 2 == 0
                light, dark = (left, right) if left_is_light else (right, left)
                record = play_game(
                    light,
                    dark,
                    config,
                    args.seed + left_index * 100_000 + right_index * 1_000 + game_index,
                )
                matchup.append(record)
                records.append(record)
                update_elo(ratings, record)
            head_to_head.append({
                "left": left.id,
                "right": right.id,
                "games": len(matchup),
                "leftSummary": summarize(matchup, left.id),
                "rightSummary": summarize(matchup, right.id),
            })
            print(
                f"match {left.id} vs {right.id}: "
                f"{summarize(matchup, left.id)['points']:.1f}-{summarize(matchup, right.id)['points']:.1f}",
                flush=True,
            )
    standings = []
    for agent in roster:
        summary = summarize(records_for_agent(records, agent.id), agent.id)
        standings.append({
            "id": agent.id,
            "label": agent.label,
            "kind": agent.kind,
            "rating": round(ratings[agent.id]),
            **summary,
        })
    standings.sort(key=lambda entry: (-entry["rating"], -entry["points"], entry["id"]))
    result = {
        "schemaVersion": 1,
        "mode": "seeded-position-ranked-ladder",
        "boardSize": 7,
        "reservePerPlayer": 14,
        "seed": args.seed,
        "gamesPerMatch": args.games_per_match,
        "simulations": args.simulations,
        "standings": standings,
        "headToHead": head_to_head,
        "games": records,
        "device": str(device),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "games": len(records), "standings": standings}, sort_keys=True))


if __name__ == "__main__":
    main()
