#!/usr/bin/env python3
"""Run the frozen 0% versus 50% curriculum arena on common seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LAB_ROOT = REPO_ROOT / "research/20260824-gnn-cnn-lab"
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from python.contract import agent_manifest  # type: ignore  # noqa: E402
from python.game import BoardConfig  # type: ignore  # noqa: E402
from python.league import AgentSpec, GNNAgent, checkpoint_hash, play_game, summarize, update_elo  # type: ignore  # noqa: E402
from python.train import choose_device, load_model  # type: ignore  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--portfolio", type=Path, required=True)
    parser.add_argument("--games", type=int, default=240)
    parser.add_argument("--seed", type=int, default=2026083600)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=196)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.games < 2 or args.games % 2:
        raise SystemExit("--games must be even and at least 2")
    if args.seed < 0 or args.seed + args.games > 4_294_967_296:
        raise SystemExit("seed range must fit in an unsigned 32-bit integer")

    device = choose_device(args.device)
    control_path = args.control.resolve()
    portfolio_path = args.portfolio.resolve()
    control_model = load_model(control_path, device)
    portfolio_model = load_model(portfolio_path, device)
    control_model.eval()
    portfolio_model.eval()
    control_id = "seeded-policy-value-0pct"
    portfolio_id = "seeded-policy-value-50pct"
    control = AgentSpec(
        control_id,
        "Ordinary curriculum control",
        "puct",
        GNNAgent(control_model, args.simulations),
        agent_manifest(runtime="python", node_budget=args.simulations, model_hash=checkpoint_hash(control_path)),
    )
    portfolio = AgentSpec(
        portfolio_id,
        "50% mixed-root curriculum",
        "puct",
        GNNAgent(portfolio_model, args.simulations),
        agent_manifest(runtime="python", node_budget=args.simulations, model_hash=checkpoint_hash(portfolio_path)),
    )
    config = BoardConfig(7, 14, args.max_plies)
    records = []
    ratings = {control_id: 1_000.0, portfolio_id: 1_000.0}
    for index in range(args.games):
        control_is_light = index % 2 == 0
        light, dark = (control, portfolio) if control_is_light else (portfolio, control)
        record = play_game(light, dark, config, args.seed + index, opening_random_plies=2)
        records.append(record)
        update_elo(ratings, record)
        if (index + 1) % max(1, args.games // 12) == 0 or index + 1 == args.games:
            print(f"arena: {index + 1}/{args.games}", flush=True)

    result = {
        "schemaVersion": 1,
        "mode": "curriculum-common-seed-arena",
        "boardSize": 7,
        "reservePerPlayer": 14,
        "maxPlies": args.max_plies,
        "seed": args.seed,
        "gamesPerMatch": args.games,
        "openingRandomPlies": 2,
        "simulations": args.simulations,
        "controlCheckpoint": str(control_path),
        "portfolioCheckpoint": str(portfolio_path),
        "ratings": {key: round(value) for key, value in ratings.items()},
        "standings": [
            {"id": control_id, "label": control.label, **summarize(records, control_id)},
            {"id": portfolio_id, "label": portfolio.label, **summarize(records, portfolio_id)},
        ],
        "games": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "standings": result["standings"]}, sort_keys=True))


if __name__ == "__main__":
    main()
