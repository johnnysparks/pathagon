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
    QAdvGuidedAgent,
    RandomAgent,
    checkpoint_hash,
    play_game,
    summarize,
    update_elo,
)
from research.gnn.train import choose_device, load_model


DEFAULT_QADV_ID = "qadv-arbiter-7x7-v0.1.0"
DEFAULT_QADV_LABEL = "The Q-Arbiter"
GUIDED_QADV_ID = "qadv-arbiter-guided-7x7-v0.2.0"
GUIDED_QADV_LABEL = "The Q-Arbiter · Guided Search"
DEFAULT_GNN = REPO_ROOT / "research/runs/gnn/benchmark-7x7/generated/batch-20260824-neural-reval-20260824/reval-gnn-30k.pt"
DEFAULT_CNN = REPO_ROOT / "research/runs/gnn/benchmark-7x7/generated/batch-20260824-neural-reval-20260824/reval-cnn-30k.pt"


def coordinate(index: int, board_size: int) -> str:
    row, column = divmod(index, board_size)
    return f"{chr(65 + column)}{row + 1}"


def format_move(move: dict, board_size: int) -> str:
    action = move["action"]
    destination = coordinate(action["to"], board_size)
    if action["kind"] == "place":
        return f"P{destination}"
    return f"R{coordinate(action['from'], board_size)}→{destination}"


def interesting_move_digest(record: dict) -> str:
    """Extract a few concrete move stories from a completed game record."""
    moves = record.get("moves", [])
    if not moves:
        return "no moves"
    labels = {
        side: record.get("agentSpecifications", {}).get(side, {}).get("name", record["agents"][side])
        for side in ("light", "dark")
    }
    board_size = record["config"]["boardSize"]
    highlights: list[str] = []
    captures = [move for move in moves if move.get("captured")]
    if captures:
        move = max(captures, key=lambda item: len(item["captured"]))
        count = len(move["captured"])
        plural = "piece" if count == 1 else "pieces"
        highlights.append(
            f"capture burst on ply {move['ply']}: {labels[move['player']]} {format_move(move, board_size)} removed {count} {plural}"
        )

    relocations = [move for move in moves if move["action"]["kind"] == "relocate"]
    if relocations:
        first = relocations[0]
        highlights.append(
            f"reserve-to-relocation turn on ply {first['ply']}: {labels[first['player']]} {format_move(first, board_size)}"
        )
        def distance(item: dict) -> int:
            action = item["action"]
            from_row, from_column = divmod(action["from"], board_size)
            to_row, to_column = divmod(action["to"], board_size)
            return abs(from_row - to_row) + abs(from_column - to_column)

        longest = max(relocations, key=distance)
        if distance(longest) >= 4:
            highlights.append(
                f"long relocation Δ{distance(longest)} on ply {longest['ply']}: {labels[longest['player']]} {format_move(longest, board_size)}"
            )

    winner = record.get("winner")
    if winner and len(highlights) < 3:
        final = moves[-1]
        highlights.append(
            f"closing path on ply {final['ply']}: {labels[final['player']]} {format_move(final, board_size)}"
        )
    if not highlights:
        opening = " / ".join(format_move(move, board_size) for move in moves[:3])
        highlights.append(f"quiet opening {opening}")
    return "; ".join(highlights[:3])


def build_roster(args: argparse.Namespace) -> dict[str, AgentSpec]:
    device = choose_device(args.device)
    qadv_path = Path(args.checkpoint)
    qadv_model = load_model(qadv_path, device, qadv=True)
    qadv_model.eval()
    guided = args.selector == "guided"
    roster: dict[str, AgentSpec] = {
        "qadv": AgentSpec(
            GUIDED_QADV_ID if guided else DEFAULT_QADV_ID,
            GUIDED_QADV_LABEL if guided else DEFAULT_QADV_LABEL,
            "learned",
            QAdvGuidedAgent(
                qadv_model,
                top_k=args.qadv_top_k,
                reply_k=args.qadv_reply_k,
                temperature_moves=args.temperature_moves,
                policy_temperature=args.policy_temperature,
                opening_moves=args.opening_moves,
                opening_temperature=args.opening_temperature,
                opening_randomness=args.opening_randomness,
                pathfinder_temperature=args.pathfinder_temperature,
            ) if guided else QAdvAgent(qadv_model),
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
    parser.add_argument("--selector", choices=("direct", "guided"), default="direct", help="direct Q-max or QAdv-guided shallow adversarial search")
    parser.add_argument("--qadv-top-k", type=int, default=12)
    parser.add_argument("--qadv-reply-k", type=int, default=8)
    parser.add_argument("--temperature-moves", type=int, default=48)
    parser.add_argument("--policy-temperature", type=float, default=1.15)
    parser.add_argument("--opening-moves", type=int, default=16)
    parser.add_argument("--opening-temperature", type=float, default=1.8)
    parser.add_argument("--opening-randomness", type=float, default=0.30)
    parser.add_argument("--pathfinder-temperature", type=float, default=1.15)
    parser.add_argument("--games-per-match", type=int, default=4, help="even count alternates Light/Dark assignments")
    parser.add_argument("--baseline-simulations", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=196)
    parser.add_argument("--seed", type=int, default=2026082500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--verbose-progress", action="store_true", help="print one concrete result and move digest per game")
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
    total_games = args.games_per_match * len(opponent_keys)
    if args.verbose_progress:
        print(f"[arena] starting {total_games} games: {qadv.label} guided search vs {', '.join(opponent_keys)}", file=sys.stderr, flush=True)
    for opponent_key in opponent_keys:
        opponent = roster[opponent_key]
        matchup = []
        for game_index in range(args.games_per_match):
            light, dark = (qadv, opponent) if game_index % 2 == 0 else (opponent, qadv)
            record = play_game(light, dark, config, args.seed + len(records))
            records.append(record)
            matchup.append(record)
            update_elo(ratings, record)
            if args.verbose_progress:
                winner = record.get("winner")
                winner_label = "draw" if winner is None else record["agentSpecifications"][winner]["name"]
                print(
                    f"[arena] game {len(records):03d}/{total_games}: {winner_label} · {record['plies']} plies | {interesting_move_digest(record)}",
                    file=sys.stderr,
                    flush=True,
                )
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
        "selector": args.selector,
        "qadvTopK": args.qadv_top_k,
        "qadvReplyK": args.qadv_reply_k,
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
