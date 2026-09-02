#!/usr/bin/env python3
"""Train and evaluate a compact JEPA on Rust-emitted afterstates."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch


ROOT_DIR = Path(__file__).resolve().parents[3]
MODULE_DIR = ROOT_DIR / "research/20260901-jepa-afterstate/python"
sys.path.insert(0, str(MODULE_DIR))

from jepa_afterstate import (  # noqa: E402
    ActionConditionedJEPA,
    evaluate_jepa,
    load_rust_transitions,
    train_jepa,
)


def load_initial_gnn(model: ActionConditionedJEPA, checkpoint: Path, device: torch.device) -> None:
    """Warm-start the online and EMA encoders from a policy/value GNN."""

    legacy_root = ROOT_DIR / "research/20260824-gnn-cnn-lab"
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))
    from python.train import load_model  # pylint: disable=import-outside-toplevel

    initial = load_model(checkpoint.resolve(), device, qadv=False)
    if not hasattr(initial, "state_dict") or initial.config_dict().get("architecture") == "residual-cnn-7x7":
        raise ValueError("--init-checkpoint must be a GNN policy/value checkpoint")
    model.online.load_state_dict(initial.state_dict())
    model.target.load_state_dict(initial.state_dict())
    model._freeze_target()


def split_rows(rows, heldout_fraction: float, seed: int):
    games = sorted({row.game for row in rows})
    rng = random.Random(seed)
    rng.shuffle(games)
    heldout_count = max(1, int(round(len(games) * heldout_fraction)))
    heldout_games = set(games[:heldout_count])
    train = [row for row in rows if row.game not in heldout_games]
    heldout = [row for row in rows if row.game in heldout_games]
    if not train or not heldout:
        raise ValueError("game-grouped transition split is empty")
    return train, heldout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--message-layers", type=int, default=8)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--replay-input", type=Path, help="optional JSONL replay for policy/value fine-tuning")
    parser.add_argument("--replay-steps", type=int, default=0)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=2026090102)
    parser.add_argument("--seed", type=int, default=2026090103)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if args.device != "cpu":
        torch.set_default_device(args.device)
    torch.manual_seed(args.seed)
    rows = load_rust_transitions(input_path)
    train_rows, heldout_rows = split_rows(rows, args.heldout_fraction, args.split_seed)
    model = ActionConditionedJEPA(
        hidden_size=args.hidden_size,
        message_layers=args.message_layers,
        embedding_size=args.embedding_size,
    ).to(torch.device(args.device))
    if args.init_checkpoint:
        load_initial_gnn(model, args.init_checkpoint, torch.device(args.device))
    training = train_jepa(
        model,
        train_rows,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    if args.replay_input and args.replay_steps > 0:
        legacy_root = ROOT_DIR / "research/20260824-gnn-cnn-lab"
        if str(legacy_root) not in sys.path:
            sys.path.insert(0, str(legacy_root))
        from python.game import BoardConfig  # pylint: disable=import-outside-toplevel
        from python.train import load_replay_source, train_replay  # pylint: disable=import-outside-toplevel

        replay_rows = load_replay_source(
            args.replay_input.resolve(), BoardConfig(size=7, reserve_per_player=14, ply_limit=20)
        )
        policy_loss, value_loss = train_replay(
            model.online,
            replay_rows,
            steps=args.replay_steps,
            learning_rate=args.learning_rate,
            seed=args.seed + 1,
            symmetry_augmentation=True,
            value_weight=1.0,
        )
        training["policyLoss"] = policy_loss
        training["valueLoss"] = value_loss
        training["replayExamples"] = float(len(replay_rows))
    heldout = evaluate_jepa(model, heldout_rows, batch_size=args.batch_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": {
                "architecture": "action-conditioned-jepa-afterstate-v1",
                "hidden_size": args.hidden_size,
                "message_layers": args.message_layers,
                "embedding_size": args.embedding_size,
            },
            "online_state_dict": model.online.state_dict(),
            "jepa_state_dict": model.state_dict(),
            "metadata": {
                "input": str(input_path),
                "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "rows": len(rows),
                "train_rows": len(train_rows),
                "heldout_rows": len(heldout_rows),
                "train_games": len({row.game for row in train_rows}),
                "heldout_games": len({row.game for row in heldout_rows}),
                "training": training,
                "heldout": heldout,
                "exact_engine_authority": "pathagon-jepa-export / rust-bitboard",
            },
        },
        output_path,
    )
    policy_value_path = output_path.with_name(f"{output_path.stem}-policy-value{output_path.suffix}")
    torch.save(
        {
            "model_config": model.online.config_dict(),
            "state_dict": model.online.state_dict(),
            "metadata": {
                "mode": "jepa-afterstate-online-trunk",
                "jepa_checkpoint": str(output_path),
                "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "exact_engine_authority": "pathagon-jepa-export / rust-bitboard",
            },
        },
        policy_value_path,
    )
    report = {
        "status": "complete",
        "checkpoint": str(output_path),
        "policyValueCheckpoint": str(policy_value_path),
        "rows": len(rows),
        "trainRows": len(train_rows),
        "heldoutRows": len(heldout_rows),
        "training": training,
        "heldout": heldout,
    }
    output_path.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
