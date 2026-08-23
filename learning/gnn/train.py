"""Train the GNN warm start or run compact AlphaZero-style generations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .data import ReplayExample, action_index, load_replay_examples
from .game import BoardConfig
from .model import PathagonGNN
from .selfplay import SearchExample, generate_game


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_model(model: PathagonGNN, path: Path, metadata: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_config": model.config_dict(), "state_dict": model.state_dict(), "metadata": metadata}, path)


def load_model(path: Path, device: torch.device) -> PathagonGNN:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["model_config"]
    model = PathagonGNN(config["hidden_size"], config["message_layers"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def train_replay(
    model: PathagonGNN,
    examples: Sequence[ReplayExample],
    steps: int,
    learning_rate: float,
    seed: int,
) -> Tuple[float, float]:
    if not examples:
        raise ValueError("replay dataset is empty")
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    policy_total = 0.0
    value_total = 0.0
    for _ in range(steps):
        example = examples[rng.randrange(len(examples))]
        actions = list(example.state.legal_actions())
        logits, value = model.policy_value(example.state, actions)
        target = torch.tensor([action_index(example.state, example.action)], dtype=torch.long, device=logits.device)
        expected_value = torch.tensor(example.value, dtype=value.dtype, device=value.device)
        policy_loss = F.cross_entropy(logits.unsqueeze(0), target)
        value_loss = F.mse_loss(value, expected_value)
        loss = policy_loss + value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        policy_total += float(policy_loss.detach().cpu())
        value_total += float(value_loss.detach().cpu())
    return policy_total / steps, value_total / steps


def train_search_examples(
    model: PathagonGNN,
    examples: Sequence[SearchExample],
    steps: int,
    learning_rate: float,
    seed: int,
) -> Tuple[float, float]:
    if not examples:
        raise ValueError("MCTS dataset is empty")
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    policy_total = 0.0
    value_total = 0.0
    for _ in range(steps):
        example = examples[rng.randrange(len(examples))]
        actions = list(example.actions)
        logits, value = model.policy_value(example.state, actions)
        target_policy = torch.tensor(example.policy, dtype=logits.dtype, device=logits.device)
        target_value = torch.tensor(example.value, dtype=value.dtype, device=value.device)
        policy_loss = -(target_policy * F.log_softmax(logits, dim=0)).sum()
        value_loss = F.mse_loss(value, target_value)
        loss = policy_loss + value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        policy_total += float(policy_loss.detach().cpu())
        value_total += float(value_loss.detach().cpu())
    return policy_total / steps, value_total / steps


def run_warmstart(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    config = BoardConfig(args.size, args.reserve)
    examples = load_replay_examples(Path(args.data), config)
    if args.max_examples:
        examples = examples[: args.max_examples]
    model = PathagonGNN(args.hidden, args.layers).to(device)
    policy_loss, value_loss = train_replay(model, examples, args.steps, args.learning_rate, args.seed)
    metadata = {
        "mode": "replay-warmstart",
        "data": str(args.data),
        "examples": len(examples),
        "board_size": args.size,
        "reserve_per_player": config.reserve_per_player,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
    }
    save_model(model, Path(args.out), metadata)
    print(json.dumps(metadata | {"out": args.out, "device": str(device)}, sort_keys=True))


def run_alphazero(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    config = BoardConfig(args.size, args.reserve)
    model = load_model(Path(args.resume), device) if args.resume else PathagonGNN(args.hidden, args.layers).to(device)
    model.eval()
    history: List[SearchExample] = []
    for generation in range(args.generations):
        generated: List[SearchExample] = []
        lengths = []
        for game_index in range(args.games):
            examples, final_state = generate_game(
                model,
                config,
                simulations=args.simulations,
                temperature_moves=args.temperature_moves,
                seed=args.seed + generation * args.games + game_index,
                add_root_noise=True,
            )
            generated.extend(examples)
            lengths.append(final_state.ply)
        history.extend(generated)
        if args.replay_limit and len(history) > args.replay_limit:
            history = history[-args.replay_limit :]
        policy_loss, value_loss = train_search_examples(
            model, history, args.updates, args.learning_rate, args.seed + generation
        )
        model.eval()
        metadata = {
            "mode": "alphazero-generation",
            "generation": generation,
            "board_size": args.size,
            "reserve_per_player": config.reserve_per_player,
            "games": args.games,
            "examples": len(generated),
            "replay_buffer": len(history),
            "average_plies": sum(lengths) / len(lengths) if lengths else 0.0,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
        }
        save_model(model, Path(args.out), metadata)
        print(json.dumps(metadata | {"out": args.out, "device": str(device)}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("warmstart", "alphazero"))
    result.add_argument("--data", help="schema-v2 JSONL for replay warm-start")
    result.add_argument("--out", default="training/gnn/pathagon.pt")
    result.add_argument("--resume")
    result.add_argument("--size", type=int, default=7)
    result.add_argument("--reserve", type=int, default=0)
    result.add_argument("--hidden", type=int, default=64)
    result.add_argument("--layers", type=int, default=8)
    result.add_argument("--steps", type=int, default=200)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--max-examples", type=int, default=0)
    result.add_argument("--generations", type=int, default=1)
    result.add_argument("--games", type=int, default=8)
    result.add_argument("--simulations", type=int, default=64)
    result.add_argument("--updates", type=int, default=200)
    result.add_argument("--replay-limit", type=int, default=10000)
    result.add_argument("--temperature-moves", type=int, default=8)
    result.add_argument("--seed", type=int, default=20260823)
    result.add_argument("--device", default="auto")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.mode == "warmstart":
        if not args.data:
            raise SystemExit("warmstart requires --data <schema-v2.jsonl>")
        run_warmstart(args)
    else:
        run_alphazero(args)


if __name__ == "__main__":
    main()
