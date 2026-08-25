"""Train the GNN warm start or run compact AlphaZero-style generations."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .data import ReplayExample, action_index, load_replay_examples
from .cnn_model import PathagonCNN
from .game import BoardConfig, GameState
from .model import PathagonGNN
from .selfplay import SearchExample, game_record, generate_game
from .symmetry import Symmetry, sample_symmetry, transform_action, transform_state


TrainingProgress = Callable[[int, int, float, float], None]
NeuralModel = PathagonGNN | PathagonCNN
_SELFPLAY_MODEL: Optional[NeuralModel] = None
_SELFPLAY_CONFIG: Optional[BoardConfig] = None
_SELFPLAY_SIMULATIONS = 0
_SELFPLAY_TEMPERATURE_MOVES = 0
_SELFPLAY_GENERATION = 0
_SELFPLAY_GENERATIONS = 0
_SELFPLAY_GAMES = 0


def build_model(
    architecture: str,
    hidden_size: int,
    message_layers: int,
    residual_blocks: int,
    board_size: int,
) -> NeuralModel:
    if architecture == "cnn":
        return PathagonCNN(hidden_size=hidden_size, residual_blocks=residual_blocks, board_size=board_size)
    if architecture == "gnn":
        return PathagonGNN(hidden_size=hidden_size, message_layers=message_layers)
    raise ValueError(f"unsupported architecture: {architecture}")


def transform_replay_example(example: ReplayExample, symmetry: Symmetry) -> ReplayExample:
    """Transform a replay example without changing its player-relative value."""

    return ReplayExample(
        state=transform_state(example.state, symmetry),
        action=transform_action(example.action, example.state.config, symmetry),
        value=example.value,
        seed=example.seed,
        policy=example.policy,
        policy_actions=None
        if example.policy_actions is None
        else tuple(transform_action(action, example.state.config, symmetry) for action in example.policy_actions),
        action_values=example.action_values,
        action_visits=example.action_visits,
        action_value_actions=None
        if example.action_value_actions is None
        else tuple(transform_action(action, example.state.config, symmetry) for action in example.action_value_actions),
    )


def transform_search_example(example: SearchExample, symmetry: Symmetry) -> SearchExample:
    """Transform an MCTS target while preserving action/probability alignment."""

    return SearchExample(
        state=transform_state(example.state, symmetry),
        actions=tuple(transform_action(action, example.state.config, symmetry) for action in example.actions),
        policy=example.policy,
        selected_action=transform_action(example.selected_action, example.state.config, symmetry),
        value=example.value,
        action_values=example.action_values,
        action_visits=example.action_visits,
    )


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def model_state_hash(model: NeuralModel) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return f"sha256:{digest.hexdigest()}"


def save_model(model: NeuralModel, path: Path, metadata: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_config": model.config_dict(), "state_dict": model.state_dict(), "metadata": metadata}, path)


def load_model(path: Path, device: torch.device) -> NeuralModel:
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["model_config"]
    architecture = config.get("architecture", "residual-mean-message-passing")
    if architecture == "residual-cnn-7x7":
        model = PathagonCNN(
            hidden_size=config["hidden_size"],
            residual_blocks=config["residual_blocks"],
            board_size=config.get("board_size", 7),
        ).to(device)
    else:
        model = PathagonGNN(config["hidden_size"], config["message_layers"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def train_replay(
    model: NeuralModel,
    examples: Sequence[ReplayExample],
    steps: int,
    learning_rate: float,
    seed: int,
    progress: Optional[TrainingProgress] = None,
    symmetry_augmentation: bool = True,
) -> Tuple[float, float]:
    if not examples:
        raise ValueError("replay dataset is empty")
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    policy_total = 0.0
    value_total = 0.0
    for step in range(1, steps + 1):
        example = examples[rng.randrange(len(examples))]
        if symmetry_augmentation:
            example = transform_replay_example(example, sample_symmetry(rng))
        actions = list(example.state.legal_actions())
        logits, value = model.policy_value(example.state, actions)
        expected_value = torch.tensor(example.value, dtype=value.dtype, device=value.device)
        if example.policy is None:
            target = torch.tensor([action_index(example.state, example.action)], dtype=torch.long, device=logits.device)
            policy_loss = F.cross_entropy(logits.unsqueeze(0), target)
        else:
            if example.policy_actions is None:
                raise ValueError("replay policy is missing its action alignment")
            policy_by_action = dict(zip(example.policy_actions, example.policy))
            target_policy = torch.tensor(
                [policy_by_action[action] for action in actions],
                dtype=logits.dtype,
                device=logits.device,
            )
            target_policy = target_policy / target_policy.sum()
            policy_loss = -(target_policy * F.log_softmax(logits, dim=0)).sum()
        value_loss = F.mse_loss(value, expected_value)
        loss = policy_loss + value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        policy_total += float(policy_loss.detach().cpu())
        value_total += float(value_loss.detach().cpu())
        if progress is not None:
            progress(step, steps, policy_total / step, value_total / step)
    return policy_total / steps, value_total / steps


def train_search_examples(
    model: NeuralModel,
    examples: Sequence[SearchExample],
    steps: int,
    learning_rate: float,
    seed: int,
    progress: Optional[TrainingProgress] = None,
    symmetry_augmentation: bool = True,
) -> Tuple[float, float]:
    if not examples:
        raise ValueError("MCTS dataset is empty")
    rng = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    policy_total = 0.0
    value_total = 0.0
    for step in range(1, steps + 1):
        example = examples[rng.randrange(len(examples))]
        if symmetry_augmentation:
            example = transform_search_example(example, sample_symmetry(rng))
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
        if progress is not None:
            progress(step, steps, policy_total / step, value_total / step)
    return policy_total / steps, value_total / steps


def training_progress(label: str) -> TrainingProgress:
    started = time.perf_counter()

    def report(step: int, total: int, policy_loss: float, value_loss: float) -> None:
        interval = max(1, total // 100)
        if step != 1 and step != total and step % interval != 0:
            return
        elapsed = time.perf_counter() - started
        rate = step / elapsed if elapsed > 0 else 0.0
        print(
            f"{label}: step {step}/{total} ({step / total:.1%}) "
            f"policy_loss={policy_loss:.4f} value_loss={value_loss:.4f} "
            f"elapsed={elapsed:.1f}s steps_per_second={rate:.2f}",
            file=sys.stderr,
            flush=True,
        )

    return report


def initialize_selfplay_worker(
    model_config: Dict,
    state_dict: Dict[str, torch.Tensor],
    board_size: int,
    reserve_per_player: int,
    max_plies: int,
    simulations: int,
    temperature_moves: int,
    device_name: str,
    generation: int,
    generations: int,
    games: int,
) -> None:
    global _SELFPLAY_MODEL
    global _SELFPLAY_CONFIG
    global _SELFPLAY_SIMULATIONS
    global _SELFPLAY_TEMPERATURE_MOVES
    global _SELFPLAY_GENERATION
    global _SELFPLAY_GENERATIONS
    global _SELFPLAY_GAMES

    device = choose_device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(1)
    architecture = "cnn" if model_config.get("architecture") == "residual-cnn-7x7" else "gnn"
    model = build_model(
        architecture=architecture,
        hidden_size=model_config["hidden_size"],
        message_layers=model_config.get("message_layers", 8),
        residual_blocks=model_config.get("residual_blocks", 4),
        board_size=model_config.get("board_size", board_size),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    _SELFPLAY_MODEL = model
    _SELFPLAY_CONFIG = BoardConfig(board_size, reserve_per_player, max_plies)
    _SELFPLAY_SIMULATIONS = simulations
    _SELFPLAY_TEMPERATURE_MOVES = temperature_moves
    _SELFPLAY_GENERATION = generation
    _SELFPLAY_GENERATIONS = generations
    _SELFPLAY_GAMES = games


def generate_game_worker(game_index: int, game_seed: int) -> Tuple[int, List[SearchExample], GameState]:
    if _SELFPLAY_MODEL is None or _SELFPLAY_CONFIG is None:
        raise RuntimeError("self-play worker was not initialized")
    torch.manual_seed(game_seed)
    game_started = time.perf_counter()

    def game_progress(state: GameState) -> None:
        elapsed = time.perf_counter() - game_started
        print(
            f"alphazero: generation {_SELFPLAY_GENERATION + 1}/{_SELFPLAY_GENERATIONS}: "
            f"game {game_index + 1}/{_SELFPLAY_GAMES} ply {state.ply}/{_SELFPLAY_CONFIG.max_plies} "
            f"elapsed={elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    examples, final_state = generate_game(
        _SELFPLAY_MODEL,
        _SELFPLAY_CONFIG,
        simulations=_SELFPLAY_SIMULATIONS,
        temperature_moves=_SELFPLAY_TEMPERATURE_MOVES,
        seed=game_seed,
        add_root_noise=True,
        progress=game_progress,
    )
    return game_index, examples, final_state


def run_warmstart(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    if args.architecture == "cnn" and args.size != 7:
        raise SystemExit("--architecture cnn requires --size 7")
    config = BoardConfig(args.size, args.reserve)
    load_started = time.perf_counter()
    print(f"warmstart: loading and validating replay from {args.data}", file=sys.stderr, flush=True)

    def replay_progress(records: int, examples: int) -> None:
        if records != 1 and records % 50 != 0:
            return
        elapsed = time.perf_counter() - load_started
        print(
            f"warmstart: validated {records} games / {examples} examples in {elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    examples = load_replay_examples(Path(args.data), config, progress=replay_progress)
    if args.max_examples:
        examples = examples[: args.max_examples]
    print(
        f"warmstart: training {len(examples)} examples for {args.steps} steps on {device}",
        file=sys.stderr,
        flush=True,
    )
    model = build_model(args.architecture, args.hidden, args.layers, args.cnn_blocks, args.size).to(device)
    policy_loss, value_loss = train_replay(
        model,
        examples,
        args.steps,
        args.learning_rate,
        args.seed,
        progress=training_progress("warmstart"),
        symmetry_augmentation=args.symmetry_augmentation,
    )
    metadata = {
        "mode": "replay-warmstart",
        "data": str(args.data),
        "examples": len(examples),
        "architecture": model.config_dict()["architecture"],
        "board_size": args.size,
        "reserve_per_player": config.reserve_per_player,
        "symmetry_augmentation": args.symmetry_augmentation,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
    }
    save_model(model, Path(args.out), metadata)
    print(json.dumps(metadata | {"out": args.out, "device": str(device)}, sort_keys=True))


def run_alphazero(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    selfplay_device = choose_device(args.selfplay_device)
    torch.manual_seed(args.seed)
    if args.architecture == "cnn" and args.size != 7:
        raise SystemExit("--architecture cnn requires --size 7")
    config = BoardConfig(args.size, args.reserve, args.max_plies)
    if args.games < 1:
        raise SystemExit("--games must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    if args.updates < 0:
        raise SystemExit("--updates cannot be negative")
    workers = min(args.workers, args.games)
    model = load_model(Path(args.resume), device) if args.resume else build_model(
        args.architecture,
        args.hidden,
        args.layers,
        args.cnn_blocks,
        args.size,
    ).to(device)
    if isinstance(model, PathagonCNN) and config.size != model.board_size:
        raise SystemExit(f"CNN checkpoint requires a {model.board_size}x{model.board_size} board")
    model.eval()
    history: List[SearchExample] = []
    games_path = Path(args.games_out) if args.games_out else None
    if games_path:
        games_path.parent.mkdir(parents=True, exist_ok=True)
    for generation in range(args.generations):
        generation_started = time.perf_counter()
        print(
            f"alphazero: generation {generation + 1}/{args.generations}: "
            f"generating {args.games} games with {args.simulations} simulations each "
            f"using {workers} {selfplay_device} worker{'s' if workers != 1 else ''}; training on {device}",
            file=sys.stderr,
            flush=True,
        )
        generated: List[SearchExample] = []
        lengths = [0] * args.games
        results: Dict[int, List[SearchExample]] = {}
        final_states: Dict[int, GameState] = {}
        state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        generation_model_hash = model_state_hash(model)
        process_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=process_context,
            initializer=initialize_selfplay_worker,
            initargs=(
                model.config_dict(),
                state_dict,
                config.size,
                config.reserve_per_player,
                config.max_plies,
                args.simulations,
                args.temperature_moves,
                str(selfplay_device),
                generation,
                args.generations,
                args.games,
            ),
        ) as executor:
            futures = {
                executor.submit(
                    generate_game_worker,
                    game_index,
                    args.seed + generation * args.games + game_index,
                ): game_index
                for game_index in range(args.games)
            }
            for future in as_completed(futures):
                game_index, examples, final_state = future.result()
                results[game_index] = examples
                final_states[game_index] = final_state
                lengths[game_index] = final_state.ply
                elapsed = time.perf_counter() - generation_started
                completed_examples = sum(len(result) for result in results.values())
                print(
                    f"alphazero: generation {generation + 1}/{args.generations}: "
                    f"game {game_index + 1}/{args.games} complete "
                    f"({final_state.ply} plies, {completed_examples} completed examples, elapsed={elapsed:.1f}s)",
                    file=sys.stderr,
                    flush=True,
                )
        for game_index in range(args.games):
            examples = results[game_index]
            final_state = final_states[game_index]
            game_seed = args.seed + generation * args.games + game_index
            generated.extend(examples)
            if games_path:
                with games_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(
                        game_record(
                            examples,
                            final_state,
                            game_seed,
                            args.simulations,
                            generation_model_hash,
                            args.agent_id,
                            args.agent_name,
                            args.agent_version,
                            args.agent_kind,
                            args.agent_engine,
                        ),
                        sort_keys=True,
                    ) + "\n")
        history.extend(generated)
        if args.replay_limit and len(history) > args.replay_limit:
            history = history[-args.replay_limit :]
        if args.updates:
            policy_loss, value_loss = train_search_examples(
                model,
                history,
                args.updates,
                args.learning_rate,
                args.seed + generation,
                progress=training_progress(f"alphazero: generation {generation + 1} training"),
                symmetry_augmentation=args.symmetry_augmentation,
            )
        else:
            policy_loss, value_loss = None, None
        model.eval()
        metadata = {
            "mode": "alphazero-generation",
            "architecture": model.config_dict()["architecture"],
            "generation": generation,
            "board_size": args.size,
            "reserve_per_player": config.reserve_per_player,
            "max_plies": config.max_plies,
            "games": args.games,
            "workers": workers,
            "selfplay_device": str(selfplay_device),
            "simulations": args.simulations,
            "temperature_moves": args.temperature_moves,
            "updates": args.updates,
            "replay_limit": args.replay_limit,
            "symmetry_augmentation": args.symmetry_augmentation,
            "examples": len(generated),
            "replay_buffer": len(history),
            "average_plies": sum(lengths) / len(lengths) if lengths else 0.0,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "games_out": str(games_path) if games_path else None,
        }
        save_model(model, Path(args.out), metadata)
        print(json.dumps(metadata | {"out": args.out, "device": str(device)}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("mode", choices=("warmstart", "alphazero"))
    result.add_argument("--data", help="schema-v2 JSONL for replay warm-start")
    result.add_argument("--out", default="research/runs/gnn/pathagon.pt")
    result.add_argument("--games-out", help="append generated schema-v2 games to this JSONL file")
    result.add_argument("--resume")
    result.add_argument("--architecture", choices=("gnn", "cnn"), default="gnn")
    result.add_argument("--size", type=int, default=7)
    result.add_argument("--reserve", type=int, default=0)
    result.add_argument("--max-plies", type=int, default=0, help="ply cap for AlphaZero self-play games (0 uses the board default: 196 on 7x7)")
    result.add_argument("--hidden", type=int, default=64)
    result.add_argument("--layers", type=int, default=8)
    result.add_argument("--cnn-blocks", type=int, default=4, help="residual blocks for the fixed-7x7 CNN")
    result.add_argument("--steps", type=int, default=200)
    result.add_argument("--learning-rate", type=float, default=3e-4)
    result.add_argument("--max-examples", type=int, default=0)
    result.add_argument("--generations", type=int, default=1)
    result.add_argument("--games", type=int, default=8)
    result.add_argument("--workers", type=int, default=1, help="parallel self-play worker processes")
    result.add_argument("--selfplay-device", default="cpu", help="device used by self-play workers")
    result.add_argument("--simulations", type=int, default=64, help="PUCT simulations per move (32-64 recommended for training)")
    result.add_argument("--updates", type=int, default=10_000, help="optimizer updates per generation (0 generates data without training)")
    result.add_argument("--replay-limit", type=int, default=100_000, help="maximum positions retained in the replay buffer")
    result.add_argument("--temperature-moves", type=int, default=8)
    result.add_argument("--agent-id", default="python-gnn-puct-v0.1.0")
    result.add_argument("--agent-name", default="Python GNN PUCT")
    result.add_argument("--agent-version", default="0.1.0")
    result.add_argument("--agent-kind", default="puct")
    result.add_argument("--agent-engine", default="python-gnn")
    result.add_argument(
        "--no-symmetry-augmentation",
        dest="symmetry_augmentation",
        action="store_false",
        default=True,
        help="disable D4 board-symmetry augmentation during optimizer updates",
    )
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
