#!/usr/bin/env python3
"""Cross-check Python QAdv/Pathfinder signals against the native harness."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from research.gnn.game import Action, BoardConfig, GameState
from research.gnn.pathfinder import PathfinderGuide
from research.gnn.train import load_model


# A non-terminal 28-ply placement position: both sides have exhausted their
# reserves, so the next state exercises the full relocation action space.
DEFAULT_SEQUENCE = "P7,P22,P8,P23,P9,P24,P10,P29,P11,P30,P12,P31,P13,P36,P14,P37,P15,P38,P16,P43,P17,P44,P18,P45,P19,P25,P20,P32"


def state_for(sequence: str) -> GameState:
    state = GameState.initial(BoardConfig(size=7, reserve_per_player=14, ply_limit=196))
    for token in filter(None, sequence.split(",")):
        if token.startswith("P"):
            action = Action.place(int(token[1:]))
        else:
            source, destination = token[1:].split(">", 1)
            action = Action.relocate(int(source), int(destination))
        if action not in state.legal_actions():
            raise ValueError(f"illegal action in sequence: {token}")
        state = state.apply_legal(action)
    return state


def run_native(binary: Path, arguments: list[str]) -> dict:
    result = subprocess.run([str(binary), *arguments], cwd=REPO_ROOT, check=True, text=True, capture_output=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=Path("pathagon/engine-rs/target/release/pathagon-selfplay"))
    parser.add_argument("--sequence", action="append", default=["", DEFAULT_SEQUENCE])
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--pathfinder-depth", type=int, default=2)
    parser.add_argument("--pathfinder-beam", type=int, default=8)
    parser.add_argument("--pathfinder-nodes", type=int, default=512)
    args = parser.parse_args()

    model = load_model(args.checkpoint.resolve(), torch.device("cpu"), qadv=True)
    model.eval()
    binary = args.binary.resolve()
    onnx = args.onnx.resolve()
    largest_q_error = 0.0
    largest_path_error = 0.0
    for sequence in args.sequence:
        state = state_for(sequence)
        actions = list(state.legal_actions())
        with torch.no_grad():
            policy, value, q_values, _ = model.policy_value_q(state, actions)
        native_q = run_native(binary, [
            "--eval-only", "--qadv-onnx", str(onnx), "--eval-count", str(args.count),
            *( ["--eval-sequence", sequence] if sequence else [] ),
        ])
        expected_policy = [float(value) for value in policy[: args.count]]
        expected_q = [float(value) for value in q_values[: args.count]]
        largest_q_error = max(
            largest_q_error,
            max((abs(left - right) for left, right in zip(expected_policy, native_q["policyFirst"])), default=0.0),
            abs(float(value) - native_q["value"]),
            max((abs(left - right) for left, right in zip(expected_q, native_q["qFirst"])), default=0.0),
        )
        guide = PathfinderGuide(args.pathfinder_depth, args.pathfinder_beam, args.pathfinder_nodes)
        expected_scores = [float(score) for score in guide.score_actions(state, actions)[: args.count]]
        native_path = run_native(binary, [
            "--pathfinder-only", "--eval-count", str(args.count),
            "--pathfinder-depth", str(args.pathfinder_depth),
            "--pathfinder-beam", str(args.pathfinder_beam),
            "--pathfinder-nodes", str(args.pathfinder_nodes),
            *( ["--eval-sequence", sequence] if sequence else [] ),
        ])
        largest_path_error = max(
            largest_path_error,
            max((abs(left - right) for left, right in zip(expected_scores, native_path["scoresFirst"])), default=0.0),
        )
        print(json.dumps({
            "sequence": sequence or "<initial>",
            "legalActions": len(actions),
            "qMaxAbsError": largest_q_error,
            "pathfinderMaxAbsError": largest_path_error,
        }, sort_keys=True))
    if largest_q_error > 1e-4 or largest_path_error > 1e-4:
        raise SystemExit("native parity check failed")
    print(json.dumps({"status": "pass", "qMaxAbsError": largest_q_error, "pathfinderMaxAbsError": largest_path_error}, sort_keys=True))


if __name__ == "__main__":
    main()
