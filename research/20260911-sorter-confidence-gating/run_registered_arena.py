#!/usr/bin/env python3
"""Run the preregistered confidence-gated candidate arena matrix."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-onnx", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx = args.output_dir / Path(protocol["candidate"]["onnx"]).name
    if not onnx.exists():
        shutil.copy2(args.source_onnx, onnx)
    common = [
        str(args.binary), "--jsonl", "--workers", "4",
        "--max-plies", str(protocol["board"]["maxPlies"]),
        "--opening-random-plies", str(protocol["board"]["openingRandomPlies"]),
        "--opponent", protocol["control"]["opponent"].replace("pathfinder-v0.4.0-tactical-filter", "filter-search"),
        "--weight-path", str(protocol["weights"]["path"]),
        "--weight-material", str(protocol["weights"]["material"]),
        "--weight-capture", str(protocol["weights"]["capture"]),
        "--weight-structure", str(protocol["weights"]["structure"]),
        "--weight-threat", str(protocol["weights"]["threat"]),
        "--weight-edge", str(protocol["weights"]["edge"]),
    ]
    sorter = protocol["sorter"]
    summaries = []
    for budget in protocol["budgets"]:
        for seed in protocol["seeds"]:
            name = f"arena-seed{seed}-{budget['name']}-learned"
            output = args.output_dir / f"{name}.jsonl"
            command = common + [
                "--games", str(protocol["gamesPerSeed"]), "--seed", str(seed),
                "--depth", str(budget["depth"]), "--nodes", str(budget["nodes"]),
                "--beam", str(budget["beam"]), "--candidate-id", protocol["candidate"]["id"],
                "--sorter-qadv-onnx", str(onnx), "--sorter-top-k", str(budget["sorterTopK"]),
                "--sorter-root-limit", str(budget["sorterRootLimit"]),
                "--sorter-min-margin", str(sorter["minMargin"]),
                "--sorter-max-heuristic-gap", str(sorter["maxHeuristicGap"]),
            ]
            if sorter["allActions"]:
                command.append("--sorter-all-actions")
            print("$", " ".join(command), flush=True)
            with output.open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.PIPE, text=True, check=False)
            if result.returncode:
                raise SystemExit(f"{name} failed ({result.returncode}):\n{result.stderr}")
            summary = next((json.loads(line) for line in reversed(result.stderr.splitlines()) if line.startswith("{")), None)
            if summary is None:
                raise SystemExit(f"{name} emitted no summary")
            summaries.append({"name": name, "kind": "learned", "budget": budget, "seed": seed, "summary": summary})
    (args.output_dir / "arena-run-summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
