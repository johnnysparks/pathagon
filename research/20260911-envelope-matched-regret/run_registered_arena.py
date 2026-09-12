#!/usr/bin/env python3
"""Run the preregistered 32k envelope-matched arena."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--binary", type=Path, required=True)
    p.add_argument("--candidate-onnx", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    arena = protocol["arena"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx = args.output_dir / Path(protocol["candidate"]["onnx"]).name
    if not onnx.exists(): shutil.copy2(args.candidate_onnx, onnx)
    common = [str(args.binary), "--jsonl", "--workers", "4", "--max-plies", str(arena["maxPlies"]),
              "--opening-random-plies", str(arena["openingRandomPlies"]), "--opponent", "filter-search",
              "--weight-path", str(protocol["teacher"]["weights"]["path"]), "--weight-material", str(protocol["teacher"]["weights"]["material"]),
              "--weight-capture", str(protocol["teacher"]["weights"]["capture"]), "--weight-structure", str(protocol["teacher"]["weights"]["structure"]),
              "--weight-threat", str(protocol["teacher"]["weights"]["threat"]), "--weight-edge", str(protocol["teacher"]["weights"]["edge"])]
    summaries = []
    for seed in protocol["arena"]["seeds"]:
        for kind in ("learned", "control"):
            name = f"arena-seed{seed}-32k-{kind}"; output = args.output_dir / f"{name}.jsonl"
            cmd = common + ["--games", str(arena["gamesPerSeed"]), "--seed", str(seed), "--depth", str(arena["depth"]),
                            "--nodes", str(arena["nodes"]), "--beam", str(arena["beam"]), "--candidate-id",
                            protocol["candidate"]["id"] if kind == "learned" else protocol["control"]["id"]]
            if kind == "learned":
                cmd += ["--sorter-qadv-onnx", str(onnx), "--sorter-all-actions", "--sorter-top-k", str(arena["sorterTopK"]), "--sorter-root-limit", str(arena["sorterRootLimit"])]
            print("$", " ".join(cmd), flush=True)
            with output.open("w") as stream: result = subprocess.run(cmd, stdout=stream, stderr=subprocess.PIPE, text=True)
            if result.returncode: raise SystemExit(f"{name} failed ({result.returncode}):\n{result.stderr}")
            summary = next((json.loads(line) for line in reversed(result.stderr.splitlines()) if line.startswith("{")), None)
            if summary is None: raise SystemExit(f"{name} emitted no summary")
            summaries.append({"name": name, "kind": kind, "seed": seed, "summary": summary})
    (args.output_dir / "arena-run-summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
