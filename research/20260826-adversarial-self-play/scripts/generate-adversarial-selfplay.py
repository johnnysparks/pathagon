#!/usr/bin/env python3
"""Generate a small, dated batch of targeted Rust QAdv games.

Targeted games are intentionally written outside the primary campaign.  Their
manifest records the profile and model hash so a later training job can add
them through an explicit, capped mixture instead of silently changing the
main distribution.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = REPO_ROOT / "pathagon/engine-rs/target/release/pathagon-selfplay"
DEFAULT_OUTPUT = REPO_ROOT / "research/adversarial/generated/batch-20260826-targeted-v1"

PROFILES: dict[str, dict[str, Any]] = {
    "placement-exploration": {
        "purpose": "broaden early placement decisions while retaining a Pathfinder prior",
        "opening_moves": 20,
        "opening_temperature": 2.4,
        "opening_randomness": 0.55,
        "policy_temperature": 1.20,
        "temperature_moves": 64,
        "pathfinder_guidance": 0.45,
        "placement_guidance": 0.20,
        "pathfinder_temperature": 1.15,
        "qadv_weight": 1.0,
    },
    "ranking-ambiguity": {
        "purpose": "collect nearby-action Q/A rankings instead of only decisive argmax moves",
        "opening_moves": 16,
        "opening_temperature": 1.8,
        "opening_randomness": 0.35,
        "policy_temperature": 1.35,
        "temperature_moves": 64,
        "pathfinder_guidance": 0.35,
        "placement_guidance": 0.25,
        "pathfinder_temperature": 1.15,
        "qadv_weight": 1.0,
    },
    "capture-pressure": {
        "purpose": "increase transition and capture diversity without dropping the Q/A head",
        "opening_moves": 18,
        "opening_temperature": 2.0,
        "opening_randomness": 0.45,
        "policy_temperature": 1.50,
        "temperature_moves": 72,
        "pathfinder_guidance": 0.25,
        "placement_guidance": 0.10,
        "pathfinder_temperature": 1.10,
        "qadv_weight": 1.0,
    },
    "long-horizon": {
        "purpose": "stress late-game Q/A targets and the 196-ply contract boundary",
        "opening_moves": 16,
        "opening_temperature": 1.8,
        "opening_randomness": 0.30,
        "policy_temperature": 1.15,
        "temperature_moves": 48,
        "pathfinder_guidance": 0.50,
        "placement_guidance": 0.30,
        "pathfinder_temperature": 1.15,
        "qadv_weight": 1.0,
    },
}


def run_one(binary: Path, model: Path, output_dir: Path, profile_name: str, seed: int) -> dict[str, Any]:
    profile = PROFILES[profile_name]
    destination = output_dir / f"game-{seed}.json"
    if destination.is_file():
        try:
            payload = json.loads(destination.read_text(encoding="utf-8"))
            if isinstance(payload.get("record"), dict):
                return {"profile": profile_name, "seed": seed, "status": "existing"}
        except (OSError, json.JSONDecodeError):
            destination.unlink(missing_ok=True)
    command = [
        str(binary),
        "--qadv-onnx", str(model),
        "--opponent", "neural",
        "--games", "1",
        "--seed", str(seed),
        "--max-plies", "196",
        "--opening-random-plies", "0",
        "--simulations", "128",
        "--temperature-moves", str(profile["temperature_moves"]),
        "--policy-temperature", str(profile["policy_temperature"]),
        "--opening-moves", str(profile["opening_moves"]),
        "--opening-temperature", str(profile["opening_temperature"]),
        "--opening-randomness", str(profile["opening_randomness"]),
        "--pathfinder-guidance", str(profile["pathfinder_guidance"]),
        "--placement-guidance", str(profile["placement_guidance"]),
        "--pathfinder-temperature", str(profile["pathfinder_temperature"]),
        "--pathfinder-depth", "2",
        "--pathfinder-beam", "8",
        "--pathfinder-nodes", "512",
        "--qadv-weight", str(profile["qadv_weight"]),
        "--workers", "1",
        "--progress-every", "1",
        "--jsonl",
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return {"profile": profile_name, "seed": seed, "status": "failed", "error": (completed.stderr or completed.stdout or "native runner failed").strip()}
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        return {"profile": profile_name, "seed": seed, "status": "failed", "error": f"expected one JSONL record, got {len(lines)} lines"}
    try:
        record = json.loads(lines[0])
    except json.JSONDecodeError as error:
        return {"profile": profile_name, "seed": seed, "status": "failed", "error": f"invalid native JSONL: {error}"}
    payload = {"seed": seed, "profile": profile_name, "record": record}
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return {
        "profile": profile_name,
        "seed": seed,
        "status": "complete",
        "seconds": time.perf_counter() - started,
        "plies": len(record.get("moves", [])),
        "captures": sum(len(move.get("captured", [])) for move in record.get("moves", [])),
        "qCoveredPositions": sum(bool(move.get("actionValues") and move.get("actionVisits")) for move in record.get("moves", [])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profiles", default=",".join(PROFILES), help="comma-separated profile names")
    parser.add_argument("--games-per-profile", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=2026220000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    profile_names = [name.strip() for name in args.profiles.split(",") if name.strip()]
    if not profile_names or any(name not in PROFILES for name in profile_names):
        raise SystemExit(f"profiles must be chosen from: {', '.join(PROFILES)}")
    if args.games_per_profile < 1 or args.workers < 1 or args.seed_start < 0:
        raise SystemExit("games-per-profile, workers, and seed-start must be positive")
    binary = args.binary.resolve()
    model = args.model.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if not binary.is_file() or not model.is_file():
        raise SystemExit("native binary and model must exist")
    jobs = [
        (profile, args.seed_start + profile_index * 100 + index)
        for profile_index, profile in enumerate(profile_names)
        for index in range(args.games_per_profile)
    ]
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    started = time.perf_counter()
    print(json.dumps({"jobs": len(jobs), "profiles": profile_names, "workers": args.workers, "modelSha256": model_hash}, sort_keys=True), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, binary, model, output_dir, profile, seed) for profile, seed in jobs]
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            if result["status"] == "complete":
                print(f"adversarial: profile={result['profile']} seed={result['seed']} plies={result['plies']} captures={result['captures']} q={result['qCoveredPositions']} seconds={result['seconds']:.1f}", flush=True)
            elif result["status"] == "failed":
                print(f"adversarial: profile={result['profile']} seed={result['seed']} failed: {result['error']}", flush=True)
    results.sort(key=lambda result: (result["profile"], result["seed"]))
    manifest = {
        "schemaVersion": 1,
        "status": "complete" if all(result["status"] in {"complete", "existing"} for result in results) else "partial",
        "engine": "rust-native",
        "modelSha256": model_hash,
        "profiles": {name: PROFILES[name] for name in profile_names},
        "jobs": [{"profile": profile, "seed": seed} for profile, seed in jobs],
        "results": results,
        "elapsedSeconds": time.perf_counter() - started,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "completed": sum(result["status"] in {"complete", "existing"} for result in results), "failed": sum(result["status"] == "failed" for result in results), "output": str(output_dir)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
