#!/usr/bin/env python3
"""Analyze pool-limited sorter games against paired control logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def games(path: Path) -> dict:
    rows = load(path)
    candidate_id = rows[0]["agents"]["light"]
    wins = losses = draws = 0
    for row in rows:
        candidate_light = row["agents"]["light"] == candidate_id
        winner = row.get("winner")
        if winner is None:
            draws += 1
        elif (winner == "light") == candidate_light:
            wins += 1
        else:
            losses += 1
    match = re.search(r"arena-seed(\d+)-(\w+)-", path.name)
    return {
        "file": str(path),
        "seed": int(match.group(1)) if match else None,
        "budget": match.group(2) if match else None,
        "games": len(rows),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "gamePoints": (wins + 0.5 * draws) / len(rows),
        "plies": sum(row["plies"] for row in rows),
        "nodes": sum(move["nodes"] for row in rows for move in row["moves"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arena-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate = [games(path) for path in sorted(args.arena_dir.glob("arena-seed*-learned.jsonl"))]
    control = [games(path) for path in sorted(args.control_dir.glob("arena-seed*-*-control.jsonl"))]
    aggregates = []
    for budget in sorted({item["budget"] for item in candidate}):
        left = [item for item in candidate if item["budget"] == budget]
        right = [item for item in control if item["budget"] == budget]
        cw = sum(item["wins"] for item in left)
        cl = sum(item["losses"] for item in left)
        cd = sum(item["draws"] for item in left)
        tw = sum(item["wins"] for item in right)
        tl = sum(item["losses"] for item in right)
        td = sum(item["draws"] for item in right)
        cpoints = (cw + 0.5 * cd) / (cw + cl + cd)
        tpoints = (tw + 0.5 * td) / (tw + tl + td)
        aggregates.append({
            "budget": budget,
            "seeds": sorted(item["seed"] for item in left),
            "candidate": {"wins": cw, "losses": cl, "draws": cd, "games": cw + cl + cd, "gamePoints": cpoints},
            "control": {"wins": tw, "losses": tl, "draws": td, "games": tw + tl + td, "gamePoints": tpoints},
            "difference": cpoints - tpoints,
        })
    audits = {}
    for path in sorted(args.arena_dir.glob("arena-*.audit.json")):
        audits[path.stem.removesuffix(".audit")] = json.loads(path.read_text(encoding="utf-8"))
    report = {
        "mode": "relative-regret-head-pool-limited",
        "protocol": json.loads(args.protocol.read_text(encoding="utf-8")),
        "artifacts": {"sourceOnnx": sha256(args.source_onnx)},
        "arenas": candidate,
        "controls": control,
        "arenaAggregates": aggregates,
        "audits": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
