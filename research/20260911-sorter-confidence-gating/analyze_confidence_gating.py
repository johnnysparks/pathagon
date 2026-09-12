#!/usr/bin/env python3
"""Analyze confidence-gated sorter games against paired control logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def report(path: Path) -> dict:
    rows = load(path)
    candidate = rows[0]["agents"]["light"]
    w = l = d = 0
    for row in rows:
        light = row["agents"]["light"] == candidate
        winner = row.get("winner")
        if winner is None:
            d += 1
        elif (winner == "light") == light:
            w += 1
        else:
            l += 1
    m = re.search(r"arena-seed(\d+)-(\w+)-", path.name)
    return {"file": str(path), "seed": int(m.group(1)) if m else None, "budget": m.group(2) if m else None,
            "games": len(rows), "wins": w, "losses": l, "draws": d,
            "gamePoints": (w + 0.5 * d) / len(rows), "plies": sum(r["plies"] for r in rows),
            "nodes": sum(move["nodes"] for r in rows for move in r["moves"])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arena-dir", type=Path, required=True)
    p.add_argument("--control-dir", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--source-onnx", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cand = [report(x) for x in sorted(args.arena_dir.glob("arena-seed*-learned.jsonl"))]
    ctrl = [report(x) for x in sorted(args.control_dir.glob("arena-seed*-*-control.jsonl"))]
    aggregates = []
    for budget in sorted({x["budget"] for x in cand}):
        c = [x for x in cand if x["budget"] == budget]
        t = [x for x in ctrl if x["budget"] == budget]
        def points(items):
            w = sum(x["wins"] for x in items); d = sum(x["draws"] for x in items)
            return (w + 0.5 * d) / sum(x["games"] for x in items)
        cp, tp = points(c), points(t)
        aggregates.append({"budget": budget, "seeds": sorted(x["seed"] for x in c),
                           "candidate": {"wins": sum(x["wins"] for x in c), "losses": sum(x["losses"] for x in c), "draws": sum(x["draws"] for x in c), "games": sum(x["games"] for x in c), "gamePoints": cp},
                           "control": {"wins": sum(x["wins"] for x in t), "losses": sum(x["losses"] for x in t), "draws": sum(x["draws"] for x in t), "games": sum(x["games"] for x in t), "gamePoints": tp},
                           "difference": cp - tp})
    audits = {x.stem.removesuffix(".audit"): json.loads(x.read_text()) for x in sorted(args.arena_dir.glob("arena-*.audit.json"))}
    output = {"mode": "relative-regret-head-confidence-gated", "protocol": json.loads(args.protocol.read_text()),
              "artifacts": {"sourceOnnx": digest(args.source_onnx)}, "arenas": cand, "controls": ctrl,
              "arenaAggregates": aggregates, "audits": audits}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
