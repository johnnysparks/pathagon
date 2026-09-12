#!/usr/bin/env python3
"""Summarize the 32k envelope-matched arena and replay audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def games(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    cid = rows[0]["agents"]["light"]; w = l = d = 0
    for row in rows:
        light = row["agents"]["light"] == cid; winner = row.get("winner")
        if winner is None: d += 1
        elif (winner == "light") == light: w += 1
        else: l += 1
    return {"file": str(path), "games": len(rows), "wins": w, "losses": l, "draws": d, "gamePoints": (w + .5*d)/len(rows), "plies": sum(r["plies"] for r in rows), "nodes": sum(m["nodes"] for r in rows for m in r["moves"])}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--arena-dir", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--onnx", type=Path, required=True); p.add_argument("--training", type=Path, required=True); p.add_argument("--targets", type=Path, required=True); p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    candidate = [games(x) for x in sorted(a.arena_dir.glob("arena-seed*-learned.jsonl"))]; control = [games(x) for x in sorted(a.arena_dir.glob("arena-seed*-control.jsonl"))]
    cw, cl, cd = sum(x["wins"] for x in candidate), sum(x["losses"] for x in candidate), sum(x["draws"] for x in candidate)
    tw, tl, td = sum(x["wins"] for x in control), sum(x["losses"] for x in control), sum(x["draws"] for x in control)
    cp, tp = (cw + .5*cd)/(cw+cl+cd), (tw + .5*td)/(tw+tl+td)
    audits = {x.stem.removesuffix(".audit"): json.loads(x.read_text()) for x in sorted(a.arena_dir.glob("arena-*.audit.json"))}
    out = {"mode": "envelope-matched-regret", "protocol": json.loads(a.protocol.read_text()), "training": json.loads(a.training.read_text()), "artifacts": {"targets": sha256(a.targets), "checkpoint": sha256(a.checkpoint), "onnx": sha256(a.onnx)}, "arenas": candidate, "controls": control, "aggregate": {"candidate": {"wins":cw,"losses":cl,"draws":cd,"games":cw+cl+cd,"gamePoints":cp},"control":{"wins":tw,"losses":tl,"draws":td,"games":tw+tl+td,"gamePoints":tp},"difference":cp-tp}, "audits": audits}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(out, indent=2, sort_keys=True)+"\n"); print(json.dumps(out, sort_keys=True))


if __name__ == "__main__": main()
