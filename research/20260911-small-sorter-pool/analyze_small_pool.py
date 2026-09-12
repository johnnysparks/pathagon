#!/usr/bin/env python3
"""Analyze the small-pool candidate against paired controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def read(path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()
def report(path):
    rows = read(path); cid = rows[0]["agents"]["light"]; w=l=d=0
    for row in rows:
        light = row["agents"]["light"] == cid; winner = row.get("winner")
        if winner is None: d += 1
        elif (winner == "light") == light: w += 1
        else: l += 1
    m = re.search(r"arena-seed(\d+)-(\w+)-", path.name)
    return {"file":str(path),"seed":int(m.group(1)) if m else None,"budget":m.group(2) if m else None,"games":len(rows),"wins":w,"losses":l,"draws":d,"gamePoints":(w+.5*d)/len(rows),"plies":sum(r["plies"] for r in rows),"nodes":sum(x["nodes"] for r in rows for x in r["moves"])}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--arena-dir",type=Path,required=True); p.add_argument("--control-dir",type=Path,required=True); p.add_argument("--protocol",type=Path,required=True); p.add_argument("--source-onnx",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    cand=[report(x) for x in sorted(a.arena_dir.glob("arena-seed*-learned.jsonl"))]; ctrl=[report(x) for x in sorted(a.control_dir.glob("arena-seed*-*-control.jsonl"))]; ag=[]
    for budget in sorted({x["budget"] for x in cand}):
        c=[x for x in cand if x["budget"]==budget]; t=[x for x in ctrl if x["budget"]==budget]
        def pts(xs): return (sum(x["wins"] for x in xs)+.5*sum(x["draws"] for x in xs))/sum(x["games"] for x in xs)
        cp,tp=pts(c),pts(t); ag.append({"budget":budget,"seeds":sorted(x["seed"] for x in c),"candidate":{"wins":sum(x["wins"] for x in c),"losses":sum(x["losses"] for x in c),"draws":sum(x["draws"] for x in c),"games":sum(x["games"] for x in c),"gamePoints":cp},"control":{"wins":sum(x["wins"] for x in t),"losses":sum(x["losses"] for x in t),"draws":sum(x["draws"] for x in t),"games":sum(x["games"] for x in t),"gamePoints":tp},"difference":cp-tp})
    audits={x.stem.removesuffix(".audit"):json.loads(x.read_text()) for x in sorted(a.arena_dir.glob("arena-*.audit.json"))}
    out={"mode":"relative-regret-head-small-pool","protocol":json.loads(a.protocol.read_text()),"artifacts":{"sourceOnnx":sha256(a.source_onnx)},"arenas":cand,"controls":ctrl,"arenaAggregates":ag,"audits":audits}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print(json.dumps(out,sort_keys=True))
if __name__ == "__main__": main()
