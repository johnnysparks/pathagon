#!/usr/bin/env python3
"""Analyze promote-only arenas, including completed-depth telemetry."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def report(path: Path) -> dict:
    rows=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]; cid=rows[0]["agents"]["light"]; w=l=d=0; depths=[]; nodes=[]
    for row in rows:
        light=row["agents"]["light"]==cid; winner=row.get("winner")
        if winner is None:d+=1
        elif (winner=="light")==light:w+=1
        else:l+=1
        depths += [m["completedDepth"] for m in row["moves"]]; nodes += [m["nodes"] for m in row["moves"]]
    return {"file":str(path),"games":len(rows),"wins":w,"losses":l,"draws":d,"gamePoints":(w+.5*d)/len(rows),"meanCompletedDepth":statistics.mean(depths),"meanNodes":statistics.mean(nodes),"plies":sum(r["plies"] for r in rows)}


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--arena-dir",type=Path,required=True); p.add_argument("--protocol",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    cand=[report(x) for x in sorted(a.arena_dir.glob("arena-seed*-learned.jsonl"))]; ctrl=[report(x) for x in sorted(a.arena_dir.glob("arena-seed*-control.jsonl"))]
    def aggregate(xs):
        w=sum(x["wins"] for x in xs); l=sum(x["losses"] for x in xs); d=sum(x["draws"] for x in xs); return {"wins":w,"losses":l,"draws":d,"games":w+l+d,"gamePoints":(w+.5*d)/(w+l+d),"meanCompletedDepth":statistics.mean(x["meanCompletedDepth"] for x in xs),"meanNodes":statistics.mean(x["meanNodes"] for x in xs)}
    audits={x.stem.removesuffix(".audit"):json.loads(x.read_text()) for x in sorted(a.arena_dir.glob("arena-*.audit.json"))}; out={"mode":"envelope-matched-promote-only","protocol":json.loads(a.protocol.read_text()),"candidate":aggregate(cand),"control":aggregate(ctrl),"difference":aggregate(cand)["gamePoints"]-aggregate(ctrl)["gamePoints"],"arenas":cand,"controls":ctrl,"audits":audits}; a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print(json.dumps(out,sort_keys=True))
if __name__ == "__main__": main()
