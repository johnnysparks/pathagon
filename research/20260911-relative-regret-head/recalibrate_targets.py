#!/usr/bin/env python3
"""Recompute within-position regret targets without absolute-score saturation."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def relative(scores: list[float]) -> tuple[list[float], float, float]:
    center = float(statistics.median(scores))
    mad = float(statistics.median([abs(score - center) for score in scores]))
    scale = max(250.0, 1.4826 * mad)
    return [max(-1.0, min(1.0, (score - center) / scale)) for score in scores], center, scale


def softmax(values: list[float], temperature: float = 0.35) -> list[float]:
    maximum = max(values)
    weights = [(value - maximum) / temperature for value in values]
    exp_values = [__import__("math").exp(value) for value in weights]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = []
    for line_number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        scores = [float(score) for score in row["teacherScores"]]
        values, center, scale = relative(scores)
        row["teacherValues"] = values
        row["softPolicy"] = softmax(values)
        row["teacherValueCenter"] = center
        row["teacherValueScale"] = scale
        row["teacherValueMethod"] = "within-position-median-mad-v1"
        row["targetProvenance"] = str(args.input)
        if len(row["candidateActions"]) != len(values):
            raise ValueError(f"{args.input}:{line_number}: teacher/action length mismatch")
        output.append(json.dumps(row, separators=(",", ":")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(output), "output": str(args.output), "temperature": 0.35}))


if __name__ == "__main__":
    main()
