#!/usr/bin/env python3
"""Compare normalized parity output from the three runtime runners."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: compare-parity.py <typescript.json> <python.json> [rust.json ...]")
    outputs = [load(path) for path in sys.argv[1:]]
    baseline = outputs[0]
    for label, output in zip(sys.argv[1:], outputs):
        if output != baseline:
            for index, (expected, actual) in enumerate(zip(baseline, output)):
                if expected != actual:
                    raise AssertionError(f"parity mismatch in {label} case {index} ({expected.get('name')}):\nexpected={json.dumps(expected, sort_keys=True)}\nactual={json.dumps(actual, sort_keys=True)}")
            raise AssertionError(f"parity output length mismatch in {label}: {len(output)} != {len(baseline)}")
    print(f"cross-runtime parity: {len(baseline)} generated cases; TypeScript, Rust, and Python agree")


if __name__ == "__main__":
    main()
