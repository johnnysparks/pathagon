#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PATHAGON_PYTHON:-${project_dir}/.venv-pathagon-gnn/bin/python}"
node_bin="${PATHAGON_NODE:-$(command -v node)}"

if [[ ! -x "$python_bin" ]]; then
  echo "Python parity environment is missing: $python_bin" >&2
  exit 2
fi
if [[ -z "$node_bin" || ! -x "$node_bin" ]]; then
  echo "Node parity runtime is missing" >&2
  exit 2
fi

fixture="$(mktemp "${TMPDIR:-/tmp}/pathagon-parity.XXXXXX.json")"
typescript_output="$(mktemp "${TMPDIR:-/tmp}/pathagon-parity-ts.XXXXXX.json")"
python_output="$(mktemp "${TMPDIR:-/tmp}/pathagon-parity-py.XXXXXX.json")"
rust_output="$(mktemp "${TMPDIR:-/tmp}/pathagon-parity-rs.XXXXXX.json")"
trap 'rm -f "$fixture" "$typescript_output" "$python_output" "$rust_output"' EXIT

cd "$project_dir"
python3 scripts/generate-parity-fixture.py > "$fixture"
"$node_bin" --experimental-strip-types tests/parity-runner.ts "$fixture" > "$typescript_output"
"$python_bin" -m learning.gnn.parity_runner "$fixture" > "$python_output"
cargo run --quiet --manifest-path engine-rs/Cargo.toml --bin parity -- "$fixture" > "$rust_output"
"$python_bin" scripts/compare-parity.py "$typescript_output" "$python_output" "$rust_output"
