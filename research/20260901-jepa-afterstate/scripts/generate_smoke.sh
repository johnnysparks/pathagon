#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKSPACE="$ROOT_DIR/research/20260901-jepa-afterstate/workspace"
mkdir -p "$WORKSPACE"

cargo run --manifest-path "$ROOT_DIR/pathagon/engine-rs/Cargo.toml" \
  --release --bin pathagon-jepa-export -- \
  --out "$WORKSPACE/rust-transitions.jsonl" \
  --games "${GAMES:-64}" \
  --max-plies "${MAX_PLIES:-40}" \
  --actions-per-state "${ACTIONS_PER_STATE:-32}" \
  --seed "${SEED:-2026090101}" \
  > "$WORKSPACE/export-report.json"

echo "wrote $WORKSPACE/rust-transitions.jsonl"
