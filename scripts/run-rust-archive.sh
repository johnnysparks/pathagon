#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="${1:-rust-lunatic-$(date +%Y%m%d-%H%M%S)}"
GAMES="${2:-100}"
SEED="${3:-20260823}"
OPPONENT="${4:-lunatic}"
JSONL_PATH="training/gnn/league/${LABEL}.jsonl"
CORPUS_PATH="training/rust-v1/${LABEL}"

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo is required; install Rust with https://rustup.rs/" >&2
  exit 1
fi

mkdir -p "${ROOT_DIR}/training/gnn/league" "${ROOT_DIR}/training/rust-v1"
(
  cd "${ROOT_DIR}"
  cargo run --release --manifest-path engine-rs/Cargo.toml --bin pathagon-selfplay -- \
    --games "${GAMES}" \
    --opponent "${OPPONENT}" \
    --jsonl \
    --seed "${SEED}" \
    --max-plies 196 \
    --opening-random-plies 2 \
    --corpus "${CORPUS_PATH}" \
    > "${JSONL_PATH}"
)

echo "archive=${ROOT_DIR}/${JSONL_PATH}"
echo "corpus=${ROOT_DIR}/${CORPUS_PATH}"
echo "next: python3 scripts/merge-replay-archives.py --output training/gnn/merged-${LABEL}.jsonl ${JSONL_PATH}"
