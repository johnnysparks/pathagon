#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="$ROOT_DIR/.venv-pathagon-gnn/bin/python"
ARCHIVE="$ROOT_DIR/research/20260901-strong-teacher-10k-games/workspace/generation-20/teacher-games-10000.jsonl"
WORKSPACE="$ROOT_DIR/research/20260901-strong-teacher-10k-games/workspace"
AUDIT="$WORKSPACE/generation-20/audit.json"
STAGING="$ROOT_DIR/data/corpora/teacher-d5-b256-500k-v1.staging"
CANONICAL="$ROOT_DIR/data/corpora/teacher-d5-b256-500k-v1"
MODELS="$WORKSPACE/retraining"

[[ -f "$ARCHIVE" ]] || { echo "missing completed archive: $ARCHIVE" >&2; exit 2; }
"$PYTHON" "$ROOT_DIR/research/20260901-strong-teacher-10k-games/scripts/validate_teacher_games.py" \
    --input "$ARCHIVE" --output "$AUDIT" --expected-games 10000 --seed 2026090100 \
    --max-plies 20 --opening-random-plies 4

[[ -d "$CANONICAL" ]] || { echo "missing versioned corpus directory: $CANONICAL" >&2; exit 2; }
[[ ! -e "$STAGING" ]] || { echo "staging directory already exists; inspect it before retrying: $STAGING" >&2; exit 2; }
python3 "$ROOT_DIR/scripts/compact_game_corpus.py" \
    --input "$ARCHIVE" --output "$STAGING" --no-base --progress-every 1
for path in "$STAGING"/*; do
    target="$CANONICAL/$(basename "$path")"
    [[ ! -e "$target" ]] || { echo "refusing to overwrite canonical artifact: $target" >&2; exit 2; }
    mv "$path" "$CANONICAL/"
done
rmdir "$STAGING"

"$PYTHON" "$ROOT_DIR/research/20260901-strong-teacher-10k-games/scripts/retrain_replay_architectures.py" \
    --input "$ARCHIVE" --output-dir "$MODELS" --split-dir "$MODELS/split" \
    --heldout-fraction 0.2 --split-seed 2026090101 --seed 2026090102 \
    --steps 20000 --learning-rate 3e-4 --max-eval-examples 10000 --device auto
