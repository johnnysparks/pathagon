#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="$ROOT_DIR/.venv-pathagon-gnn/bin/python"
SCRIPTS="$ROOT_DIR/research/20260901-strong-teacher-10k-games/scripts"
WORKSPACE="$ROOT_DIR/research/20260901-strong-teacher-10k-games/workspace"
STRONG_ARCHIVE_8K="$WORKSPACE/generation-20/teacher-games-8000.jsonl"
STRONG_TOPUP_ARCHIVE="$WORKSPACE/generation-strong-topup-100/teacher-games-100.jsonl"
WEAK_ARCHIVE="$WORKSPACE/generation-weak-2100-disjoint/teacher-games-2100.jsonl"
MIXED_DIR="$WORKSPACE/mixed-10k"
MIXED_ARCHIVE="$MIXED_DIR/teacher-games-mixed-10000.jsonl"
MIXED_MANIFEST="$MIXED_DIR/mixed-manifest.json"
EXPECTED_SEEDS="$MIXED_DIR/expected-seeds.json"
STRONG_AUDIT="$WORKSPACE/generation-20/audit-8000.json"
STRONG_TOPUP_AUDIT="$WORKSPACE/generation-strong-topup-100/audit.json"
WEAK_AUDIT="$WORKSPACE/generation-weak-2100-disjoint/audit.json"
MIXED_AUDIT="$MIXED_DIR/audit.json"
STAGING="$ROOT_DIR/data/corpora/teacher-d5-b256-500k-v1.staging"
CANONICAL="$ROOT_DIR/data/corpora/teacher-d5-b256-500k-v1"
MODELS="$WORKSPACE/retraining-mixed-final"

[[ -f "$STRONG_ARCHIVE_8K" ]] || { echo "missing strong archive: $STRONG_ARCHIVE_8K" >&2; exit 2; }
[[ -f "$STRONG_TOPUP_ARCHIVE" ]] || { echo "missing strong top-up archive: $STRONG_TOPUP_ARCHIVE" >&2; exit 2; }
[[ -f "$WEAK_ARCHIVE" ]] || { echo "missing weak archive: $WEAK_ARCHIVE" >&2; exit 2; }
mkdir -p "$MIXED_DIR"

"$PYTHON" "$SCRIPTS/validate_teacher_games.py" \
    --input "$STRONG_ARCHIVE_8K" --output "$STRONG_AUDIT" \
    --expected-games 8000 --seed 2026090100 --max-plies 20 \
    --opening-random-plies 4 --opponent-profile 5:256:500000 --allow-duplicate-games
"$PYTHON" "$SCRIPTS/validate_teacher_games.py" \
    --input "$STRONG_TOPUP_ARCHIVE" --output "$STRONG_TOPUP_AUDIT" \
    --expected-games 100 --seed 2026101000 --max-plies 20 \
    --opening-random-plies 4 --opponent-profile 5:256:500000
"$PYTHON" "$SCRIPTS/validate_teacher_games.py" \
    --input "$WEAK_ARCHIVE" --output "$WEAK_AUDIT" \
    --expected-games 2100 --seed 2026110100 --max-plies 20 \
    --opening-random-plies 4 --opponent-profile 3:64:12000 --allow-duplicate-games

"$PYTHON" "$SCRIPTS/assemble_mixed_corpus.py" \
    --strong-input "$STRONG_ARCHIVE_8K" --strong-input "$STRONG_TOPUP_ARCHIVE" \
    --weak-input "$WEAK_ARCHIVE" \
    --output "$MIXED_ARCHIVE" --manifest "$MIXED_MANIFEST" \
    --expected-seeds "$EXPECTED_SEEDS" --strong-games 8000 --weak-games 2000
"$PYTHON" "$SCRIPTS/validate_teacher_games.py" \
    --input "$MIXED_ARCHIVE" --output "$MIXED_AUDIT" \
    --expected-games 10000 --expected-seeds "$EXPECTED_SEEDS" \
    --max-plies 20 --opening-random-plies 4 \
    --opponent-profile 5:256:500000 --opponent-profile 3:64:12000 \
    --require-opponent-profile 5:256:500000 \
    --require-opponent-profile 3:64:12000

[[ -d "$CANONICAL" ]] || { echo "missing versioned corpus directory: $CANONICAL" >&2; exit 2; }
[[ ! -e "$STAGING" ]] || { echo "staging directory already exists; inspect it before retrying: $STAGING" >&2; exit 2; }
python3 "$ROOT_DIR/scripts/compact_game_corpus.py" \
    --input "$MIXED_ARCHIVE" --output "$STAGING" --no-base --progress-every 1
for path in "$STAGING"/*; do
    [[ "$(basename "$path")" == "README.md" ]] && { mv "$path" "$MIXED_DIR/compacted-corpus-readme.md"; continue; }
    target="$CANONICAL/$(basename "$path")"
    [[ ! -e "$target" ]] || { echo "refusing to overwrite canonical artifact: $target" >&2; exit 2; }
    mv "$path" "$CANONICAL/"
done
rmdir "$STAGING"

"$PYTHON" "$SCRIPTS/retrain_replay_architectures.py" \
    --input "$MIXED_ARCHIVE" --output-dir "$MODELS" --split-dir "$MODELS/split" \
    --heldout-fraction 0.2 --split-seed 2026090101 --seed 2026090102 \
    --steps 5000 --learning-rate 3e-4 --max-eval-examples 2000 --device auto

echo "mixed corpus and retraining complete: $MIXED_ARCHIVE"
