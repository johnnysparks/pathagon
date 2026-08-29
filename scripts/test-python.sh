#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PATHAGON_PYTHON:-${project_dir}/.venv-pathagon-gnn/bin/python}"

if [[ ! -x "$python_bin" ]]; then
  echo "Python test environment is missing: $python_bin" >&2
  echo "Create it with: python3 -m venv .venv-pathagon-gnn && .venv-pathagon-gnn/bin/python -m pip install -r research/20260824-gnn-cnn-lab/python/requirements.txt" >&2
  exit 2
fi

cd "$project_dir"
exec "$python_bin" -m unittest discover -s research/20260824-gnn-cnn-lab/python -t research/20260824-gnn-cnn-lab -p 'test_*.py'
