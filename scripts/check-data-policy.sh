#!/usr/bin/env bash
set -euo pipefail

forbidden="$({ git ls-files -- research/runs training research/adversarial/generated public/lab || true; } | sed '/^research\/runs\/README\.md$/d')"
if [[ -n "${forbidden}" ]]; then
  echo "Durable data policy violation: generated workspace files are tracked:" >&2
  echo "${forbidden}" >&2
  exit 1
fi

while IFS= read -r -d '' artifact; do
  bytes="$(wc -c < "${artifact}" | tr -d ' ')"
  if (( bytes > 5242880 )); then
    echo "Durable data policy violation: ${artifact} is larger than 5 MiB (${bytes} bytes)." >&2
    echo "Keep it in external storage and commit a hash/manifest instead." >&2
    exit 1
  fi
done < <(git ls-files -z -- research/corpora research/experiments research/fixtures)

echo "Durable data policy passed."
