#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

forbidden="$(git ls-files -- 'research/**/workspace/**' training apps/web/public/lab || true)"
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
done < <(git ls-files -z -- data pathagon/contracts apps/web/public/models)

echo "Durable data policy passed."
