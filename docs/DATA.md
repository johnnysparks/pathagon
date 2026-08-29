# Data and artifact policy

`data/` is for durable project inputs: canonical games, exact labels, strict
fixtures, and deployable model metadata. Files must have stable meaning,
consistent fields, provenance, deterministic ordering where practical, and an
active consumer or test.

Loose research outputs belong in the originating dated path's ignored
`workspace/`. A research README should describe what was generated and what
was discarded, but one-time games, targets, checkpoints, and logs do not need
to be preserved. Imperfect reproducibility is acceptable for retired research.

Promotion into `data/` is deliberate. Prefer canonical move history plus
reconstructable facts over repeated materialized board states. Do not duplicate
games by agent pair. Version reusable sidecars, define perspective and action
ordering, and keep evaluation eligibility explicit.

The automated durable-data check rejects tracked files over 5 MiB in `data/`,
`pathagon/contracts`, and deployed model assets. Exceptions should be rare,
reviewed, and documented with why Git is the right store.
