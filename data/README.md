# Durable data

This directory contains compact data that is part of the supported project.
Unlike loose research outputs, these files have stable semantics, consistent
formats, deterministic generation where practical, and automated validation.

| Path | Contents |
| --- | --- |
| [`corpora/games-v1/`](corpora/games-v1/) | Content-addressed move histories, outcomes, observations, and reusable sidecars. |
| [`fixtures/`](fixtures/) | Cross-runtime, tactical, and regression fixtures. |
| [`golden/`](golden/) | Exact position values and their manifests. |

Do not use `data/` as a landing zone for a new experiment. Generate locally in
the dated research path, then deliberately promote only reusable, reviewable
data. Files over 5 MiB fail the durable-data check and should be regenerated or
kept outside Git unless the project explicitly revises that policy.
