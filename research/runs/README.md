# Research run workspace

Everything below this directory is ignored except this file. Use it for active,
reproducible run output that can be deleted and regenerated.

Durable artifacts have separate homes:

- normalized games and training metadata: [`research/corpora/games-v1/`](../corpora/games-v1/)
- experiment decisions, outcomes, hashes, and selected small models: [`research/experiments/`](../experiments/)
- curated universal fixtures: [`research/fixtures/`](../fixtures/)

Before declaring a serious run complete, promote its move histories into the
canonical corpus and create or update its experiment record. Keep bulky replay,
logs, checkpoints, and intermediate products in this ignored workspace or an
external artifact store; do not add exceptions to `.gitignore` for individual
runs. See [`docs/EXPERIMENTS.md`](../../docs/EXPERIMENTS.md).
