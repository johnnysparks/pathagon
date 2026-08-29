# Contributing

Start by deciding which lifecycle the change belongs to.

- New product behavior belongs in the owning app.
- Stable game rules, search, interchange, or opponents belong in `pathagon/`.
- Reusable truth-scored games and fixtures belong in `data/`.
- An uncertain idea starts as `research/YYYYMMDD-short-question/README.md` with
  generated work in its ignored `workspace/`.

Supported Rust and data changes require focused tests, consistent formats, and
review of representative game output. Research code does not need production
coverage. Promotion is a rewrite/port with a clear contract, not a directory
move that silently blesses experimental code.

Before submitting a supported change, run the relevant narrow checks and then
`npm test`. Keep commits coherent and do not add generated archives, model
training state, or large one-time outputs merely for reproducibility theater.
