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

## Learning experiments

Freeze the deployment search envelope and control opponent before collecting
data. Split by complete game or source family, check that the teacher is
stronger at the same envelope, and record the protocol before training or
arena selection. Keep action ranking and state-value calibration as separate
measurements. Use the learned model as an advisory tie-breaker or single-move
promotion inside native search, and record completed depth, nodes, latency,
legality, and representative games. Do not promote a candidate because it
matches a mismatched teacher, improves one offline metric, or wins at only one
search budget.
