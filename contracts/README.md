# Pathagon interchange contract

[`pathagon-contract-v1.schema.json`](pathagon-contract-v1.schema.json) is the
canonical wire contract for TypeScript, Python, and Rust. It defines rule
configuration, actions, complete rule-relevant positions, replay moves,
termination reasons, engine metadata, and agent specifications.

Each agent specification carries a manifest with runtime, rules version,
evaluator weights, search depth, node budget, beam width, board configuration,
and an optional model hash. The compact replay fixture is consumed by all
three contract test suites.

New records use `contractVersion: 1`. Older schema-v2 records remain readable
through the TypeScript archive normalizer, which fills missing configuration and
agent metadata with explicit legacy defaults.

The contract intentionally uses portable square arrays in `Position` rather
than Rust bitboards or Python masks. Runtime-specific search diagnostics are
optional move fields; board, reserves, turn, forbidden squares, and relocation
markers are mandatory because they affect replay and repetition.

Contract changes require the cross-runtime parity suite. See
[`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for runtime ownership and
[`docs/DATA.md`](../docs/DATA.md) for archive policy.
