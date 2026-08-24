# Pathagon interchange contract

`pathagon-contract-v1.schema.json` is the canonical wire contract for the
three runtimes. It defines the rule configuration, actions, complete
rule-relevant positions, replay moves, termination reasons, engine metadata,
and agent specifications. Each agent specification carries a manifest with
runtime, rules version, evaluator weights, search depth, node budget, beam,
and an optional model hash. The small replay fixture is consumed by the
TypeScript, Python, and Rust contract tests.

New records use `contractVersion: 1`. Older schema-v2 records remain readable
through the TypeScript archive normalizer; their missing configuration and
agent metadata are filled with explicit legacy defaults. New records should
not add `boardSize`, `reservePerPlayer`, or a second engine string at the
top level—those values belong under `config` and `engine`.

The contract intentionally uses portable square arrays in `Position` rather
than Rust bitboards or Python masks. Runtime-specific search diagnostics are
optional move fields, while the board, reserves, turn, forbidden squares, and
relocation markers are mandatory because they affect replay and repetition.
