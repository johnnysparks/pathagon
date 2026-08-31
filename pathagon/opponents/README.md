# Supported opponents

This directory is the registry for opponents that are part of the current
game, rather than experiments. An opponent belongs here only when its rules
version, stable ID, Rust search/runtime configuration, required model hashes,
and lifecycle status are documented and covered by integration tests.

The current built-in opponents are implemented by the Rust engine and exposed
to the browser through WASM. Their user-facing catalog remains in
`apps/web/app/opponents.ts` until another app needs the same presentation
metadata. Experimental checkpoints and Python agents stay with their dated
research path; promotion means porting behavior to Rust and recording only
high-value deployable artifacts here.

## `pathfinder-v0.5.0-trained-evaluator`

The Pathfinder · Trained is a Rust tactical-filter search opponent using the
same 4-ply, 2,000-node, beam-8 search envelope as the research control, with
evaluator weights evolved against the tactical-filter baseline. Its durable
configuration and evidence are recorded in
[`pathfinder-v0.5.0-trained-evaluator.json`](pathfinder-v0.5.0-trained-evaluator.json).
The 120-game held-out screen scored 70–47–3 against
`pathfinder-v0.4.0-tactical-filter` with paired colors and two randomized
opening plies. Browser/WASM integration, native identity smoke tests, focused
web tests, and replay review are complete. The opponent is promoted with a
provisional rating pending a longer post-deployment ladder. A reproduction
after the alpha-beta sentinel-bound fix scored 70–48–2 with 5,886,532 nodes;
the original 70–47–3 evidence remains recorded in the opponent manifest, with
the difference explained in the budgeted Pathfinder research record.

## `pathfinder-action-transition-v4-xent`

The Pathfinder · Transition v4 is the current user-facing default. It uses the
Rust/WASM tactical-safe search wrapper with the versioned explicit
placement/relocation transition model in
[`data/models/pathfinder-action-transition-v4-xent/`](../../data/models/pathfinder-action-transition-v4-xent/).
The 1,000-game paired arena scored 565–401–34 against the v0.5 trained
evaluator (58.2% points), with positive point rates in both colors and a
complete native replay audit. Its stable identity, search envelope, model hash,
and provenance are recorded in
[`pathfinder-action-transition-v4-xent.json`](pathfinder-action-transition-v4-xent.json).
