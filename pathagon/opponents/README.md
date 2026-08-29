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
