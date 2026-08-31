# Pathfinder action-transition v3 xent

This is the packaged research candidate from
`research/20260829-nextgen-action-transition`. It is a 32-feature, 24-24-1
explicit placement/relocation scorer. The Rust/WASM wrapper accepts the JSON
bytes, exposes the model identity, ranks only rules-generated tactical-safe
roots, and delegates the final move to Pathfinder search.

- Artifact: `transition-policy.json`
- SHA-256: `4f08a5a68057051e99c469aaf4a6e839885ebdcb167e6b82b076836c0b24b7f4`
- Rules namespace: `pathagon-rules-v1`
- Status: prior packaged model; v4 is the current default opponent

The model is opt-in through `loadTransitionPolicyEngine()` in the web Rust
adapter. The heldout depth-8 disagreement labels that informed the next
research step are tracked separately in the games-v1 corpus sidecar.
