# 20260902 Unified opponent artifacts

Status: completed

## Idea

Finish the six-card opponent roster without presenting research placeholders
as models. Promote real GNN policy/value, GNN Q/Advantage, and JEPA
afterstate rank/value artifacts behind the shared Rust/WASM runtime contract.

## Starting point

Pathman and Seer already had browser runtimes. Tile Driver, Double Dragon, and
Yann Tileson had visible cards but were correctly fail-closed because the
repository contained no promoted browser artifacts. The prior QAdv research
ladder was explicitly unstable, and the JEPA smoke checkpoint was
embedding-only.

## What happened

The strong-teacher GNN checkpoint was exported and loaded by the native Rust
inference path. The root-Q-aware QAdv checkpoint from the seeded-position lane
was exported with its 24 transition features and complete root-Q provenance.
The QAdv model remains a provisional control because the earlier paired arena
did not clear its strength gate.

The JEPA path was extended with independent action-ranking and bounded
afterstate-value heads, trained on 22,905 Rust-emitted, mirror-audited
transitions split by game. The exported model was checked against the Python
model, loaded by native Rust, and bound into the browser WASM module. The old
embedding-only smoke artifact was not promoted.

## Data and artifacts

Disposable transition corpora, checkpoints, reports, and ONNX exports remain
under this path's ignored `workspace/`. The three browser artifacts and their
provenance manifests are promoted under `data/models/` and copied to the
browser's public model directory. SHA-256 identities are recorded in each
manifest.

## Project impact

All three learned cards now have real, model-owned runtime implementations.
Their controls are wired through GNN PUCT, Q-seeded PUCT, or JEPA bounded
beam search respectively. Their status is browser-playable provisional, not a
claim that each has cleared the full paired strength ladder.

## Hiccups

The first JEPA experiment only exported the online GNN trunk, which would have
made an embedding-only checkpoint look like a player if the registry had
treated it as playable. The action head and loader gate were added before
promotion. The QAdv research evidence also showed why artifact availability
and competitive strength must remain separate statuses.

## Next decision

Keep all six cards selectable and keep Pathman as the default. Run the longer
paired arenas, replay audits, and browser latency checks for the three newly
playable cards before changing any provisional strength labels or the default.
