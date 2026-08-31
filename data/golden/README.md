# Persistent golden position tables

This directory contains durable, architecture-independent Pathagon truth data.
It is deliberately separate from `work/`: local runs are disposable
experiment outputs; these tables accumulate exact results and remain useful to
every future model, solver, and engine.

## Table semantics

The first table family is `historyless-wdl-v1`. Its value means:

> exact win/draw/loss result from this position, assuming a fresh position
> with no prior repetition history and no finite move horizon.

The live engine may still override a lookup when its own threefold or max-ply
rules require a draw. Unknown positions are absent. Only exact values belong in
the golden table; model estimates, search bounds, visit counts, and provenance
live in separate sidecars.

## Binary format

The canonical 7×7/14-reserve position key is 14 bytes:

```text
49 cells × 2 bits: empty, Light, Dark, forbidden  = 98 bits
turn                                                 = 1 bit
last-relocated Light (0..48, 49 = none)             = 6 bits
last-relocated Dark  (0..48, 49 = none)             = 6 bits
                                                         111 bits
```

Rows are fixed-width and sorted by key:

```text
[key: 14 bytes][value: 1 byte]
```

Values are side-to-move-relative: `0 = loss`, `1 = draw`, `2 = win`.
The unused high bit of the key is reserved and must remain zero. The key is
canonicalized over all eight rules-preserving D4 symmetries before storage and
lookup; forbidden squares and relocation markers are transformed along with
the pieces.

`GoldenTable` writes immutable-style sorted shards atomically. `FlatGoldenTable`
performs an O(log n) lookup directly against a shard without loading the table
into memory. New exact observations must be monotonic: a duplicate value is
idempotent, while a contradictory value is an error.

Ring frontiers may also carry a compact action sidecar. The promoted
`fresh-frontier-wdl-v1` Ring-1 sidecar uses the `PGACT02` format: a 16-byte
header followed by sorted rows containing `[canonical key: 14 bytes][flags:
u8][root W/D/L: u8][root distance: u16][known action count: u16]`, then sparse
`[action code: u16][action W/D/L/unknown: u8][distance: u16]` records. Action
codes use the corpus base-64 action numbering. Unknown actions are omitted on
incomplete rows, so the one-byte completeness flag preserves the distinction
without repeating an unknown label for every legal move. Rust continues to
read the older `PGACT01` action-only format for rollback compatibility.

The codec is size-aware for the historical 5×5 curriculum boards as well. A
5×5 key is 8 bytes (the relocation markers need 5 bits), and each board-size /
reserve pair has its own shard namespace. Reserve is intentionally not packed
into the key, so a 5×5/r8 key must never be looked up in the 5×5/r10 table.

## Directory contract

`manifest.json` describes the rules namespace, key/value encoding, shard
hashes, counts, and source lineage. Keep manifests and small shards in Git.
Move large shards to Git LFS or content-addressed object storage while retaining
their stable URI, SHA-256, byte size, and retention policy in the manifest.

The current shards are seeded from every replay-bearing archive discoverable
under `research/`, plus every canonical corpus game shard. Those rows are
unconditionally exact terminal truths. Repetition/max-ply draws are retained
in the source inventory but intentionally excluded because this table ignores
play history; solving non-terminal parents requires legal transitions and
minimax/backward propagation.

Native target generation can consult a promoted action book with:

```bash
cargo run --release --manifest-path pathagon/engine-rs/Cargo.toml \
  --bin pathfinder_targets -- \
  --golden-table data/golden/tables/fresh-frontier-wdl-v1/7x7-r14/shard-00.bin \
  --golden-sidecar data/golden/sidecars/fresh-frontier-wdl-v1/7x7-r14/ring-01.bin
```

Known Ring-1 actions become explicit policy/value/distance metadata; positions
without a proven action continue through ordinary search.

The first promoted Ring-2 proof is kept as a separate one-row control shard:
`fresh-frontier-wdl-v2/7x7-r14`. Its root is an exact side-to-move loss at
distance 2 with a complete 21-action label set. Ring-1 remains the default
control table; the Ring-2 shard is not automatically overlaid because the
current flat lookup accepts one table and one sidecar. Use the v2 table and
sidecar explicitly when reproducing this experiment, and keep the manifest
with them as the source/proof boundary.

```bash
cargo run --release --manifest-path pathagon/engine-rs/Cargo.toml \
  --bin pathfinder_targets -- \
  --golden-table data/golden/tables/fresh-frontier-wdl-v2/7x7-r14/shard-00.bin \
  --golden-sidecar data/golden/sidecars/fresh-frontier-wdl-v2/7x7-r14/ring-02.bin
```

The Rust-native follow-up is `fresh-frontier-wdl-v3/7x7-r14`: two independently
solved Ring-2 roots, stored as a 30-byte WDL shard and a 266-byte `PGACT02`
sidecar. The promotion executable is
`pathagon-endgame-promote`; it is the authoritative writer and gate for new
Ring-2 rows. The earlier Python verifier remains useful as an independent
cross-check, but is not required by the Rust promotion path.

The current expanded control is `fresh-frontier-wdl-v4/7x7-r14`: three
independently solved Ring-2 roots, stored as a 45-byte WDL shard and a
391-byte `PGACT02` sidecar. It passed the same Rust gates with 24 symmetry
checks and zero contradictions. This is still a standalone Ring-2 experiment;
the Ring-1 table remains the rollback/control artifact. Native Rust target
generation can now overlay ordered layers without rewriting either artifact.
Rust promotion canonicalizes both keys and action coordinates before writing a
sidecar, and the exporter rejects any action that is not legal in the decoded
canonical representative:

```bash
cargo run --release --manifest-path pathagon/engine-rs/Cargo.toml \
  --bin pathfinder_targets -- \
  --golden-layers "data/golden/tables/fresh-frontier-wdl-v1/7x7-r14/shard-00.bin,data/golden/sidecars/fresh-frontier-wdl-v1/7x7-r14/ring-01.bin;data/golden/tables/fresh-frontier-wdl-v4/7x7-r14/shard-00.bin,data/golden/sidecars/fresh-frontier-wdl-v4/7x7-r14/ring-02.bin"
```

Layers are listed from highest to lowest priority; an absent key falls through
to the next layer. The browser WASM boundary still needs a bundle-fetch
adapter before this native layering is used in the live UI.

To rebuild that seed table:

```bash
python3 scripts/build-golden-terminal-table.py
```

The rebuild also writes `source-inventory.json`, recording each input path,
content hash, replay count, and promotion/exclusion count, along with the
number of archive and canonical-shard files scanned. Fixture/audit positions
under `data/fixtures/` are intentionally not promoted: their finite-horizon
labels are not unbounded historyless game outcomes.
