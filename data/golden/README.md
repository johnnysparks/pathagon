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
`fresh-frontier-wdl-v1` Ring-1 sidecar uses the `PGACT01` format: a 16-byte
header followed by sorted `[canonical key: 14 bytes][action count: u16][action
code: u16]*` rows. Action codes use the corpus base-64 action numbering. The
sidecar is partial by design; it records verified winning actions, not a claim
that all other legal actions lose.

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

To rebuild that seed table:

```bash
python3 scripts/build-golden-terminal-table.py
```

The rebuild also writes `source-inventory.json`, recording each input path,
content hash, replay count, and promotion/exclusion count, along with the
number of archive and canonical-shard files scanned. Fixture/audit positions
under `data/fixtures/` are intentionally not promoted: their finite-horizon
labels are not unbounded historyless game outcomes.
