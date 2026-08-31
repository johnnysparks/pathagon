# 20260830 Endgame retrograde frontier

Status: Ring-1 pilot complete; Ring-2 generation, compact persistence, corrected
propagation, bounded focused expansion, two-root promotion, and Rust-native
promotion gates validated

## Idea

Build a monotonic, exact endgame dataset by starting with legal terminal
positions and expanding one verified move backward at a time. Promote only
proof-backed W/D/L values; keep node-limited, time-limited, and incomplete
action labels as unknown.

## Starting point

The project has a historyless terminal golden table with 41,794 canonical
7x7/14-reserve rows and a bounded tactical proof solver. The tactical solver
is useful for move selection, but its horizon and budget cutoffs cannot be
promoted as draws. The first pilot uses the canonical replay corpus as the
reachability witness source.

## What happened

The strict Rust oracle now distinguishes loss, draw, win, and unknown, tracks
optional distance-to-terminal, and reports whether every legal action has a
known result. A separate `pathagon-endgame-frontier` executable extracts Ring
1: every unique parent immediately before a replay-proven path terminal,
including all witnessed finishing actions and source game keys.

The promotion script replays and validates every candidate, canonicalizes it
over the eight rules-preserving D4 symmetries, writes a versioned binary WDL
shard, and writes a compact action/provenance sidecar. Ring 1 intentionally
marks complete optimal-action sets as unknown because a witnessed winning move
does not prove that every other legal move loses.

The complete replay pilot scanned 38,547 corpus games, admitted 35,595 raw
penultimate positions, and produced 35,561 canonical rows after 34 D4 merges.
It carried 35,595 proven actions and zero complete-action-set claims. A
constructive-mode smoke run over 20 games verified 307 additional candidate
transitions; those are retained as research evidence only because their fresh
root metadata is not a historical reachability proof. The first promotion
attempt also caught a relative-path manifest bug and the runtime smoke test
caught a Rust marker-width bug; both were fixed before promotion. A symmetry
inversion test then caught and corrected an action-canonicalization bug.
The promoted rows had zero overlaps with contradictory or pre-existing gold;
the independent verifier passed sorted-key, header, action-bound, size, hash,
and count checks. The durable Ring-1 sidecar is now `PGACT02`: it stores the
root distance, sparse per-action W/D/L/distance labels, and the complete-action
set bit while leaving unlisted actions implicit-unknown on partial rows. The
Rust reader remains backward-compatible with `PGACT01`.

The full Ring-2 export scanned the same 38,547 games and emitted 35,562
replay-witnessed parent records with 4,485,656 complete forward edges and
4,521,100 graph records including explicit unknown child stubs. Joining the
promoted Ring-1 table as exact seeds solved 35,561 inner rows, but no Ring-2
parent was closed: every parent still had at least one child outside the exact
seed set. The promotion audit therefore retained all Ring-2 parents as
unknown and promoted zero rows. This is an expected proof boundary, not a
training failure.

After correcting retrograde early-win propagation and adding bounded Rust
expansion, a focused low-branching experiment selected one 21-child Ring-2
root. The three-pass slice expanded 103,009 reachable stubs and produced a
1,312,242-node compact graph. Its exact solve proved 182 rows, including the
selected root as a side-to-move loss at distance 2; the remaining rows stayed
unknown because the slice is not closed. The promotion verifier replayed the
root against the original full graph and passed every gate: inventory/seeded,
forward transition witness, symmetry, complete action sets, and zero
contradictory existing gold. That singleton was the first non-empty durable
Ring-2 promotion; a third independent root has now passed the same gates and
the three rows are consolidated in the Rust-native v4 artifact.

## Data and artifacts

Disposable extractor output, checkpoints, and summaries belong in this path's
ignored `workspace/`. The promoted
`fresh-frontier-wdl-v1` artifacts are [the 533,415-byte WDL shard](../../data/golden/tables/fresh-frontier-wdl-v1/7x7-r14/shard-00.bin), [the 889,046-byte action metadata sidecar](../../data/golden/sidecars/fresh-frontier-wdl-v1/7x7-r14/ring-01.bin), and [their manifest](../../data/golden/fresh-frontier-wdl-v1-manifest.json). The compact sidecar is a sorted binary key-to-sparse-action-metadata index so durable data stays below the repository's 5 MiB file limit. Do not place solver traces, queues, or implementation-shaped tensors in durable data.

The first promoted Ring-2 control is [a 15-byte WDL shard](../../data/golden/tables/fresh-frontier-wdl-v2/7x7-r14/shard-00.bin), [a 141-byte action sidecar](../../data/golden/sidecars/fresh-frontier-wdl-v2/7x7-r14/ring-02.bin), and [its manifest](../../data/golden/fresh-frontier-wdl-v2-manifest.json). It is intentionally a standalone proof shard rather than a replacement for Ring-1; the current runtime lookup takes one table/sidecar pair. The manifest records the full-graph source and the `closed-ring-2-only` promotion decision.

The Rust-native follow-up is [a 30-byte two-root WDL shard](../../data/golden/tables/fresh-frontier-wdl-v3/7x7-r14/shard-00.bin), [a 266-byte action sidecar](../../data/golden/sidecars/fresh-frontier-wdl-v3/7x7-r14/ring-02.bin), and [its manifest](../../data/golden/fresh-frontier-wdl-v3-manifest.json). It was emitted by `pathagon-endgame-promote` after 35,562 Ring-2 records were scanned; both rows closed, all 16 D4 checks passed, and no contradictory gold was found.

The third-root pass is [a 45-byte three-root WDL shard](../../data/golden/tables/fresh-frontier-wdl-v4/7x7-r14/shard-00.bin), [a 391-byte action sidecar](../../data/golden/sidecars/fresh-frontier-wdl-v4/7x7-r14/ring-02.bin), and [its manifest](../../data/golden/fresh-frontier-wdl-v4-manifest.json). The third root's slice contained 1,290,536 nodes and 1,385,832 edges; exact propagation found 203 values, and the Rust promotion gate admitted the third closed parent alongside the two v3 roots. The unresolved remainder stayed unknown in ignored workspace artifacts.

The tablebase executable uses compact binary values and action labels for
large research outputs: a fixed-width key plus one outcome byte and one `u16`
distance, with absent keys representing unknown. Human-readable JSON remains
available with `--format json`; compact runs write a small metadata JSON beside
the value and action binaries. The full Ring-2 exact values are about 590 KB
in compact form versus about 435 MB in the earlier nested JSON.

The workspace density audit found three additional, low-risk wins:

1. The 4,521,100-node legal-edge graph now has a Rust `PGGRF01` binary form.
   The full graph measured 1.6 GB as JSONL and 202 MB as binary (about 87%
   smaller); a 42,201-node smoke graph measured 18 MB and 1.9 MB. The
   `pathagon-endgame-compact` converter creates the binary plus a small
   metadata manifest, and `pathagon-endgame-tablebase` reads either format.
   Passing `--format jsonl` to the same converter expands the binary into
   deterministic graph-only JSONL evidence. Child keys are fixed-width bytes,
   and two-character corpus actions are a packed code plus a local child index,
   so no proof-bearing graph information is discarded.
2. Checkpoints now use a small JSON manifest plus a compact value file instead
   of duplicating every solved value in pretty JSON. `read_checkpoint` remains
   compatible with the older inline-value checkpoint format, so restarts do
   not require a migration step.
3. Cold evidence compresses well: zstd level 1 reduced the full JSONL export
   to 63 MB and the 202 MB compact graph to 48 MB in this workspace. Keep zstd
   files for archival or transfer copies; the uncompressed binary graph remains
   the hot solver artifact so the solver does not pay decompression and
   reparsing costs.

The layered training interchange is intentionally disposable: the Rust
feature-bearing JSONL is 746 MiB, while its zstd level-1 archive is 56.5 MiB
(92.4% smaller). Keep the JSONL only when training or inspecting it; the
versioned compact table/sidecar pair remains the durable artifact.

The workspace still contains intentionally duplicated historical passes and
the original JSONL export. The safe management rule is to retain one
canonical artifact per pass, a manifest with counts/hashes/command, and only
failed or promotion-relevant evidence; do not delete existing exports
automatically while the frontier protocol is still being validated.

## Project impact

This path establishes the first replay-witnessed frontier layer for the golden
dataset and provides a strict contract for later retrograde rings. It does not
yet claim a complete 7x7 tablebase, and it does not promote approximate search
results. The Rust engine can now load the shard and sidecar, recover a legal
verified action through D4 inversion, short-circuit search on that action, and
emit `golden-ring-1` training targets with explicit provenance.

The separate `pathagon-endgame-tablebase` binary now provides the persistent
retrograde core for later rings. It accepts canonical-key JSONL nodes with
complete-edge declarations, propagates wins/losses with shortest/longest
proven distances, treats closed unresolved regions as draws, keeps incomplete
regions unknown, and writes a deterministic value file plus a restartable
checkpoint. It also emits stable value shards and a merge tool that rejects
misplaced or contradictory rows. Nodes may carry an exact seed from an inner
ring, and labeled edges produce per-action W/D/L/Unknown results in the
solver output. Its replay-ring extractor accepts Ring 2 and any later ring
number, verifies the entire replay suffix, and only admits constructive
candidates as proposal-only records.

Ring 2 is now a separate Rust export mode. `--ring 2` replays the source game
and verifies the final two-edge suffix, then enumerates every legal action from
the predecessor, writes canonical child edges, and creates explicit seeded
Ring-1 child stubs plus explicit unknown child stubs. A 20-game smoke run
produced 19 verified Ring-2 parents, 2,555 complete legal edges, 19 seeded
inner-ring children, and 2,555 unresolved edge targets. The small tablebase
run resumed from its checkpoint byte-for-byte and its four deterministic
shards merged successfully.

The independent Rust/Python oracle check covers one 3x3 and three 4x4 roots,
including root and every action label at horizon three. All four fixtures
agree; the report is retained in `workspace/small-agreement-report.json`.

The full compact Ring-2 run was re-solved with eight workers, merged from 32
deterministic shards byte-for-byte, and checked with one deterministic sample
from every non-empty shard. The compact output carries solver version, rules
version, proof lineage, exact/unknown statistics, per-action values, and an
explicit complete-optimal-action-set map.

The Rust `pathagon-endgame-expand` executable now materializes missing child
stubs in bounded passes. It decodes the canonical key, reconstructs inventory,
enumerates the legal Rust action boundary, emits canonical child edges, appends
deduplicated records for newly discovered children, and terminalizes a
reachable child only when its path state proves a loss or a no-action draw. A
100-stub Ring-2 smoke pass emitted 19,635 legal edges and appended 19,635
unknown child records; a second 100-stub pass emitted another 19,992 edges.
The tablebase re-read the resulting 42,201-record graph and still left all
incomplete branches unknown.

The expander now accepts compact graphs directly and can take a
`--focus-roots <keys.txt>` list. It orders candidates by BFS distance from the
replay-witnessed roots, then by canonical key, so bounded passes spend budget on
direct Ring-2 children before deeper descendants. A 50,000-stub broad pass
added 1,500,613 edges and 5,514 terminal proofs; corrected minimax propagation
raised exact rows from 35,561 to 45,735. The promotion verifier passed its
inventory, transition, and contradiction gates but found zero closed Ring-2
parents, so no approximate rows were promoted. A focused 50,000-direct-child
pass produced no immediate terminals and 4,414,520 descendants, confirming
that breadth-first closure needs a narrower deepening budget before it is
cost-effective.

The new `pathagon-endgame-slice` Rust executable extracts only nodes reachable
from a supplied root list, preserving missing children as unknown. This keeps
deepening experiments bounded and makes a single-root proof auditable without
copying unrelated graph regions. It accepts the same compact graph format and
can expand it to deterministic graph-only JSONL evidence when needed.

The new `pathagon-endgame-promote` Rust executable is now the authoritative
Ring-2 promotion path. It reads compact value shards, replays every legal
edge from the corpus graph, checks canonical D4 symmetry and child minimax,
rejects contradictory existing gold, and writes the compact WDL table,
`PGACT02` sidecar, SHA-256 manifest, and gate report without importing the
Python rules package. Additional solved shard directories can be supplied with
`--extra-shards`, keeping each independently solved slice separately
auditable.

The Rust target writer now accepts the ordered `--golden-layers` form. It
preserves exact W/D/L outcome, terminal distance, action-set completeness,
proven winning actions, and urgency actions (fastest proven wins or longest
delayed losses) in target metadata. Partial action labels remain explicitly
partial; they are emitted as a set rather than being silently converted into a
unique optimum.

```bash
pathagon/engine-rs/target/debug/pathagon-endgame-promote \
  --graph research/20260830-endgame-retrograde-frontier/workspace/ring-02-full.jsonl \
  --shards research/20260830-endgame-retrograde-frontier/workspace/ring-02-one-root-pass-0003.shards \
  --extra-shards research/20260830-endgame-retrograde-frontier/workspace/ring-02-second-root-pass-0003.resolved.shards,research/20260830-endgame-retrograde-frontier/workspace/ring-02-third-root-pass-0003.shards \
  --existing-table data/golden/tables/fresh-frontier-wdl-v1/7x7-r14/shard-00.bin \
  --table data/golden/tables/fresh-frontier-wdl-v4/7x7-r14/shard-00.bin \
  --sidecar data/golden/sidecars/fresh-frontier-wdl-v4/7x7-r14/ring-02.bin \
  --manifest data/golden/fresh-frontier-wdl-v4-manifest.json \
  --table-family fresh-frontier-wdl-v4 --ring 2
```

The retrograde resolver now implements the monotonic early-win rule directly:
one known child loss proves a parent win even when sibling branches remain
unknown. Loss still requires every legal child to be a known win, and draws
still require complete graph/cycle evidence. This distinction is covered by a
regression test and is material to future Ring-2/Ring-3 closure.

The layered training gate remains intentionally research-only. Rust exported
35,564 rows (35,561 Ring-1 wins and three exact Ring-2 losses), including Rust
action features and precomputed forced-block targets. The permanent held-out
split contains 3,597 Ring-1 rows plus one Ring-2 loss; the other two Ring-2
rows are used for training so the value gate is not structurally impossible.
The batched linear candidate reached 48.47% held-out witnessed-action/urgency
accuracy versus 46.86% for the frozen v4 control, matched forced-block safety
at 100% on 185 eligible rows, and reached 99.97% aggregate W/D/L accuracy.
The candidate selected an urgency-valid move on the held-out Ring-2 row, but
its value head still missed that forced-loss class (0/1). The report therefore
fails the explicit Ring-2 value gate and keeps the candidate research-only;
the exact Rust table remains the authoritative answer for promoted positions.

The Rust `pathagon-endgame-match` executable now provides the complementary
held-out match gate. It deliberately searches without consulting gold, applies
the selected action through the Rust rules boundary, and records exact-action
matches, terminal wins, exhaustion, depth, and node cost. A full baseline run
over the 3,597-row Ring-1 partition was started but intentionally stopped when
the Ring-2 expansion effort was paused; no partial match result is treated as
evidence. A bounded 64-row smoke run completed first: the baseline matched
54/64 proven actions (84.375%), produced 64/64 terminal wins, and consumed
70,162 nodes. This is a harness sanity check, not a promotion gate; the full
partition run remains paused.

The exporter caught and fixed a canonicalization defect during this gate:
Ring-2 promotion had stored source-orientation actions beside canonical keys.
Rust promotion now transforms action coordinates with the selected D4 symmetry,
and the exporter rejects any non-legal canonical action before producing
training evidence.

## Next decision

Continue bounded expansion passes over Ring-2 unknown stubs, retaining each
pass under `workspace/`, until a measured resource budget is reached. Close
additional low-branching roots by supplying verified records for their missing
child states (or by adding independently replay-witnessed constructive
admissions), then rerun the exact minimax and promotion gates. The first
two-root promotion is now available for training/evaluation against the
permanent held-out split and user games. Keep the direct layered Rust lookup
enabled for exact rows; improve the learner only after a larger Ring-2
training population makes the value gate statistically useful.
