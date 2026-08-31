# 20260830 Endgame retrograde frontier

Status: Ring-1 pilot complete; Ring-2 generation, compact persistence, and restartable propagation validated

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

## Data and artifacts

Disposable extractor output, checkpoints, and summaries belong in this path's
ignored `workspace/`. The promoted
`fresh-frontier-wdl-v1` artifacts are [the 533,415-byte WDL shard](../../data/golden/tables/fresh-frontier-wdl-v1/7x7-r14/shard-00.bin), [the 889,046-byte action metadata sidecar](../../data/golden/sidecars/fresh-frontier-wdl-v1/7x7-r14/ring-01.bin), and [their manifest](../../data/golden/fresh-frontier-wdl-v1-manifest.json). The compact sidecar is a sorted binary key-to-sparse-action-metadata index so durable data stays below the repository's 5 MiB file limit. Do not place solver traces, queues, or implementation-shaped tensors in durable data.

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

The first training gate is intentionally not a promotion: a linear
gold-aware policy/value/urgency adapter trained on 1,815 Ring-1 rows reached
72.97% witnessed-action accuracy on 185 held-out rows, equal to the frozen v4
control at 72.97%. Ring 1 is a one-move witness layer with incomplete optimal
action sets, so this result is evidence that the gate runs, not evidence of a
stronger model.

## Next decision

Continue bounded expansion passes over Ring-2 unknown stubs, retaining each
pass under `workspace/`, until a measured resource budget is reached. Close the
Ring-2 frontier by supplying verified records for its missing child states (or
by adding independently replay-witnessed constructive admissions), then rerun
the exact minimax and promotion gates. After a non-empty Ring-2
promotion, train and evaluate against the permanent held-out split and user
games before promoting a stronger model. Keep this Ring-1 artifact as the
rollback/control layer until that later ring is independently validated.
