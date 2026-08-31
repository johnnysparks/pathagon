# 20260830 Endgame retrograde frontier

Status: completed

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
and count checks.

## Data and artifacts

Disposable extractor output, the verbose 15.8 MB provenance sidecar, and
summaries belong in this path's ignored `workspace/`. The promoted
`fresh-frontier-wdl-v1` artifacts are [the 533,415-byte WDL shard](../../data/golden/tables/fresh-frontier-wdl-v1/7x7-r14/shard-00.bin), [the 640,116-byte action sidecar](../../data/golden/sidecars/fresh-frontier-wdl-v1/7x7-r14/ring-01.bin), and [their manifest](../../data/golden/fresh-frontier-wdl-v1-manifest.json). The compact sidecar is a sorted binary key-to-proven-actions index so durable data stays below the repository's 5 MiB file limit. Do not place solver traces, queues, or implementation-shaped tensors in durable data.

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
regions unknown, and writes a deterministic value file plus checkpoint.

## Next decision

Continue with a separate Ring-2 path: add a disk-backed predecessor frontier,
exact retrograde propagation, constructive-state reachability checks, and
promotion gates that require deterministic shards, no W/D/L contradictions,
and explicit unknowns at every budget boundary. Keep this Ring-1 artifact as
the rollback/control layer until a later ring is independently validated.
