# 20260827 Pathfinder Rust sorter

Status: `running`

## Hypothesis

A small learned policy or Q/Advantage model can improve Pathfinder by ordering
its bounded Rust alpha-beta root candidates without replacing Pathfinder's
rules, evaluator, or search authority.

Pathfinder remains the north-star agent. The neural artifact is only a sorter;
it is not a standalone promoted opponent.

## Lineage and protocol

- Parent/baseline: `rust-pathfinder-v0.3.0`
- Candidate: `rust-pathfinder-onnx-sorter-v0.1.0`
- Selected model artifact hash:
  `sha256:2e403c351396f876ba32f487acd6d53e1b0aaa34d59d28a46ed5e93a26342520`
- Source model hash:
  `sha256:5e1de60645e7b94b2d0921f73cd6287db8a9e8a1bb85c5960ce1f098509f29e3`
- Rules/configuration: 7x7, 14 reserves, `pathagon-rules-v1`
- Frozen native screen: depth 4, beam 8, 2,000 nodes, paired colors,
  two randomized opening plies
- Source revision: `e6e72f9692967147a82e696d26b7add73f31a828`

Representative screens are normalized in [`results.tsv`](results.tsv). Python
pilots established integration behavior and contributed their completed games
to `research/corpora/games-v1`. The native Rust runner now owns the complete
game archive beside every summary, so new screens have replayable move
membership; older summary-only screens retain their historical gap.

## Findings so far

- Policy top-k 2 was approximately even over 120 games (62–58), as was the
  Q/Advantage sorter (61–59). Neither is promotion evidence.
- The imitation checkpoint and tactical-extension variant both lost 49–71,
  useful negative results that rule out those recipes at the tested budget.
- Root-limited and transposition-table screens produced positive 40-game
  samples, but repeated samples changed direction. The latest `tt-root16`
  top-k 2 run finished 20–19–1.
- Scoring every legal action did not produce a stable gain and adds inference
  work. It is not the default direction.
- Three heuristic-gap guards (100, 250, and 500 Pathfinder score points) were
  even or negative at 40 games. A root-capped killer-move ablation was also
  mixed and slower. The uncapped ordering-only control lost 48–71 over 120
  games, so the compact policy is not promotion evidence.

## Decision

Keep the experiment active but unpromoted. Continue only with Rust-owned play,
Pathfinder as the frozen baseline, exact canonical move retention, and larger
paired samples. Discard obsolete Python training intermediates and failed model
variants after preserving this result history.

## Canonical games

The Python pilots and all archive-complete native screens are retained through
the observation source IDs listed in `manifest.json`. Their duplicate JSONL
payloads were removed after ingestion. Earlier native summary-only screens have
no canonical game membership.

The active native baseline added another 80 canonical games. Its derived
depth-5 Pathfinder target set remains local and ignored while the experiment is
active; it is an implementation-shaped training intermediate, not part of game
identity.

## Artifacts

The selected 187 KB ONNX sorter and its manifest are checked in under
`artifacts/`. Other checkpoints, optimizer state, materialized datasets, and
temporary export variants are disposable legacy data.
