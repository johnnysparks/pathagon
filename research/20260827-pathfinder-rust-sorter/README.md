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
to `data/corpora/games-v1`. The native Rust runner now owns the complete
game archive beside every summary, so new screens have replayable move
membership; older summary-only screens retain their historical gap.

The native target emitter is `pathagon/engine-rs/src/bin/pathfinder_targets.rs`. It
replays those Rust archives and asks the configured Rust Pathfinder search for
exact action targets. `--target-temperature 750` emits a soft policy over
the scored legal actions, while the default emits a one-hot best-action target;
the resulting JSONL remains consumable by the existing learner.

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
- Exact native targets from 400 baseline games (11,357 positions) produced a
  59–61 one-hot screen. Soft targets from the same corpus produced 64–56,
  51–69, and 64–55–1 across three fresh screens (184–175–1 aggregate), which
  is directionally positive but not significant enough to promote.
- Scoring every action with the larger soft-target model fell 43–77, and a
  temporary PVS search-side ablation fell 52–68 without reducing nodes. Both
  are retained as negative controls; the default search remains the staged
  alpha-beta Pathfinder path.
- Removing the value-loss term from the same 400-game soft-target recipe gave
  63–57, 50–70, and 61–59 (174–186 aggregate), so policy-only optimization is
  a negative control rather than a convergence shortcut.
- Keeping the neural order but lifting the root cap to all legal actions fell
  50–69–1 and consumed about 23% more nodes; the cap is therefore retained as
  an explicit compute/strength control, not hidden default behavior.
- Rust now emits independent per-action rank targets with `rankActions`,
  `rankScores`, `rankExhausted`, and `rankNodesUsed`; the learner trains a
  pairwise rank loss over those targets. The 400-game top-8 corpus reached
  41.3% rank top-1 and 66.9% pairwise accuracy offline, but the native rank
  sorter scored 161–197–2 across three 120-game screens. The top-2 target
  variant finished 180–179–1, so neither is promoted.
- The pure-Rust root-probe control (`--candidate probe-search`) runs a bounded
  shallow alpha-beta scout and charges it against the shared node ceiling. At
  depth 2/256 nodes/8 actions it scored 59–61 against unmodified depth-4
  Pathfinder; depth 1/64/8 scored 55–65 and depth 2/512/16 scored 51–69.
  It remains an opt-in search experiment rather than the default.
- Full-root transposition-table/killer/history ordering was exactly even at
  60–60. Selective depth-5 controls with beams 4 and 6 regressed 21–99 and
  33–86. The immediate-threat root guard had one 65–54–1 screen, but its three
  screens aggregate to 181–176–3, so it is not a promotion result.
- The hard tactical-safe root filter is a separate native search improvement,
  not a learned sorter. It removes a root move only when the opponent has an
  immediate winning reply and at least one safe alternative exists; otherwise
  it searches the full root set. Same-depth screens aggregate to 659–496–5 over
  1,160 games. Against unmodified depth-5 Pathfinder with the same 2,000-node
  ceiling, a depth-4 filter candidate scored 313–86–1 and 316–83–1, or
  629–169–2 over 800 games. This is strong evidence for the pure-Rust
  `pathfinder-v0.4.0-tactical-filter` default, while the learned sorter and
  rank models remain unpromoted.
- The Rust target emitter now accepts `--tactical-filter`. The 400-game filtered
  archive produced 13,759 replayable positions with eight independent rank
  targets per position; the output was accepted by the existing learner.

## Decision

Promote the native tactical-safe filter as the model-free Pathfinder default,
while keeping the learned sorter/rank models unpromoted. Continue the sorter
experiment with Rust-owned play, exact canonical move retention, and larger
paired samples; `--no-tactical-root-filter` is the frozen unfiltered control.

## Canonical games

The Python pilots and all archive-complete native screens are retained through
the observation source IDs listed in `manifest.json`. Their duplicate JSONL
payloads were removed after ingestion. Earlier native summary-only screens have
no canonical game membership.

The active native baseline added another 80 canonical games. Its derived
depth-5 Pathfinder target set remains local and ignored while the experiment is
active; it is an implementation-shaped training intermediate, not part of game
identity. The filtered target archive is likewise a reproducible training
intermediate, not a new game identity.

## Artifacts

The selected 187 KB ONNX sorter and the newer native-target/native-soft
exports remain as ignored local artifacts under `artifacts/`; they are not
present in the current Git tree. The selected sorter can be recovered from
the pre-refactor Git snapshot recorded in `manifest.json`. Keep a local or
external backup before deleting the ignored files.
