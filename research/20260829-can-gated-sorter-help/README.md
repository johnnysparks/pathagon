# 20260829 Can a gated sorter help?

Status: `completed · not promoted`

## Idea

Earlier learned root sorters were always active, including on roots where
Pathfinder's native ordering was already clear. This path tested whether a
compact learned sorter could be useful only on ambiguous roots while keeping
the tactical-safe root filter, alpha-beta search, and native ordering
authoritative.

The model was allowed to reorder the first eight surviving native candidates.
It could not invent moves, bypass legality, suppress the tactical-safe filter,
or replace the native fallback.

## Starting point

The reproduced historical artifact was the compact sorter from the 20260827
Rust sorter path:

- artifact: `research/20260827-pathfinder-rust-sorter/artifacts/compact-gnn-policy.onnx`
- size: 187,458 bytes
- SHA-256: `2e403c351396f876ba32f487acd6d53e1b0aaa34d59d28a46ed5e93a26342520`
- rules: `pathagon-rules-v1`, 7×7 board, 14 reserves per player
- historical search control: depth 4, beam 8, 2,000 nodes, tactical-safe
  root filter

The few-seconds-per-turn product envelope was not yet selected by the sibling
compute-budget path, so this first milestone used the frozen 2,000-node profile
for search-result diagnostics. No deployable product claim is made from these
native timings.

## What happened

### Calibration protocol

The research-only Rust harness in `src/main.rs` generated 24 deterministic,
independent random-walk source families. Each family contributed roots at plies
8, 16, 24, 32, 48, and 64. The first 12 families were calibration and the last
12 were untouched holdout, giving 72 roots per split. The split was by source
family, so roots from one walk could not occur in both partitions.

The seven gate definitions were fixed before the run:

| gate | minimum model confidence | maximum native heuristic gap |
| --- | ---: | ---: |
| strict | 0.40 | 100 |
| high-confidence | 0.20 | 250 |
| balanced | 0.10 | 250 |
| low-confidence | 0.05 | 500 |
| permissive | 0.00 | 500 |
| ambiguous-only | 0.00 | 100 |
| always-on control | none | none |

Confidence is the model-logit advantage over the native first action. Native
ambiguity is the successor heuristic gap between the first two safe native
actions. The audit recorded model/native ordering, teacher top-1 agreement,
score regret, tactical category, and whether the completed Pathfinder search
result changed.

The teacher was depth 5, beam 16, and 6,000 nodes over the full safe root set.
It exhausted its budget on all 144 roots, completing depth 2 on 42 roots, depth
3 on 13, and depth 4 on 89. Consequently, score-regret values are retained as
diagnostic output only; they are not used as promotion evidence.

### Holdout result

The native baseline was the better teacher top-1 reference on every tested
gate. `search changes` counts only roots where an activated model reorder
changed the completed historical Pathfinder action.

| gate | activated | activation rate | native top-1 | selected top-1 | activated search changes | tactical regressions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strict | 29/72 | 40.3% | 18/72 (25.0%) | 15/72 (20.8%) | 2/29 | 0 |
| high-confidence | 42/72 | 58.3% | 18/72 (25.0%) | 11/72 (15.3%) | 7/42 | 0 |
| balanced | 44/72 | 61.1% | 18/72 (25.0%) | 11/72 (15.3%) | 8/44 | 0 |
| low-confidence | 56/72 | 77.8% | 18/72 (25.0%) | 6/72 (8.3%) | 11/56 | 0 |
| permissive | 57/72 | 79.2% | 18/72 (25.0%) | 5/72 (6.9%) | 11/57 | 0 |
| ambiguous-only | 32/72 | 44.4% | 18/72 (25.0%) | 14/72 (19.4%) | 3/32 | 0 |
| always-on control | 66/72 | 91.7% | 18/72 (25.0%) | 4/72 (5.6%) | 13/66 | 0 |

The strict and ambiguous-only gates demonstrate that activation can be made
selective, but neither preserves native agreement. Lowering the confidence or
loosening the ambiguity bound worsens the holdout result. The always-on control
reproduces the historical failure mode. This rejects the hypothesis that the
reproduced compact sorter has a useful calibrated activation region.

Holdout category coverage was 39 placement, 7 movement, 12 late-movement, 13
multi-capture, and 1 forced-block root. There were no immediate-win roots, so
the immediate-win safety category was not sufficiently exercised to support a
promotion claim. No selected action violated legality or the tactical-safe
root set.

### Runtime diagnostic

On the reference native release build, model inference measured 7.8 ms median,
9.0 ms p95, and 16.9 ms maximum across the 144 roots. The native historical
search measured 52.6 ms median and 192.0 ms p95; the model-reordered search
measured 48.2 ms median and 185.0 ms p95. These are native diagnostics, not the
browser/WASM product benchmark required for promotion. Since the calibration
gate failed and the product envelope was not frozen, no 400-game arena was
run.

The exact command was:

```text
cargo run --release --manifest-path research/20260829-can-gated-sorter-help/Cargo.toml -- --families 24 --holdout-families 12 --output research/20260829-can-gated-sorter-help/workspace/calibration.json
```

The full per-root report, including raw score regrets and representative
activated roots, remains in the ignored
`research/20260829-can-gated-sorter-help/workspace/calibration.json` file.

## Data and artifacts

The reusable research harness is preserved in `Cargo.toml`, `Cargo.lock`, and
`src/main.rs`. The full calibration JSON and the smaller smoke report remain
ignored under `workspace/`; no generated corpus, model, checkpoint, target
tensor, replay export, or browser trace was promoted.

The historical ONNX artifact remains ignored in its parent research path and is
not selected for deployment. Its hash is recorded above for reproducibility.

## Project impact

No supported Rust, browser, manifest, or data contract changed. The tactical-safe
native filter and trained v0.5 evaluator remain the supported strength line.
This path adds evidence that confidence and native ambiguity alone cannot make
the historical compact sorter safe and useful, and it prevents another broad
sorter model sweep before a new model demonstrates a calibrated held-out
signal.

The final strength, latency, WASM fallback, legality, and representative-game
promotion gates were intentionally not claimed: the preliminary calibration
gate failed, the product envelope was unresolved, and immediate-win coverage
was absent. In particular, no runtime sorter or fallback implementation was
promoted.

## Next decision

Retire the gated-sorter branch for the current cycle. Do not promote the
historical compact sorter or train additional sorter variants. Revisit only
after the product compute envelope is frozen and a new model supplies a
source-disjoint calibration set with adequate immediate-win and forced-block
fixtures, an explicit native fallback, and a positive whole-game result.
