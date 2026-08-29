# 20260828 Seeded position curriculum

Status: `completed`

## Hypothesis

A controlled mixture of ordinary, reachable-midgame, parity-balanced, and
legal synthetic roots will improve near-terminal policy recall, action ranking,
and value calibration without sacrificing whole-game 7x7 strength.

## Lineage

- Parent experiment: `20260827-pathfinder-rust-sorter`
- Parent/baseline agent: `rust-pathfinder-v0.4.0-tactical-filter`
- Candidate family: `seeded-gnn-puct-v0.1.0`
- Git revision at setup: `a97b09ab`

## Protocol

- Rules: `pathagon-rules-v1`, 7x7, 14 reserves, 196-ply cap.
- Root classes: ordinary, reachable, parity-balanced, asymmetric synthetic.
- Seeded conditions: 0%, 25%, and 50%; within seeded roots, 40:30:30
  reachable/parity/asymmetric composition.
- Near-terminal target: minimum rules-aware connection distance ≤3, balanced
  across side to move and color where possible.
- Search: Python GNN PUCT for continuation generation, with root action scores
  from the Pathfinder guide. The follow-up native emitter then rewrote every
  position with depth-4, 1,500-node, beam-8 Pathfinder targets, tactical-safe
  filtering, temperature-750 soft policy, and top-8 rank targets (rank budget
  1,000 nodes).
- Splits: by root/game seed; promotion tactical fixtures remain held out.

## Run result

The matched run generated 288 games (96 per condition) and 16,695 replayable
positions in total. Training used the common 5,194-position slice per
condition, 400 updates, the same 32-wide/4-layer GNN, value loss weight 1.0,
rank loss weight 0.25, and symmetry augmentation. Both the policy/value model
and the Q/advantage head were retrained for every condition.

| seeded fraction | ordinary / reachable / parity / asymmetric | near roots | target positions | average plies |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 96 / 0 / 0 / 0 | 0/96 | 6,050 | 63.02 |
| 25% | 72 / 10 / 7 / 7 | 5/96 | 5,451 | 56.78 |
| 50% | 48 / 19 / 14 / 15 | 20/96 | 5,194 | 54.10 |

The 220-game color-balanced ranked ladder included all six retrained models,
the parent checkpoint, Pathfinder, Surveyor, Lunatic, and Coin Flip. Pathfinder
finished at 1,198 Elo, the parent learned checkpoint at 1,109, and the seeded
policy/value variants at 1,032 (0%), 1,010 (25%), and 1,001 (50%). The QAdv
variants finished at 919 (0%), 913 (50%), and 830 (25%). The parent beat the
0% and 50% policy/value candidates 4-0 and split 2-2 with the 25% candidate;
all three QAdv candidates lost 4-0 to the parent.

## Decision

Do not promote a seeded checkpoint or QAdv head from this short ladder. The
curriculum implementation, validator, replay provenance, native target path,
and canonical corpus linkage are retained. The result supports the coverage
hypothesis (50% produced 20 near roots versus none in the control), but does
not support a strength hypothesis: every retrained candidate remained below
the parent and Pathfinder, and direct QAdv selection was especially weak. A
longer, held-out tactical/ordinary evaluation and a target-repair or longer
training run are required before increasing the seeded share.

## Artifacts

Large generated games, score vectors, checkpoints, and match archives remain
under this path's ignored `workspace/`. This directory retains only
small summaries, hashes, canonical game references, and the final decision.

The generated game and target archives were linked into
`data/corpora/games-v1` (source IDs are recorded in `sources.tsv`), raising
the corpus to 38,739 unique games and 129,433 observations with zero ingest
errors.
