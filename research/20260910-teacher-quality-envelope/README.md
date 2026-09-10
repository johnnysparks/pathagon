# Teacher quality at the deployment envelope

Status: completed on 2026-09-10 — no promotion

## Idea

The three-generation learning run used deeper search to create policy targets,
but never established that those teacher choices were better than the deployed
depth-5 player. This diagnostic isolates that missing link. It compares the
deployed transition-policy search with the deeper teacher on held-out positions
whose acceptable action set is independently adjudicated by rules or a
rules-only proof search.

## Starting point

The evaluated model is the supported v4 transition scorer,
`data/models/pathfinder-action-transition-v4-xent/transition-policy.json`,
SHA-256 `f11d7ddee101ccab35ee162e53c95ced076b1fb10242443ad562dbd51c1085d4`.
The engine, evaluator weights, tactical filter, and 7×7/14-reserve namespace
are unchanged. Deployment is depth 5 / beam 256 / 256,000 nodes. The teacher is
depth 6 / beam 256 / 512,000 nodes. Both use the same transition model and
frozen evaluator; the raw control uses the same deployment budget with the
learned root ordering removed.

## What happened

The Rust harness in `src/main.rs` evaluated two disjoint diagnostic suites.

The terminal suite sampled 32 evenly spaced keys from the 3,597-key held-out
partition of the replay-witnessed Ring-1 frontier. Every sampled root is a
winning position in the promoted Ring-1 WDL table. The oracle enumerates every
legal action and accepts exactly the actions whose transition creates the
side-to-move winning path. The complete action set was cross-checked against
the persisted Ring-1 proof sidecar before scoring any search result.

The proof suite scanned the canonical replay corpus and retained 32 unique
positions from the game-key holdout bucket `sha256(game-key) mod 5 = 0`. It
excluded immediate wins, used the independent Rust AND/OR proof solver at a
three-ply horizon and 50,000-node ceiling, and kept only non-exhausted,
discriminating roots: at least one action was provably better than another.
Twenty-seven roots were finite-horizon draws (safe actions exist) and five were
forced wins. A proof loss was not treated as an action-quality case because all
legal moves are losing within that horizon.

The measured quality was identical across all three selectors:

| Suite | Roots | Deployment | Teacher | Raw deployment | Teacher-only wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ring-1 immediate wins | 32 | 32/32 (100%) | 32/32 (100%) | 32/32 (100%) | 0 |
| Three-ply proof roots | 32 | 32/32 (100%) | 32/32 (100%) | 32/32 (100%) | 0 |

The teacher and deployment selected different actions on two proof roots; both
actions were in the complete oracle set. There were no cases where the teacher
was oracle-valid and deployment was not. Deployment completed an iterative
depth of 4.59 on average in the proof suite versus 5.06 for the teacher, while
the mean node counts were 128,578 and 342,437 respectively. On the terminal
suite the corresponding means were 224,426 and 455,097 nodes. The larger
teacher budget therefore bought deeper work and more exhaustion, not a better
decision on this adjudicated sample.

The durable summary is [`results.json`](results.json). The command below also
writes the detailed, ignored `workspace/final.json`, which includes every
sampled state, the complete oracle action set, selected actions, legality,
exhaustion, completed depth, and node counts.

```bash
cargo run --release \
  --manifest-path research/20260910-teacher-quality-envelope/Cargo.toml \
  --bin teacher-quality -- \
  --heldout data/golden/partitions/fresh-frontier-wdl-v1/ring-01-heldout.txt \
  --model data/models/pathfinder-action-transition-v4-xent/transition-policy.json \
  --golden-table data/golden/tables/fresh-frontier-wdl-v1/7x7-r14/shard-00.bin \
  --golden-sidecar data/golden/sidecars/fresh-frontier-wdl-v1/7x7-r14/ring-01.bin \
  --corpus-dir data/corpora/games-v1/games \
  --out research/20260910-teacher-quality-envelope/workspace/final.json \
  --max-terminal 32 --max-proof 32 --proof-horizon 3 --proof-nodes 50000
```

## Data and artifacts

The research harness, lockfile, and this narrative remain in Git. The report
and smoke outputs are under the ignored `workspace/` directory. No games,
checkpoints, tensors, or new model were promoted into `data/`; the existing
golden shards and v4 model are referenced read-only.

## Project impact

This path closes the teacher-quality measurement gap for the tested tactical
deployment envelope. It shows that the current teacher is at least as good as
deployment on the sampled solved/proved roots, but provides no evidence that
the extra depth creates a learnable strategic improvement. The production
engine, model, and user-facing default are unchanged.

## Hiccups and limits

The first report mislabeled finite-horizon proof outcomes of zero as losses;
the harness was corrected and the full 32+32 run was repeated. The original
sample size of 16+16 was retained only as smoke evidence.

The terminal table stores historyless canonical keys, so decoded roots have a
fresh `ply` value; they are used here only for the immediate-transition oracle,
not for live repetition or max-ply claims. The proof oracle is finite-horizon,
not a complete 7×7 tablebase, and its draw label means “no forced result within
three plies.” The corpus holdout is disjoint by game-key bucket within the
source corpus, but this diagnostic cannot prove that no model-training archive
shares those games. Thirty-two roots per suite are sufficient to expose a
large teacher gap, not to certify tiny strategic differences.

## Next decision

Retire this path as a negative teacher-quality result and keep the current
default. Do not spend another training campaign on deeper labels alone. The
next useful learning experiment should create labels that distinguish quiet
multi-ply choices beyond this tactical proof envelope, then repeat this oracle
check before any arena or promotion decision.
