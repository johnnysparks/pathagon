# Architecture

Pathagon has four active planes: the browser product, the rules/search
engines, the research laboratory, and the game archive.

## Runtime ownership

| Plane | Canonical responsibility |
| --- | --- |
| TypeScript in [`app/`](../app/) | Browser game state, UI, reference/coaching behavior, and regression fixtures |
| Rust in [`engine-rs/`](../engine-rs/) | Bitboard rules, search, native self-play, evaluator training, and high-volume generation |
| Python in [`research/gnn/`](../research/gnn/) | GNN/CNN construction, replay training, export, scoring, and learner leagues |
| D1 through [`db/`](../db/) | Durable human and imported self-play archives |

The Rust/WASM adapter is an active integration boundary. It must pass the
cross-runtime parity tests before it becomes the default browser engine. The
browser contract should remain stable while the implementation underneath it
changes.

## Research scope

The canonical model target is 7x7 with 14 reserves per player. The graph code
can exercise other board sizes, but those are curriculum or regression tools,
not part of the primary strength comparison. The fixed CNN is intentionally
7x7 only.

The playable baseline ladder is:

- Coin Flip: uniform random legal action
- Lunatic: deliberately weak one-ply local-pattern heuristic
- Surveyor: shallow broad-beam search
- Pathfinder: deeper iterative search

Search variants must receive distinct agent IDs. Improving Lunatic with board
search, for example, should create a new agent rather than silently changing
the baseline used by old games.

## Data flow

```text
offline games / human games with consent
              ↓
       contract validation
              ↓
       D1 or local archive
              ↓
    dataset split + manifest
              ↓
       model training/export
              ↓
 held-out scoring + pairwise arena
              ↓
     optional promotion / publication
```

Leaderboard standings are a view over imported match records. They are not a
separate source of truth and should never be updated by an undocumented live
generator.

## Agent identity

Every recorded agent should identify its runtime, rules version, evaluator or
model hash, search depth, node budget, beam width, and board configuration.
Display names belong in the UI; stable IDs belong in contracts, archives, and
evaluation reports.
