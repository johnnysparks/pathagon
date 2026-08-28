# Pathagon AI strength experiment roadmap

Date: 2026-08-27

## Goal

Make the canonical 7x7, 14-reserve opponent materially harder to defeat while
keeping move time and browser delivery practical. Strength means repeatable,
color-balanced wins against held-out opponents and tactical correctness—not a
lower training loss or a higher provisional Elo alone.

## What the evidence says now

Pathagon already has the major ingredients of a modern board-game program:
deterministic rules, alpha-beta-style heuristic search, PUCT, policy/value CNN
and GNN models, a Q/Advantage head, high-throughput Rust self-play, adversarial
case mining, and disjoint evaluation support. The next gains should come from
making these pieces produce reliable policy improvement rather than adding
another model family immediately.

The current evidence identifies four bottlenecks:

1. **Tactical blindness.** On the 24-position 4x4 audit, unguarded neural search
   found all immediate wins but 0/8 forced blocks and 0/8 forced forks at both
   32 and 128 simulations. A generic three-ply proof guard recovered all 24.
2. **Weak action ranking.** The latest held-out Q/Advantage evaluation covers
   218,909 positions. Pairwise ordering accuracy is 63.5%, but the played move
   is the target-Q maximum only 7.0% of the time; relocation positions are the
   hardest, with an average selected-action rank of 114.3.
3. **Prediction metrics do not imply playing strength.** In 2,500 clean,
   unique-opening cross-play games, the Q/Advantage player went 0–1,000 against
   Pathfinder, 0–500 against Surveyor, 0–500 against Lunatic, and 498–2 against
   Coin Flip. This is a useful failed experiment: the head distinguishes random
   play but has not learned robust adversarial choice.
4. **Historical self-play lacked diversity.** A prior audit found 25,732 local
   records but only 6,586 unique trajectories (74.4% exact repeats). The newer
   17,500-game Rust campaign fixed the mechanics—17,500 unique openings, no
   exact duplicate games, and full Q coverage—so future work can start from a
   much cleaner corpus.

The current strongest recorded league member is still Pathfinder, with 13–1 in
the latest 15-player diagnostic round. That league used only 14 games per
player, so it is a scouting result, not a promotion-grade estimate.

### Implementation status after this roadmap

The QAdv tree-guidance slice is now implemented in Rust: QAdv-guided PUCT can
opt into action-value seeds at every expanded node, with a single combined
policy/value/Q inference call. The flag is disabled for ordinary policy/value
and browser/WASM paths, and a regression test covers a non-root expansion.
The bounded rule-grounded proof extension described in E1.1 is also wired into
the native QAdv player. It is opt-in by horizon, triggers only on sharp tactical
states, uses QAdv values to order the root, and refuses to act when its fixed
node budget is exhausted. A 7x7 immediate-win regression covers the expanded
solver boundary. QAdv guidance still improves ordering and priors; only the
bounded rule search proves a forced result.

The first reproducible fixture and ablation artifacts are now checked in:
`scripts/build-tactical-suite.py` emits the deterministic 300-position,
solver-labelled suite in `research/fixtures/tactical-suite-300.jsonl`, and
`scripts/benchmark-rust-qadv-ablation.py` runs root-only, tree-seeded, proof,
and 2x-simulation variants on identical seeds. Transposition-aware graph
search is deliberately reported as pending rather than being treated as a
completed control.

The first expanded local audit is recorded in
`research/fixtures/tactical-suite-300-evaluation.json`. On these 300
solver-labelled roots, unguarded Python PUCT reached 6.5% immediate-win policy
accuracy at zero simulations, 43.5% forced-defense accuracy at 32 simulations,
and 0% forced-fork accuracy at 0/32/64 simulations. The rule-priority guard
raised policy accuracy to 100% on immediate wins and forks and 87.0% on forced
defenses. Its visit policy is intentionally unchanged, so the fork visit
accuracy remains 0%; this is why the native experiment selects a proof action
after rule search rather than only rewriting the training distribution.

A compact-root-sorter experiment is now wired into the Python league. The
17,475-parameter warm-start GNN only reorders Pathfinder's existing root beam;
the baseline and candidate keep the same alpha-beta depth, beam, and node
ceiling. The candidate also has a transposition-aware table and one-ply
tactical extension as a separate smarter-search variant. The matched screen
is reproducible with `scripts/benchmark-pathfinder-sorter.py`. In a 20-game
5x5 screen (depth 3, beam 8, 2,000 nodes, two randomized opening plies), the
compact sorter won 13–7, while the sorter-only ablation produced the same
13–7 result. On a short 7x7 screen with the same budget, the sorter did not
yet beat Pathfinder (2–6 at depth 3); at matched depth 4 it scored 5–3 in
eight games. These are promising but not promotion-grade samples, so the
next gate is a longer, color-balanced 7x7 run before changing training data.

The same experiment is now available in the native engine. `tract-onnx` loads
the policy/value graph in Rust, while Pathfinder's alpha-beta evaluator and
rules remain the source of truth; Python is only a benchmark launcher. The
native command is `--sorter-onnx ... --sorter-top-k K --opponent deep-search`.
The QAdv graph can also be used as a root sorter with
`--sorter-qadv-onnx ...`; its action-value head is a hint and does not replace
the rule-grounded search result. A 120-game compact-policy screen at depth 4,
beam 8, and 2,000 nodes produced 62–58 at top-k 2 on one seed and 56–64 at
top-k 4 on another, so this is not yet a promotion-grade win. The larger
64x8 GNN control scored 16–24 in a 40-game screen, and a deeper-Pathfinder
imitation checkpoint scored 49–71 in a 120-game screen. These negatives are
why the roadmap still requires a longer fixed-seed gate before generating new
training data from the sorter.

The QAdv-as-sorter path is also implemented in Rust/ONNX. Its 120-game
top-k-2 screen finished 61–59, which is a useful tie-level control but not a
promotion result.

An optional `--sorter-all-actions` mode lets ONNX rank the complete legal root
set before Pathfinder keeps the top-k hints. Three 40-game chunks (120 games
total) with the compact policy produced 64–56, while a 40-game QAdv all-actions
screen was exactly 20–20. The all-actions policy mode is the best current
screen, but its 53.3% aggregate is still below the predeclared promotion gate
and costs roughly 2–3x the move time, so it remains an experiment.

The tactical leaf extension is implemented behind an explicit search option
but is disabled in the default sorter after a matched 120-game screen fell to
49–71. The tactical root guard remains bounded and ordering-only. This keeps
the native candidate's current comparison focused on the ONNX root signal.

The next native iteration now emits exact Pathfinder targets directly from
Rust with `engine-rs/src/bin/pathfinder_targets.rs`. One-hot targets from 400
native baseline games (11,357 positions) produced 59–61 in a 120-game screen.
Soft targets at temperature 750 trained on that same corpus produced
64–56, 51–69, and 64–55–1 across three fresh screens (184–175–1 aggregate),
which is directionally positive but below the promotion gate. Scoring every
legal action fell 43–77, and a temporary PVS ablation fell 52–68 without
reducing node count, so neither is retained as the default. These results keep
the exact-target loop active while preserving Pathfinder as the search
authority.

The matching policy-only optimizer ablation (value-loss weight 0) scored
63–57, 50–70, and 61–59 (174–186 aggregate), so removing the value head's
training signal did not provide a reliable sorter improvement.

Finally, retaining the learned order while lifting the root cap to all legal
actions scored 50–69–1 and used about 23% more nodes. The capped top-k search
therefore remains the active compute/strength control.

The rank-focused follow-up is implemented end to end. Rust can emit
independent per-action Pathfinder rankings (`rankActions`/`rankScores`) with
the cumulative budget metadata, and the learner applies a pairwise ranking
loss to those targets. The 400-game, 11,357-position top-8 corpus reached
41.3% rank top-1 and 66.9% pairwise accuracy offline, but its fresh native
screens were 161–197–2 in aggregate; a top-2 target variant was 180–179–1.
These are useful diagnostics, not a reason to promote a model.

There is also a pure-Rust `search_best_action_with_root_probe` control. It
spends a bounded shallow alpha-beta scout on Pathfinder's first root actions,
charges those nodes against the same total budget, and feeds the resulting
order into the full search. Against the unmodified depth-4 Pathfinder at the
same 2,000-node ceiling, depth-2/256-node/8-action probing scored 59–61,
depth-1/64-node/8-action probing scored 55–65, and depth-2/512-node/16-action
probing scored 51–69. The probe is therefore available as an opt-in search
experiment, while the ordinary Pathfinder path remains the incumbent.

The same native search now exposes two additional ordering controls without
changing the evaluator: full-root transposition-table/killer/history ordering
and the bounded immediate-threat root guard. The former was exactly even at
60–60, while depth-5 selective variants (beam 4 and 6) regressed 21–99 and
33–86. The guard's first 120-game screen was 65–54–1, but two fresh screens
were 60–59–1 and 56–63–1 (181–176–3 aggregate). That initial spike was noise;
the guard remains an opt-in tactical experiment and Pathfinder remains the
incumbent.

The next iteration adds a hard, rule-grounded tactical-safe root filter. When
the root contains both risky and safe moves, it removes moves that allow the
opponent an immediate winning reply; if there is no safe/risky split it falls
back to the complete legal root set. This preserves Pathfinder's evaluator and
alpha-beta authority while preventing a directly refutable root choice. At the
same 2,000-node ceiling and equal depth, five screens totaled 659–496–5 over
1,160 games. More importantly, a shallower depth-4 filter candidate against
unmodified depth-5 Pathfinder scored 313–86–1 and 316–83–1 (629–169–2 over 800
games, 78.8% of decisive games). The native filter is now the default
model-free Pathfinder variant (`rust-pathfinder-v0.4.0-tactical-filter`), with
`--no-tactical-root-filter` retaining the unfiltered control for A/B tests.

The stronger search is also feeding the next training loop. The Rust target
emitter accepts `--tactical-filter` and writes the same one-hot/soft policy and
pairwise rank metadata from the filtered root. A fresh 400-game archive yielded
13,759 replayable positions with eight rank targets each; the resulting JSONL
loads through the existing learner without a second rules implementation.

The follow-up native screens kept the learned sorter unpromoted. Heuristic-gap guards
of 100, 250, and 500 Pathfinder score points were even or negative in 40-game
chunks. Root caps of 1–8 learned candidates also regressed, and the uncapped
ordering-only control finished 48–71 over 120 games. A killer-move ordering
ablation increased latency and changed direction across 40-game chunks. These
results make the current compact checkpoint a useful integration fixture, not
a strength checkpoint; the next training run needs exact native move targets
and a held-out, paired Rust gate before it can feed improved search data.

To regenerate the fixtures and rerun the Python tactical ablation:

```sh
./.venv-pathagon-gnn/bin/python scripts/build-tactical-suite.py \
  --output research/fixtures/tactical-suite-300.jsonl
./.venv-pathagon-gnn/bin/python scripts/evaluate-4x4-endgame.py \
  --suite research/fixtures/tactical-suite-300.jsonl \
  --budgets 0,32,64 \
  > research/fixtures/tactical-suite-300-evaluation.json
```

For native QAdv controls, build the release inference binary and run
`scripts/benchmark-rust-qadv-ablation.py`; it keeps seed, opening policy,
opponent, and tactical simulation budget fixed while sweeping root-only,
tree-seeded, proof, and 2x-simulation variants.

For the compact-sorter screen, use the checked-in checkpoint and keep the
baseline/candidate budgets matched:

```sh
./.venv-pathagon-gnn/bin/python scripts/benchmark-pathfinder-sorter.py \
  --size 7 --reserve 14 --opening-random-plies 2 \
  --depth 4 --beam 8 --nodes 2000 --candidate-depth 4 \
  --top-k 4 --games 200 \
  --out research/runs/gnn/league/pathfinder-sorter-7x7-screen.json
```

## Experimental operating system

Every experiment should change one primary variable and use the same frozen
evaluation protocol.

### Frozen benchmark suite

Use three complementary evaluations:

- **Tactics:** solver-labelled positions, expanded from the existing 24 cases
  to at least 300 positions across immediate wins, forced defenses, forks,
  captures, relocations, repetition avoidance, and quiet setup moves.
- **Cross-play:** paired, color-swapped games against Pathfinder, Surveyor,
  Lunatic, the current neural champion, one historical champion, and a random
  sanity check. Use identical opening seeds for incumbent and candidate.
- **Stress play:** adversarially mined positions and openings where candidate
  versions disagree, evaluation margins are small, or a prior candidate loses.

Record win/draw/loss by opponent and color, tactical top-1 accuracy, policy
entropy, nodes and inference calls per move, median and p95 move time, unique
trajectory rate, and termination reason. Report confidence intervals for match
rates. Do not promote on loss curves, Q error, or Elo alone.

### Promotion gate

A candidate advances only if it:

- has no statistically credible regression against any anchor opponent;
- improves paired score against Pathfinder and the incumbent champion;
- reaches at least 95% on the expanded one-to-three-ply tactical suite, with no
  missed immediate wins or forced one-ply defenses;
- stays within the chosen browser move-time budget; and
- passes the existing rule, replay, symmetry, and native/WASM parity checks.

Start with 200 paired games per serious matchup for screening. Increase to at
least 1,000 paired games for a promotion candidate, or use a predeclared
sequential test with explicit accept/reject boundaries. Keep all promotion
seeds and adversarial cases out of training.

## Roadmap

### Phase 0 — establish the strength scoreboard (2–3 days)

**E0.1: Reproducible incumbent baseline**

- Freeze Pathfinder, the best neural checkpoint, and the current QAdv-guided
  configuration by agent ID, hash, search budget, and runtime.
- Run the frozen tactical suite and 200 paired games per anchor matchup.
- Deliver one report containing strength, latency, and compute per move.

**Decision:** no later result is credible unless rerunning the baseline on the
same seeds reproduces its score and latency within the declared tolerance.

**E0.2: Loss taxonomy**

- Replay a stratified sample of Pathfinder and neural losses.
- Label the first decisive mistake as immediate tactic, horizon error, move
  ordering/pruning error, value error, policy omission, repetition, or unknown.
- Use counterfactual root search to measure how much extra depth or simulation
  budget is required to repair each loss.

**Decision:** the distribution determines whether Phase 1 emphasizes proof
search depth, graph reuse, or action-ranking recall.

### Phase 1 — repair tactical search before retraining (1–2 weeks)

This is the highest-confidence path because the local audit already shows that
generic proof search fixes cases the network misses.

**E1.1: Neural-ordered proof extension**

- Generalize the existing small-board solver into a bounded tactical extension
  available on 7x7.
- Trigger it selectively at the root and unstable leaves: immediate threats,
  captures, sharp connection-distance changes, small policy margins, or high
  value uncertainty.
- Order legal moves with the policy/Q head but prove outcomes from game rules.
- Compare horizons 2, 3, and 4 and fixed node budgets.

**Hypothesis:** rule-grounded extensions recover forced blocks and forks with a
smaller latency penalty than multiplying all MCTS simulations.

**Success:** ≥95% tactical accuracy and a positive paired score delta against
Pathfinder at no more than 2x incumbent p95 move time.

**E1.2: Monte Carlo graph search / transposition reuse**

- Key nodes by the complete rule-relevant state, including turn, reserves,
  forbidden squares, relocation restrictions, and repetition context.
- Compare the existing tree with a transposition-aware directed acyclic graph.
- Measure unique states evaluated, cache reuse, tactical accuracy, strength,
  and memory—not only raw node count.

**Hypothesis:** placements and relocations transpose often enough that graph
reuse buys additional effective depth under the same inference budget.

**E1.3: Low-budget root allocation**

- A/B test current PUCT against Gumbel action sampling plus sequential halving
  at 16, 32, 64, and 128 simulations.
- Keep the network, openings, and total inference budget fixed.

**Hypothesis:** the Gumbel variant improves candidate-action coverage and move
quality when the root has many legal relocation actions.

### Phase 2 — make self-play adversarial and informative (1–2 weeks)

**E2.1: Diverse opening and opponent mixture**

- Generate equal-size batches using (a) current self-play, (b) temperature and
  top-k openings, and (c) a league mixture of current, historical, heuristic,
  and targeted exploiter opponents.
- Cap repeated trajectory families and maintain color/opening balance.
- Compare unique-game rate, relocation share, tactical-state coverage, max-ply
  rate, and downstream paired strength per 1,000 generated games.

**Success:** the mixed batch improves strength per generated position without
  lowering tactical accuracy. Raw archive size is not a success metric.

**E2.2: Prioritized hard-position replay**

- Sample positions using a mixture of uniform replay and priorities: large
  search-vs-network disagreement, high TD/Q error, small top-two Q margin,
  tactical proof correction, candidate disagreement, and known losses.
- Sweep prioritized fractions of 0%, 25%, 50%, and 75% with importance weights
  or capped priorities to prevent a new attractor.

**Hypothesis:** 25–50% hard replay improves rare tactical and relocation choices
without overfitting the adversarial suite.

**E2.3: League exploiters**

- Maintain immutable snapshots of promoted candidates.
- Train one general player against a weighted mixture and short-lived
  exploiters targeted at the general player's current losses.
- Evaluate the general player against the full frozen roster; do not ship an
  exploiter that merely beats one opponent.

### Phase 3 — retrain targets and heads (2–3 weeks)

Run these only after Phase 1 produces better search targets.

**E3.1: Search-policy distillation**

- Train the policy on the improved search distribution, including proof-search
  corrections, rather than the action actually played.
- Compare visit-count, Gumbel-improved, and proof-corrected policy targets.
- Measure top-k recall of solver-optimal and high-search-value actions.

**E3.2: Action-value target repair**

- Audit sign, perspective, centering, unvisited-action treatment, and selected
  action alignment end to end in both Python and Rust.
- Train three matched models: policy/value only, the current QAdv objective,
  and QAdv with pairwise ranking loss plus proof-labelled tactical actions.
- Weight placement and relocation examples separately or use phase-specific
  heads; relocation currently dominates the ranking failure.

**Stop rule:** retire direct Q-max play if it cannot beat Lunatic after target
repair. The Q head may still be useful for move ordering or uncertainty.

**E3.3: Auxiliary strategic targets**

- Add inexpensive targets already derivable from rules/search: connection
  distance, capture count, ownership after the move, threat count, legality,
  game phase, and eventual move count/outcome.
- Ablate each target family and keep only those that improve paired strength or
  search efficiency.

### Phase 4 — architecture and efficiency ablations (2–4 weeks)

**E4.1: Phase-aware action encoder**

- Compare the existing full GNN and CNN with an action-centric model that
  encodes `(state, action, afterstate)` and separate placement/relocation heads.
- Match parameter count and training examples so architecture, not scale, is
  the changed variable.

**E4.2: Uncertainty-gated compute**

- Calibrate policy/value uncertainty using ensemble disagreement or a compact
  uncertainty head.
- Spend additional proof/search nodes only on low-margin, high-uncertainty
  positions; use the fast path for obvious moves.

**Success:** equal or better match strength with lower median move time and no
  tactical regression.

**E4.3: Champion distillation for the browser**

- Distill the strongest search agent into the smallest network that retains
  top-k move recall, then add a small fixed search budget in WASM.
- Treat native strength and browser strength as separate promotion gates.

## Recommended first experiment packet

The first implementation cycle should contain only four changes:

1. Build the expanded solver-labelled tactical suite from mined positions.
2. Add a selective, neural-ordered bounded proof extension to the Rust search.
3. Run the fixed-budget root/tree/proof/2x-node comparison on identical
   openings; add graph reuse when E1.2 lands.
4. Produce a paired report against Pathfinder and the best frozen neural model.

This packet answers the most valuable immediate question: **does smarter use
of the current evaluator beat spending more nodes?** If yes, use the improved
search to generate Phase 2 and Phase 3 training targets. If no, the loss
taxonomy will show whether to prioritize Gumbel root allocation or target
repair next.

## Experiment matrix

| ID | Primary variable | Fixed controls | Main metric | Advance when |
| --- | --- | --- | --- | --- |
| E0.1 | None; baseline | seeds, openings, budgets | reproducibility | scores and latency reproduce |
| E1.1 | proof horizon/trigger | model, root budget | tactics + paired score | ≥95% tactics, positive score delta |
| E1.2 | tree vs graph | model, inference calls | win rate per inference | stronger at equal calls |
| E1.3 | PUCT vs Gumbel | model, simulations | low-budget paired score | positive at two adjacent budgets |
| E2.1 | data generator mix | training recipe | strength / 1k games | beats uniform generation |
| E2.2 | priority fraction | data count, optimizer | held-out + stress score | gains both; no suite overfit |
| E3.2 | QAdv target/loss | trunk, data, steps | ranking + match score | beats policy/value and Lunatic |
| E4.1 | action encoder | params, data, steps | score / inference ms | Pareto improvement |

## Research basis

- AlphaZero established policy/value learning from neural-guided self-play
  search for perfect-information games:
  https://arxiv.org/abs/1712.01815
- Gumbel AlphaZero replaces heuristic root exploration with sampling without
  replacement and sequential halving, and is specifically aimed at stronger
  policy improvement under small simulation budgets:
  https://openreview.net/forum?id=bERaNdoegnO
- KataGo reports large self-play efficiency gains from improved targets,
  architecture, and training techniques rather than scale alone:
  https://arxiv.org/abs/1902.10565
- Monte Carlo Graph Search shares information across transpositions and reports
  strength and memory improvements in AlphaZero-style search:
  https://arxiv.org/abs/2012.11045
- Proof-Number MCTS combines tactical proof information with MCTS and reports
  improvements across deterministic board games, including Lines of Action:
  https://arxiv.org/abs/2303.09449
- Prioritized Experience Replay provides the basis for replaying informative
  errors more often, with bias correction:
  https://arxiv.org/abs/1511.05952
- Prioritized Level Replay supports a related curriculum principle: revisit
  examples with high estimated learning potential:
  https://arxiv.org/abs/2010.03934

## Cloud execution note

Before starting a new distributed campaign, confirm the project's selected
Region in AWS Settings > View all projects > Overview > Additional Info >
Region (or in `~/.aws/config`). All Regional compute must remain there. Also
confirm the project plan and spend-limit status before launching, declare an
experiment cost cap, and decide after each campaign whether to retain or clean
up the created resources.
