# 20260828 Pathfinder boundary-aware evaluator evolution

Status: completed

## Idea

Re-evolve the tactical-safe Pathfinder evaluator with a longer three-generation
search while holding the promoted runtime envelope fixed. The goal is to test
whether the v0.5 weights were a short-run local improvement or whether another
generation can improve the evaluator without spending more search budget.

## Starting point

The promoted control is `pathfinder-v0.4.0-tactical-filter`; the current trained
opponent is `pathfinder-v0.5.0-trained-evaluator`. Both use depth 4, a 2,000-node
budget, beam width 8, paired colors, and randomized openings. The v0.5 weights
are path 241, material 112, capture 887, structure 40, threat 154, and edge
74. A current 120-game reproduction scored 70 wins, 48 losses, and 2 draws
for v0.5. The earlier 70–47–3 record used the same protocol before commit
`42e89299` corrected alpha-beta sentinel bounds; that correction is retained.

This run starts from the deterministic handcrafted evaluator used by the
existing training runner, with tactical-safe filtering enabled. It uses three
generations, population 8, four paired training openings, twelve paired held-out
openings per generation, seed 20260828, 120-ply games, and two randomized
opening plies. Generated corpora and reports stay in this path's ignored
`workspace/`.

## What happened

The three-generation run completed in 194.65 seconds with 24 trials, 192
training games, 72 held-out games, 6,311 training positions, and 2,551
evaluation positions. No candidate was promoted. The best held-out candidates
were `rust-evo-g2-c0-287-127-735-46-140-73` and
`rust-evo-g3-c1-259-96-832-50-142-88`, each scoring 13 wins, 11 losses, and 0
draws (541/1000 game points) in 24 games. The promotion threshold was 550/1000,
so neither result justified replacing the current evaluator.

## Data and artifacts

The runner writes `champion.json`, `report.json`, and training/evaluation replay
corpora under `workspace/`. These are disposable experiment outputs and are not
promoted automatically. No new durable model or fixture is created by this
path yet.

## Project impact

No runtime behavior was promoted. The result is useful as negative evidence:
longer evolution from the handcrafted seed did not reproduce the v0.5-quality
candidate under the fixed envelope, so the next improvement path should either
seed from the promoted evaluator or change the training signal rather than
simply increasing generations.

## Next decision

Keep the generated report and replay corpora as unpromoted workspace evidence.
Do not alter the supported Pathfinder identities; revisit evaluator evolution
with a promoted-v0.5 starting point or a stronger position-level objective.
