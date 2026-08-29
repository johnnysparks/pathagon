# 20260828 Sol versus deep Pathfinder

Status: `completed — split 1–1`

## Idea

Play genuine games in which GPT-5.6 Sol selects every Light move from visible
board state, while the production Rust engine enforces the rules and unmodified
deep Pathfinder selects every Dark move. The first game used medium reasoning;
the rematch used the user's `Extra High` selection. These are not scripted
distillations of model preferences.

## Starting point

The game uses the standard 7×7 board, 14 reserves per player, and a 160-ply
cap. Sol is Light and connects bottom to top. Pathfinder is Dark and connects
left to right using depth 5, beam 8, and a 2,000-node ceiling—the documented
deep-search control.

The research-only Rust referee replays the complete action sequence through
`pathagon-engine`. On Sol turns it emits the board and a bounded shallow
analysis shortlist as advisory context; the language model may select any
legal move. On Pathfinder turns it emits the deep search result. Thus Rust is
the state/rules/search authority, while Sol's decisions are made in chat.

## What happened

Sol selected every Light move in chat from the referee's current position and
bounded shortlist. No scripted Sol policy selected or ranked the final moves.

### Game 1 — Medium reasoning

Pathfinder won as Dark on ply 26 by completing a left-to-right connection.

The complete game was:

```text
P45,P10,P9,P7,P2,P13,P16,P8,P11,P9,P12,P4,P5,P3,P6,P18,
P19,P0,P17,P1,P18,P2,P24,P11,P31,P12
```

The final board was:

```text
DDDDDLL
DDDDDDD
..LLLL.
...L...
...L...
.......
...L...
```

Pathfinder's last `P12` filled the only gap in its second-row chain. Sol could
not legally contest that square after Pathfinder's preceding `P11` captured
the Light stone on `P12`, because the immediate recapture square was forbidden.
Sol's `P31` created a one-move-away vertical threat through `P38`, but Dark's
win was immediate.

The most important strategic pattern was Pathfinder's capture ladder across
the upper two rows. Sol's early `P9` flank was captured by `P8`, and later
attempts to block the top route committed several moves without dismantling
Dark's connected structure. Sol deliberately overrode the shallow advisory
shortlist at `P6`; the referee accepted it as legal and continued normally.
Pathfinder began returning the `POS_INF` search sentinel during the middle
game. The second game proved that this value is not reliable evidence of a
forced win, so this run does not claim to locate the earliest lost position.

### Game 2 — user-selected Extra High reasoning

Sol won as Light on ply 55. The complete game was:

```text
P44,P9,P10,P2,P3,P8,P17,P11,P7,P10,P13,P6,P20,P5,P1,P15,
P14,P0,P16,P2,P4,P18,P12,P1,P19,P24,P18,P17,P11,P33,P34,
P27,P32,P15,P41,P16,R34>25,P21,R44>23,P31,R41>30,P26,
R23>33,P24,R30>34,P40,R33>23,P26,R13>27,P24,R23>39,
R21>26,P41,P46,R7>48
```

The final board was:

```text
DDDLLDD
.DDDLL.
LDDDLLL
...D.DL
....L.L
....L.L
....D.L
```

The strategic improvement was not simply deeper calculation. Sol established
durable boundary blockers at `P7` and `P13`, turned defensive stones into two
vertical lanes, and repeatedly used relocation destinations to trigger local
captures. The decisive sequence created connected descents through `P39` and
`P41`. Pathfinder used its returned reserve to block the first bottom endpoint
at `P46`; `R7>48` completed the surviving right-hand lane.

The game also exposed a serious reasoning failure. At `P12`, Sol incorrectly
assumed that a newly placed stone would capture the entire bracketed chain
`D8–D11`. The engine correctly left the chain in place. Pathagon capture rays
only remove the immediately adjacent opponent stone when the square two steps
away in that direction is friendly; a move can do this independently in up to
four directions. Captured stones return to reserve, relocations can trigger the
same capture rule, and captured squares are forbidden only for the immediate
reply. Once corrected, explicit two-square capture accounting was the main
reason Sol recovered.

Pathfinder returned its exact internal `POS_INF` value (`536870911`) several
times in this game, including completed depth-5 searches, but later evaluations
reversed and Pathfinder lost. Terminal win scores use a different scale near
`1_000_000_000`. Therefore `POS_INF` is an exposed alpha-beta bound/sentinel,
not a proven-game result. Treating it as “forced win” in the first-game notes
was incorrect.

Board rows are printed top to bottom. `L` is Sol, `D` is Pathfinder, `*` is a
temporarily forbidden capture square, and `.` is empty. Squares are row-major
from 0 at top-left through 48 at bottom-right.

## Data and artifacts

Compiler output and scratch transcripts are disposable. Both final move lists
are recorded above as the durable evidence. Bulk alternative analyses or
repeated replays belong in ignored `workspace/`.

The referee lives under [`rust/`](rust/). It is intentionally research-only:
it replays an action sequence through the production engine, prints a bounded
shallow shortlist on Sol turns, and invokes the unmodified deep Pathfinder
configuration on Dark turns. Generated compiler artifacts are ignored.

## Project impact

The run validates a useful evaluation procedure: Rust owns state, legality,
and the established opponent while the model under test owns each of its own
decisions. The split result does **not** establish that Pathfinder or either
reasoning setting is generally stronger. Opening `P44` instead of `P45` also
changed, and two games have no statistical power.

There are three durable project takeaways:

1. Any model or human interface should show the post-move board, captured
   squares, forbidden squares, reserves, and winner explicitly. The referee's
   Pathfinder response initially printed only the pre-move board; its new
   `postMove` payload fixes that research interface, while production consumers
   still need the same audit.
2. The public search result must not present `POS_INF` as an ordinary score.
   Its leakage from apparently completed searches deserves an engine test and
   audit before search scores are used as proof labels or game annotations.
3. Strong play depends heavily on relocation-triggered captures and parallel
   connection lanes. Evaluations and future opponent documentation should
   include representative reserve-to-relocation endgames, not only placement
   openings.

No engine, opponent, dataset, or fixture is promoted. Two games are useful
narrative evidence but are not enough to characterize either player, and the
research referee does not meet supported production-tooling standards.

## Follow-up action items

### P0 — Audit the Pathfinder score contract

- [x] Reproduce the public `POS_INF` result from the recorded game prefixes and
  isolate the root cause: the internal bounds were numerically inside the
  terminal score range.
- [x] Add production Rust regression tests proving that a completed,
  nonterminal search cannot return `POS_INF` or `NEG_INF` as an ordinary score.
- [ ] Make public search output distinguish exact evaluations, lower/upper
  bounds, budget-exhausted estimates, and terminal results. Never expose an
  internal sentinel as a proven-game evaluation.
- [ ] Recheck every consumer that writes search scores into game records,
  labels, leaderboards, or training data before treating those scores as truth.

Done when the recorded prefixes return a documented score kind, terminal wins
remain on the `WIN_SCORE` scale, and strict engine tests prevent sentinel
leakage.

Progress: the bound-range bug is fixed and covered by a production Rust
regression test. This is a post-hoc engine repair; the two archived move lists
remain historical records from the pre-fix referee. Score-kind metadata and
downstream consumer audits remain open.

### P0 — Make move transitions explicit

- [x] Return the post-move board, action, captured squares, forbidden squares,
  reserves, next player, and winner together for every engine-driven move.
- [x] Update the research referee before any reuse; audit production gameplay,
  self-play, and web adapters for the same pre-move/post-move ambiguity.
- [x] Add contract tests covering an ordinary placement, a placement capture,
  a relocation capture, a multi-direction capture, and a terminal move.

Done when a consumer never has to replay or infer a move to know its exact
result.

Progress: the Rust runtime source and research referee now expose complete
post-move transitions, with production tests covering each move shape. The
checked-in generated browser bundle still needs a normal engine rebuild before
web callers use the new export. `npm run build:engine` was attempted, but this
checkout does not have the required `wasm-bindgen` 0.2.127 CLI installed; no
generated bundle files were changed.

### P1 — Tighten capture and relocation documentation

- [ ] Document the exact capture ray: the destination captures only an
  immediately adjacent opponent stone supported by a friendly stone two
  squares away, independently in each orthogonal direction.
- [ ] Document that captured stones return to reserve, captured destinations
  are forbidden for the immediate reply, and relocations trigger captures at
  their destination.
- [ ] Put a compact diagram and representative examples beside the stable Rust
  engine and in the canonical game-rules documentation.
- [ ] Audit production tests for the mistaken multi-stone bracket assumption
  made during this game; add negative tests proving longer bracketed chains are
  not swept.

Done when the rules, engine tests, and user-facing explanations describe the
same capture behavior without relying on inference from gameplay.

### P1 — Add relocation-endgame coverage

- [ ] Extract only the smallest durable positions needed to cover
  relocation-triggered captures, returned-reserve transitions, parallel
  connection lanes, and forced endpoint blocks.
- [ ] Promote those positions into a strict, versioned fixture location under
  `data/`; do not promote the research transcript wholesale.
- [ ] Add scrutinized Rust tests or benchmarks for the dual-lane sequence that
  ended with Pathfinder blocking `P46` and Light winning through `P48`.
- [ ] Verify Pathfinder considers both connection lanes under its documented
  depth, beam, and node limits.

Done when reserve-to-relocation play is represented in production tests and
the fixtures have an explicit schema version and expected outcome.

### P2 — Define any future model-play protocol before running it

- [ ] Predeclare colors, openings, game count, model/reasoning setting,
  Pathfinder configuration, shortlist size, and stopping rules.
- [ ] Record whether the model has access to earlier games or rules feedback;
  learned rematches must not be compared as independent samples.
- [ ] Use color-balanced games and fixed openings before making opponent-strength
  claims. Report game-level results separately from statistical conclusions.
- [ ] Capture the actual runtime model metadata when available rather than
  relying solely on a user-selected UI label.

Done when another run could be interpreted without post-hoc assumptions. This
item does not recommend repeating the informal 1–1 test.

## Next decision

Do not repeat this informal test merely to settle the 1–1 score. If the method
is used again, first fix the transition display and sentinel-score contract,
then run a predeclared, color-balanced set. Only promote game records under
`data/` if they become part of a stable, versioned evaluation corpus.
