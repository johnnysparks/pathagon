# Separated value and action-policy experiment

Status: pilot implementation. The Q/advantage path is isolated from the
production and existing policy-value models. The implementation lives in
`research/20260824-gnn-cnn-lab/python/transition.py`, the `qadv` training mode in
`research/20260824-gnn-cnn-lab/python/train.py`, and the evaluation/arena harnesses in
the historical runners in [`scripts/`](scripts/).

## Motivation

The current neural replay target assigns every position the eventual game
result from the side-to-move perspective: `+1`, `-1`, or `0`. That is a valid
policy-value target, but it is a poor description of many Pathagon positions.
Near the end of a game, both players can be only a few tactical moves from a
win. A scalar board value can reasonably remain close to neutral while one
action—capturing a bridge, blocking a path, or creating a fork—is much better
than the alternatives.

The recent re-evaluated batch confirms the symptom. After 30,000 updates, the
policy heads improved substantially, but the value heads remained close to the
zero-prediction baseline. The current data also stores MCTS policy targets but
does not store per-action root values. The heuristic root-afterstate score is
used during search but is not a supervised value target.

The experiment therefore separates:

- `V(s)`: a baseline estimate for the side to move;
- `Q(s,a)`: the expected result after choosing action `a`;
- `A(s,a) = Q(s,a) - V(s)`: how much better the action is than the position's
  average action;
- `P(a|s)`: the policy distribution used to choose and explore actions.

The key hypothesis is that action-relative supervision is more useful than
asking a board-only scalar to encode tactical urgency.

## Proposed model

Keep the shared board encoder and existing policy head. Add an action-value
head that consumes state context plus explicit transition information:

```text
context = Encoder(s)
transition = TransitionFeatures(s, a, apply(s, a))
V        = value_head(context)
A(s, a)  = advantage_head(context, action_features, transition)
Q(s, a)  = V + A(s, a) - mean_legal_actions(A)
```

The mean-centering makes the decomposition identifiable: the action head
cannot arbitrarily move value between `V` and `A`. All values remain from the
side-to-move perspective.

The transition representation should expose effects that are difficult to
infer from a static board alone:

- captured pieces, capture count, and captured bridge/frontier locations;
- own and opponent connection-distance deltas;
- newly created or removed winning-path threats;
- fork count and opponent-fork prevention;
- legal-action and mobility changes before versus after the move;
- reserve changes and relocation phase;
- connected-component and edge-contact changes;
- whether the successor is an immediate win, forced block, or repetition risk.

The first implementation should use deterministic, inspectable features. A
learned afterstate encoder can follow if the feature audit shows useful signal.

## Training targets

### Primary target: root action values

Extend self-play records with root action values aligned to the legal-action
list. MCTS already has the necessary child estimates; serialize the values in
the root player's perspective, with the same sign convention used by PUCT.

The record should retain the existing policy target and add fields equivalent
to:

```json
{
  "policy": [0.1, 0.9],
  "actionValues": [-0.2, 0.7],
  "actionVisits": [4, 12],
  "actionValueSource": "mcts-root-q-v1"
}
```

Action values from four simulations are too noisy for a serious target. The
pilot corpus used for the first checkpoint uses 128 simulations, 64 games,
and retains root visit counts so target quality can be audited. Even this
remains a shallow corpus relative to the legal action space, especially for
relocations, so its checkpoint is an evaluation instrument rather than a
strength claim.

### Secondary targets

Retain final outcome targets for `V(s)`, but supplement them with one of:

- n-step returns with terminal bootstrap;
- search-leaf values from the root tree;
- an auxiliary normalized heuristic/afterstate delta target;
- pairwise action preferences when absolute Q values are unreliable.

A raw `H(s') - H(s)` target should be treated as an auxiliary signal, not the
definition of a good move. A locally attractive capture can still allow a
forced opponent fork.

## Losses and ablations

The initial comparison should use a small set of explicit ablations:

| Variant | Policy | Value | Action target |
| --- | --- | --- | --- |
| Baseline | MCTS policy | final outcome | none |
| Transition policy | MCTS policy | final outcome | transition features only |
| Dueling Q/A | MCTS policy | final outcome | root Q and centered advantage |
| Bootstrapped Q/A | MCTS policy | n-step/search value | root Q and centered advantage |

Candidate objective:

```text
L = L_policy
  + λv * Huber(V, value_target)
  + λq * Huber(Q, root_q_target)
  + λrank * pairwise_rank_loss(A)
```

The ranking term matters because the practical question is often whether a
move is better than its alternatives, not whether its absolute value is
calibrated to a particular number. It should be evaluated both with and
without the absolute Q regression term.

## Evaluation gates

Do not judge this experiment solely by board-value MSE. Report:

1. policy NLL and selected-action top-1/top-5;
2. action-value pairwise ranking accuracy;
3. ranking accuracy by transition bucket: capture, block, fork, relocation,
   path completion, and quiet move;
4. calibration of `V` and `Q` against final outcomes;
5. successor-feature sanity checks, including path distance and mobility;
6. color-balanced stochastic arena results at multiple simulation budgets.

The current four-simulation deterministic arena is not sufficient: recent
matches were decided entirely by second-player order. Arena runs should use
root noise or another controlled stochasticity, disjoint seeds, alternating
colors, and enough games to expose whether action-relative gains transfer to
playing strength.

## Implementation sequence

1. Run a transition audit over the existing re-evaluated replay without
   retraining. Measure whether selected actions have better transition deltas
   than legal alternatives and whether captures/blocks/forks explain the
   stored MCTS policy.
2. Extend `SearchExample` and replay serialization with root action values and
   visit counts. Validate sign orientation with a small hand-built tree.
3. Add a separate action-value/advantage head to both the GNN and CNN while
   leaving the current policy head intact.
4. Train the four ablations on one fixed game-grouped split.
5. Inspect action-level examples and run the stochastic arena before deciding
   whether the value head or search pipeline should be promoted.

The existing re-evaluated batch remains a historical policy/value baseline;
it should not be relabeled as Q/A data after the fact because it lacks root
action-value targets.
