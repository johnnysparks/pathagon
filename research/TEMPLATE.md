# YYYYMMDD Research path title

Status: idea | running | completed | inconclusive | abandoned

## Idea

What might improve, what is changing, and why this path is worth trying.

## Starting point

Describe the relevant current engine/opponent/data state and any parent path.
Exact hashes and commands are welcome when useful, but this is a historical
explanation rather than a schema-compliance exercise.

## What happened

Describe the work, important protocol choices, outcome, failed attempts, and
the evidence that drove the decision.

## Data and artifacts

Explain what was generated, what remains in Git, what is ignored in
`workspace/`, and what was discarded. Reference promoted data under `data/`.

## Project impact

List what was promoted into `pathagon/`, `apps/`, or `data/`, what it attempted
to move forward, where it succeeded or failed, and what was left behind.

## *Hiccups* - Most important section for dead-end research

What are some potential issues, things we overlooked, bugs or curiosities. What
prevented from this run going perfectly? 

*WHY* did the hypothesis fail, and what did we learn from the failure?

## Next decision

State whether to continue, revisit, promote, or retire the path.

## Decision gates

For learned-opponent work, record the frozen search envelope and control,
source-disjoint train/held-out split, teacher quality at that envelope, held-out
action ranking, state-value calibration, paired whole-game strength, legality
audit, and completed-depth/node/latency cost. Treat a model as advisory inside
native search unless this evidence supports promotion. Historical negative
results remain useful when they identify which gate failed and what data or
integration change the next path will test.
