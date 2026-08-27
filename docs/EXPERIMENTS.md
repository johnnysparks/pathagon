# Experiment records and artifact retention

The repository should preserve the through-line of how Pathagon agents were
discovered without turning Git into storage for every training byproduct. An
experiment record is durable research evidence; a run directory is temporary
workspace unless its contents are explicitly promoted.

## Storage boundary

| Artifact | Durable location | Rule |
| --- | --- | --- |
| Rules, configurations, move histories, outcomes | `research/corpora/games-v1/` | Keep in Git, content-addressed by game |
| Universal training labels or annotations | Versioned sidecar under `research/corpora/` | Keep in Git when normalized, useful, deterministic, and reasonably sized |
| Experiment hypothesis, lineage, protocol, result, and decision | `research/experiments/<experiment-id>/` | Always keep in Git for serious experiments, including failures |
| Notable agent/model identity | Experiment manifest; optionally `research/agents/` when reused | Keep IDs, parentage, configuration, and hashes in Git |
| Large replay exports already represented by game keys | External store or local ignored archive | Do not duplicate in Git |
| Checkpoints, optimizer state, tensors, traces, activations, and debug dumps | Experiment-specific external store | Keep only when needed to reproduce, resume, audit, or promote |
| Small promotion checkpoint or deployable model | Git only when explicitly justified by a report | Record both source-model and artifact hashes |

Git suitability is not only a file-size test. Data promoted into Git must have
stable semantics, deterministic ordering, a versioned schema, and meaningful
diffs. Derived board states, captures, legal actions, and other facts that Rust
can reconstruct from a canonical game should not be stored again.

## Experiment identity and layout

Use an ID of the form `YYYYMMDD-short-slug`, with a suffix when multiple
experiments share a slug. A serious experiment lives at:

```text
research/experiments/YYYYMMDD-short-slug/
  README.md
  manifest.json
  games.tsv              # optional exact game-key list
```

Use the template in
[`research/experiments/TEMPLATE.md`](../research/experiments/TEMPLATE.md).
`README.md` is the reviewable narrative. `manifest.json` is the machine-readable
index used by future tooling. `games.tsv` contains references, not copied moves.

Every manifest must record:

- schema version, experiment ID, status, and start/finish dates;
- hypothesis and the one primary variable being changed;
- Git commit and exact commands or runner version;
- parent experiment, parent agent, or baseline from which the candidate derives;
- candidate and opponent agent IDs, model/checkpoint hashes, and full search/runtime configuration;
- canonical dataset or game-corpus version, split/seed policy, and exclusions;
- exact game membership using game keys, observation source IDs, or a deterministic selector with its result hash;
- metrics by opponent and color, resource/cost summary when material, and the final decision;
- external artifacts with stable location, SHA-256, byte size, purpose, and retention requirement.

Do not use mutable labels such as `latest`, filenames alone, or an agent display
name as model identity. A model hash may be absent for a non-model agent, but a
learned agent must identify its exact weights.

## Agent, opponent, and model relationships

Canonical games are never copied into directories by agent pair. The same game
can support many later analyses, and pair-based copies would silently change
training weight. Agent/opponent/model membership belongs in:

1. the canonical observation keyed by game;
2. the experiment manifest's roster and configuration; and
3. the experiment's `games.tsv` or observation-source selector.

When a candidate becomes a reusable reference agent, record its stable agent ID,
model hash, creating experiment, parent agent/model, rules version, search
configuration, and lifecycle status (`candidate`, `historical`, `champion`, or
`retired`). Failed intermediate checkpoints do not each need an agent record.

## Universal training sidecars

Training data may live in Git when its meaning is independent of a particular
Python/Rust implementation or model architecture. Examples include solver
labels, normalized policy distributions over canonical legal actions, root
action values with a declared perspective and search budget, consented human
annotations, and curated failure categories.

Such data must:

- use the canonical game key plus ply/action as its foreign key;
- define value perspective, action ordering, units, bounds, and producer;
- include producer model/search hashes and provenance without making them part
  of the game identity;
- be deduplicated, deterministically sorted, versioned, and sharded;
- have train/evaluation eligibility explicitly labeled; and
- move to external storage when its Git cost outweighs its durable reuse value.

Model activations, optimizer moments, architecture-shaped tensors, temporary
augmentation output, and repeated materialized train/held-out copies are not
universal sidecars.

## Failures and inconclusive attempts

Failure is a result, not a reason to erase the experiment. Retain enough in Git
to avoid repeating the same path:

- what was attempted and why;
- the parent/baseline and exact changed variable;
- where it failed (`setup`, `generation`, `training`, `evaluation`, `strength`,
  `latency`, `cost`, or `integration`);
- completed canonical game keys, if any;
- the decisive metrics, error summary, or representative counterexample;
- what was learned and the next decision; and
- hashes/locations of any external artifacts that remain important.

It is normally safe to discard failed checkpoints, temporary datasets, verbose
logs, and partial optimizer state after this record exists. Preserve a large
failed artifact externally only when it is needed to reproduce a surprising
result, resume expensive work, investigate correctness, or support a later
experiment.

## External artifact references

An external reference must use a stable object key or dataset URI, never a
temporary signed URL. Record:

- storage provider and stable URI/key;
- SHA-256 and byte size;
- content type and compression;
- producing experiment and role;
- whether it is required for audit, reproduction, resumption, or deployment;
- retention/expiration policy and restoration instructions; and
- access classification, without credentials or secrets.

Before removing a local large artifact, verify its hash at the external
location or confirm that the canonical corpus and experiment record make the
artifact unnecessary.

## Completion checklist

An experiment is not complete until its status and decision are recorded, all
completed games are linked into the canonical corpus, important universal
labels are promoted or intentionally externalized, and disposable intermediates
are identified. Promotion additionally requires the gates in
[`LEARNING.md`](LEARNING.md).
