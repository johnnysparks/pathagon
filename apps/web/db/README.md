# Game archive

The archive is the durable record of completed games. Leaderboard standings
are computed from imported records; the browser does not generate official
leaderboard matches.

The `/lab` page has two deliberately separate surfaces: the official ladder and
pairwise standings include only the six opponents implemented through the Rust/
WASM engine (Transition v4, v0.5, v0.4, Surveyor, Lunatic, and Coin Flip).
Historical Python, neural, and other research identities remain queryable and
replayable in the archive, but cannot receive Elo or affect official pairwise
results until they are ported, validated, and promoted.

## Storage

Cloudflare D1 has two tables:

- `human_games` stores completed anonymous web games;
- `selfplay_games` stores validated offline Rust and Python matches. Legacy
  TypeScript records remain readable for historical archive compatibility.

Human-game rows also carry a bounded JSON `metadata` blob. The browser uses it
for exploratory Pathfinder search traces (dial settings, model card, elapsed
time, positions searched, checkpoints, and whether the current-best button
interrupted the search). It is intentionally separate from the compact replay
contract so older games remain readable.

The complete contract-v1 replay is retained with searchable metadata: engine,
mode, run ID, agents, result, termination reason, seed, and ply count. Git is
for small curated corpora, manifests, selected checkpoints, and reports—not a
log store.

Archive validation has two layers. Contract validation checks the shape and
bounds of a record; self-play validation additionally replays every action and
checks captures, the winner, policy/Q-target alignment, and provable terminal
conditions. A `max-plies` record must end exactly at `config.maxPlies`. A
`threefold-repetition` reason is retained as producer metadata because the
current contract does not include the full position-history sequence needed to
prove it independently.

The checked-in Drizzle migrations are the schema authority. Request handlers do
not create or alter tables at request time.

## Import offline self-play

Run an offline match batch, then import the completed records:

```bash
./scripts/run-rust-archive.sh macbook-lunatic-001 1000 20260824 lunatic

npm run selfplay:archive -- \
  --file work/selfplay/macbook-lunatic-001.jsonl \
  --url https://your-pathagon-domain.example \
  --engine rust
```

For a protected deployment, provide its bearer token through
`PATHAGON_ARCHIVE_TOKEN`. Do not commit the token or place it in a README.

The uploader normalizes records to contract v1, replays them against the
configured reference rules, and derives stable IDs from engine, run, seed, and
position. Reusing an archive is therefore idempotent.

## Query records

```text
GET /api/selfplay?engine=rust&agent=rust-pathfinder-v0.1.0&result=win&limit=100
GET /api/selfplay?mode=arena&format=jsonl
GET /api/selfplay/<archive-id>
GET /api/games/<game-id>
```

Self-play records support metadata filters and JSONL export. Human game IDs are
bearer tokens; there is intentionally no listing endpoint for them.

## Training use

Archive exports may be used to build datasets after validation, deduplication,
and a documented train/held-out split. Human games remain a separate source
until their privacy and consent policy permits training use.
