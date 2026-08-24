# Game archive

The archive is the durable record of completed games. Leaderboard standings
are computed from imported records; the browser does not generate official
leaderboard matches.

## Storage

Cloudflare D1 has two tables:

- `human_games` stores completed anonymous web games;
- `selfplay_games` stores validated offline TypeScript, Rust, and Python
  matches.

The complete contract-v1 replay is retained with searchable metadata: engine,
mode, run ID, agents, result, termination reason, seed, and ply count. Git is
for small curated corpora, manifests, selected checkpoints, and reports—not a
log store.

The checked-in Drizzle migrations are the schema authority. Request handlers do
not create or alter tables at request time.

## Import offline self-play

Run an offline match batch, then import the completed records:

```bash
./scripts/run-rust-archive.sh macbook-lunatic-001 1000 20260824 lunatic

npm run selfplay:archive -- \
  --file training/gnn/league/macbook-lunatic-001.jsonl \
  --url https://pathagon-game.sparks-house-6466.chatgpt.site \
  --engine rust
```

For an owner-only Site, provide the bearer token through
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
