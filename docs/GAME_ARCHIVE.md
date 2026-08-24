# Game archive

The canonical archive is the Cloudflare D1 database already bound to the
site. It has two separate tables:

- `human_games` stores completed anonymous web games. The game ID is the
  bearer token and there is no listing endpoint.
- `selfplay_games` stores replay-validated TypeScript and Rust matches. The
  engine, mode, run ID, agents, result, termination reason, seed, and ply count
  are indexed metadata; the complete contract-v1 replay is retained as JSON.

This keeps the archive queryable without turning the repository into a log
store. Git continues to hold only small, curated corpora and reproducibility
fixtures. Hugging Face and R2 are intentionally not part of this pipeline.

## Upload local self-play

Run a TypeScript arena and archive its completed games:

```bash
npm run selfplay -- --mode arena --games 20 --seed 20260823
npm run selfplay:archive -- \
  --file selfplay/progress/runs/arena-20260823.json \
  --url https://pathagon-game.sparks-house-6466.chatgpt.site \
  --engine typescript
```

For the current owner-only Site, provide the Sites bearer through
`PATHAGON_ARCHIVE_TOKEN`; omit it if the Site is later made public:

```bash
PATHAGON_ARCHIVE_TOKEN='…' npm run selfplay:archive -- \
  --file selfplay/progress/runs/arena-20260823.json \
  --url https://pathagon-game.sparks-house-6466.chatgpt.site
```

For Rust JSONL output, save the complete records and then use the same
uploader. The aggregate summary line is ignored automatically:

```bash
cargo run --release --manifest-path engine-rs/Cargo.toml --bin pathagon-selfplay -- \
  --games 100 --seed 20260823 --jsonl > /tmp/pathagon-rust-selfplay.jsonl
npm run selfplay:archive -- \
  --file /tmp/pathagon-rust-selfplay.jsonl \
  --url https://pathagon-game.sparks-house-6466.chatgpt.site \
  --engine rust
```

Uploads are normalized to contract v1 and replayed against the configured
reference rules before insertion. Older schema-v2 files are accepted through
the compatibility normalizer. Reusing the same generated run file is
idempotent because the uploader derives stable record IDs from the engine, run,
seed, and position in the run.

## Query records

The API supports metadata filters and a JSONL export shape:

```text
GET /api/selfplay?engine=rust&agent=rust-pathfinder-v0.1.0&result=win&limit=100
GET /api/selfplay?mode=arena&format=jsonl
GET /api/selfplay/<archive-id>
```

The JSON response returns `{ count, limit, offset, games }`. Each game includes
its archive metadata and the replay record, so a dataset-building job can
filter by agent, color, outcome, reason, seed, or run without decoding the
compact Git corpus first.
