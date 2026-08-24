# Pathagon

Digital preservation of Mark Fuchs's two-player wooden strategy game, with a mobile web client, deterministic rules engine, AI opponents, and a reproducible self-play laboratory.

The repository now contains two production rules implementations: TypeScript is the browser reference engine, while the small Rust bitboard engine runs high-volume headless search and self-play. Shared parity fixtures keep their move legality, captures, and win resolution aligned.

```bash
npm test
npm run selfplay -- --mode arena --games 20 --seed 20260822
npm run selfplay:league -- --games 8 --seed 20260823
cargo test --manifest-path engine-rs/Cargo.toml --release
npm run rust:train -- --generations 3 --population 6 --training-pairs 6 --evaluation-pairs 12
```

See [`docs/SELF_PLAY.md`](docs/SELF_PLAY.md) for experiment structure and [`docs/RUST_ENGINE.md`](docs/RUST_ENGINE.md) for the native engine contract.

The experimental 7x7 GNN/CNN PUCT learners live under
[`learning/gnn/`](learning/gnn/). The canonical target is the historical 7x7
board with 14 reserves per player: the scale-compatible GNN remains useful
for regression, while the new compact CNN is intentionally fixed to 7x7.
Smaller-board data is retained as curriculum and test material, not mixed into
the canonical 7x7 training distribution. These learners are not yet the
browser opponent. See [`docs/LEARNING_TOURNAMENTS.md`](docs/LEARNING_TOURNAMENTS.md)
for the clone-on-another-Mac, generate, merge, and retrain workflow.

The playable opponent ladder mixes search and heuristic baselines: Coin Flip is random, Lunatic is a deliberately naive one-ply pattern heuristic, The Surveyor searches two plies, and The Pathfinder uses iterative deepening up to four plies. "Expert" currently describes its search budget, not a solved-game or unbeatable claim.

TypeScript promotion training is retired; Rust owns evaluator-weight promotion,
while Python owns checkpoint/GNN league experiments. The TypeScript runner still
supports browser-reference arenas and historical league comparisons.

## Anonymous human game archive

After the first move, the web client creates a random game ID and displays it
as selectable, copyable text. When the game finishes, the replay is validated
and stored in D1 under that ID; no account or player identity is required.
Anyone with the token can retrieve the move stream for study with
`GET /api/games/<game-id>`. There is no listing endpoint, so the token is the
only lookup key.

## Self-play game archive

Completed local TypeScript and Rust self-play records can be uploaded to the
same private D1 database for later filtering and analysis. The database keeps
searchable run metadata alongside each replay; Git remains reserved for small,
curated corpora. See [`docs/GAME_ARCHIVE.md`](docs/GAME_ARCHIVE.md) for the
uploader and JSONL query examples.

## Runtime foundation

A clean full-stack starter running on
[vinext](https://github.com/cloudflare/vinext), with optional Cloudflare D1 and
Drizzle support.

## Prerequisites

- Node.js `>=22.13.0`
- Rust `1.98.0` via rustup for the native tournament engine
- Python 3 with a virtual environment for the GNN learner
- Linux with `flock`, `curl`, and GNU `timeout`

Run the Python learner's complete regression suite with the project virtual
environment (including the cross-runtime contract tests):

```bash
./scripts/test-python.sh
```

The versioned cross-runtime interchange contract is documented in
[`contracts/README.md`](contracts/README.md).

Every new replay agent specification includes a manifest-backed identity:
runtime, rules version, evaluator weights, depth, node budget, beam, and model
hash. The JSON schema and native validators are exercised by all three runtime
test suites.

## Sites Lifecycle

The Sites lifecycle CLI runs the locked dependency install before returning this checkout. Edit the source under `app/`, then checkpoint when a coherent milestone is ready to inspect or share. The remote Sites builder runs `npm run build` against the pushed commit. Do not repeat install or build as a normal pre-checkpoint step.

This starter does not use `wrangler.jsonc`.

`install:ci` is intentionally a single, non-retrying `npm ci`. It refuses a concurrent install for the same project, consumes a matching image-seeded npm cache with `--prefer-offline` while retaining registry fallback for a missing cache object, otherwise downloads and verifies the complete vinext tarball recorded in `package-lock.json`, limits npm to one socket, and terminates a stalled install. `build` applies a short timeout. These helpers target Linux and use GNU `timeout`; they are not native macOS scripts.

Scripts that need writable project-scoped home, npm, XDG, and temporary paths use `scripts/sites-env.sh`. The `dev` and `start` scripts honor the caller's runtime environment and keep Wrangler logs inside the checkout. The generated `.sites-runtime/` directory is disposable and ignored by Git.

## Included Shape

- edit site code under `app/`
- `app/chatgpt-auth.ts` provides optional dispatch-owned ChatGPT sign-in helpers
- `.openai/hosting.json` declares optional Sites D1 and R2 bindings
- `vite.config.ts` simulates declared bindings for local development
- `db/index.ts` reads the D1 binding from the Cloudflare Worker environment
- `db/schema.ts` starts intentionally empty
- `examples/d1/` contains an optional D1 example surface
- `drizzle.config.ts` supports local migration generation when needed

## Workspace Auth Headers

OpenAI workspace sites can read the current user's email from
`oai-authenticated-user-email`.

SIWC-authenticated workspace sites may also receive
`oai-authenticated-user-full-name` when the user's SIWC profile has a non-empty
`name` claim. The full-name value is percent-encoded UTF-8 and is accompanied by
`oai-authenticated-user-full-name-encoding: percent-encoded-utf-8`.

Treat the full name as optional and fall back to email when it is absent:

```tsx
import { headers } from "next/headers";

export default async function Home() {
  const requestHeaders = await headers();
  const email = requestHeaders.get("oai-authenticated-user-email");
  const encodedFullName = requestHeaders.get("oai-authenticated-user-full-name");
  const fullName =
    encodedFullName &&
    requestHeaders.get("oai-authenticated-user-full-name-encoding") ===
      "percent-encoded-utf-8"
      ? decodeURIComponent(encodedFullName)
      : null;

  const displayName = fullName ?? email;
  // ...
}
```

## Optional Dispatch-Owned ChatGPT Sign-In

Import the ready-to-use helpers from `app/chatgpt-auth.ts` when the site needs
optional or required ChatGPT sign-in:

- Use `getChatGPTUser()` for optional signed-in UI.
- Use `requireChatGPTUser(returnTo)` for server-rendered pages that should send
  anonymous visitors through Sign in with ChatGPT.
- Use `chatGPTSignInPath(returnTo)` and `chatGPTSignOutPath(returnTo)` for
  browser links or actions.
- Pass a same-origin relative `returnTo` path for the destination after sign-in
  or sign-out. The helper validates and safely encodes it.
- Mark protected pages with `export const dynamic = "force-dynamic"` because
  they depend on per-request identity headers.

Dispatch owns `/signin-with-chatgpt`, `/signout-with-chatgpt`, `/callback`, the
OAuth cookies, and identity header injection. Do not implement app routes for
those reserved paths. Routes that do not import and call the helper remain
anonymous-compatible.

SIWC establishes identity only; it does not prove workspace membership. Use the
Sites hosting platform's access policy controls for workspace-wide restrictions,
or enforce explicit server-side membership or allowlist checks.

Use SIWC for account pages, user-specific dashboards, saved records, and write
actions tied to the current ChatGPT user. Leave public content anonymous.

## Diagnostic Commands

- `npm run install:ci`: perform the one bounded lockfile install
- `npm run dev`: start the Vite/Vinext development server
- `npm run build`: build the deployable Sites artifact
- `npm run start`: start the built Vinext application
- `npm test`: build and verify the rendered development-preview metadata
- `npm run db:generate`: generate Drizzle migrations after schema changes

Use build commands for targeted diagnosis after a remote failure, not as part of the normal checkpoint path.

The timeout defaults can be overridden for a controlled canary with `SITES_INSTALL_TIMEOUT`, `SITES_INSTALL_KILL_AFTER`, `SITES_BUILD_TIMEOUT`, and `SITES_BUILD_KILL_AFTER`. A timeout fails the command; the helpers never retry an unchanged install or build.

## Learn More

- [vinext Documentation](https://github.com/cloudflare/vinext)
- [Drizzle D1 Guide](https://orm.drizzle.team/docs/get-started/d1-new)
