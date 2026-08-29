# Pathagon web

This workspace contains the browser game, leaderboard lab, API routes, D1
adapter and migrations, Worker entry point, generated WASM/model assets, and
web-specific tests.

Run it from the repository root with `npm run dev`, or from this directory
with `npm run dev` after installing the root workspace.

## Deployment status

The app uses standard Vite, Vinext, and Cloudflare Worker tooling. GPT Sites
packaging and its unused authentication helper have been removed. Local
development creates a placeholder D1 binding. Pushes to `master` verify the web
app and Rust engine, then deploy to a personal Cloudflare Worker through GitHub
Actions. The first successful run creates the `pathagon-web` D1 database; later
runs reuse it and apply checked-in migrations before deployment.

The repository requires `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`
GitHub Actions secrets. The token needs permission to edit Workers and D1. Set
the optional `NEXT_PUBLIC_APP_URL` GitHub Actions variable after attaching a
stable custom domain if absolute social preview URLs matter.

The web property owns everything here, including `app/`, `db/`, `drizzle/`,
`public/`, `worker/`, and its build configuration. Stable game rules and
interchange formats live under [`../../pathagon/`](../../pathagon/).
