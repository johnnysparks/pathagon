# Applications

Each child is an independently deployable product. An app owns its UI, server
routes, database adapter and migrations, static assets, deployment
configuration, and app-level tests. Shared game behavior moves into
`pathagon/` before another app depends on it.

- [`web/`](web/) — browser gameplay, game archive APIs, and leaderboard lab.
