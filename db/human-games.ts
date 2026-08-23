import type { HumanGameSubmission } from "../app/game-record";

const tableSql = `CREATE TABLE IF NOT EXISTS human_games (
  id TEXT PRIMARY KEY,
  schema_version INTEGER NOT NULL,
  recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  opponent_id TEXT NOT NULL,
  winner TEXT NOT NULL,
  plies INTEGER NOT NULL,
  actions TEXT NOT NULL,
  compact TEXT NOT NULL,
  validation TEXT NOT NULL DEFAULT 'replay-valid',
  source TEXT NOT NULL DEFAULT 'web-human-v1'
)`;
const recordedIndexSql = "CREATE INDEX IF NOT EXISTS human_games_recorded_at_idx ON human_games(recorded_at)";
const opponentIndexSql = "CREATE INDEX IF NOT EXISTS human_games_opponent_idx ON human_games(opponent_id)";

async function database() {
  const { env } = await import("cloudflare:workers");
  const d1 = env.DB;
  if (!d1) throw new Error("Human game database is unavailable");
  await d1.batch([
    d1.prepare(tableSql),
    d1.prepare(recordedIndexSql),
    d1.prepare(opponentIndexSql),
  ]);
  return d1;
}

export async function storeHumanGame(id: string, game: HumanGameSubmission, compact: string) {
  const d1 = await database();
  return d1.prepare(`INSERT OR IGNORE INTO human_games
    (id, schema_version, opponent_id, winner, plies, actions, compact)
    VALUES (?, 1, ?, ?, ?, ?, ?)`)
    .bind(id, game.opponentId, game.winner, game.actions.length, JSON.stringify(game.actions), compact)
    .run();
}
