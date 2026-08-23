import { compactHumanGame, validateHumanGame } from "../app/game-record";
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

type HumanGameRow = {
  id: string;
  recorded_at: string;
  opponent_id: string;
  winner: string;
  plies: number;
  actions: string;
  compact: string;
  source: string;
};

export async function getHumanGame(id: string) {
  const d1 = await database();
  const row = await d1.prepare(`SELECT id, recorded_at, opponent_id, winner, plies, actions, compact, source
    FROM human_games WHERE id = ?`).bind(id).first<HumanGameRow>();
  if (!row) return null;

  const game = validateHumanGame({
    opponentId: row.opponent_id,
    winner: row.winner,
    actions: JSON.parse(row.actions),
  });
  if (game.actions.length !== row.plies || compactHumanGame(game) !== row.compact) {
    throw new Error("Stored game record failed integrity validation");
  }

  return {
    id: row.id,
    recordedAt: row.recorded_at,
    opponentId: game.opponentId,
    winner: game.winner,
    plies: game.actions.length,
    actions: game.actions,
    compact: row.compact,
    source: row.source,
  };
}
