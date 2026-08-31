import { compactHumanGame, validateHumanGame } from "../app/game-record";
import type { HumanGameSubmission } from "../app/game-record";

async function database() {
  const { env } = await import("cloudflare:workers");
  const d1 = env.DB;
  if (!d1) throw new Error("Human game database is unavailable");
  return d1;
}

export async function storeHumanGame(id: string, game: HumanGameSubmission, compact: string) {
  const d1 = await database();
  return d1.prepare(`INSERT OR IGNORE INTO human_games
    (id, schema_version, opponent_id, winner, plies, actions, compact, metadata)
    VALUES (?, 2, ?, ?, ?, ?, ?, ?)`)
    .bind(id, game.opponentId, game.winner, game.actions.length, JSON.stringify(game.actions), compact, JSON.stringify(game.metadata ?? {}))
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
  metadata: string;
  source: string;
};

export async function getHumanGame(id: string) {
  const d1 = await database();
  const row = await d1.prepare(`SELECT id, recorded_at, opponent_id, winner, plies, actions, compact, metadata, source
    FROM human_games WHERE id = ?`).bind(id).first<HumanGameRow>();
  if (!row) return null;

  const game = validateHumanGame({
    opponentId: row.opponent_id,
    winner: row.winner,
    actions: JSON.parse(row.actions),
    metadata: JSON.parse(row.metadata || "{}"),
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
    metadata: game.metadata ?? {},
    source: row.source,
  };
}
