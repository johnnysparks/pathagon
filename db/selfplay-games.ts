import type { SelfPlayGameRecord } from "../app/selfplay-record.ts";

export type SelfPlayArchiveEntry = {
  id: string;
  engine: string;
  mode: string;
  runId: string | null;
  record: SelfPlayGameRecord;
};

const tableSql = `CREATE TABLE IF NOT EXISTS selfplay_games (
  id TEXT PRIMARY KEY,
  schema_version INTEGER NOT NULL,
  recorded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  engine TEXT NOT NULL,
  mode TEXT NOT NULL,
  run_id TEXT,
  seed INTEGER NOT NULL,
  light_agent TEXT NOT NULL,
  dark_agent TEXT NOT NULL,
  winner TEXT,
  result TEXT NOT NULL,
  reason TEXT NOT NULL,
  plies INTEGER NOT NULL,
  record TEXT NOT NULL,
  source TEXT NOT NULL DEFAULT 'selfplay-v1'
)`;
const recordedIndexSql = "CREATE INDEX IF NOT EXISTS selfplay_games_recorded_at_idx ON selfplay_games(recorded_at)";
const engineModeIndexSql = "CREATE INDEX IF NOT EXISTS selfplay_games_engine_mode_idx ON selfplay_games(engine, mode)";
const agentsIndexSql = "CREATE INDEX IF NOT EXISTS selfplay_games_agents_idx ON selfplay_games(light_agent, dark_agent)";
const resultIndexSql = "CREATE INDEX IF NOT EXISTS selfplay_games_result_idx ON selfplay_games(result, winner)";

async function database() {
  const { env } = await import("cloudflare:workers");
  const d1 = env.DB;
  if (!d1) throw new Error("Self-play database is unavailable");
  await d1.batch([
    d1.prepare(tableSql),
    d1.prepare(recordedIndexSql),
    d1.prepare(engineModeIndexSql),
    d1.prepare(agentsIndexSql),
    d1.prepare(resultIndexSql),
  ]);
  return d1;
}

export async function storeSelfPlayGames(entries: SelfPlayArchiveEntry[]) {
  if (!entries.length) return { inserted: 0 };
  const d1 = await database();
  const statements = entries.map(({ id, engine, mode, runId, record }) => d1.prepare(`INSERT OR IGNORE INTO selfplay_games
    (id, schema_version, engine, mode, run_id, seed, light_agent, dark_agent, winner, result, reason, plies, record)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
    .bind(
      id,
      record.contractVersion,
      engine,
      mode,
      runId,
      record.seed,
      record.agents.light,
      record.agents.dark,
      record.winner,
      record.result,
      record.reason,
      record.plies,
      JSON.stringify(record),
    ));
  const results = await d1.batch(statements);
  return { inserted: results.reduce((count, result) => count + (result.meta.changes ?? 0), 0) };
}

export type SelfPlayQuery = {
  engine?: string;
  mode?: string;
  agent?: string;
  winner?: "light" | "dark";
  result?: "win" | "draw";
  reason?: SelfPlayArchiveEntry["record"]["reason"];
  runId?: string;
  limit: number;
  offset: number;
};

type SelfPlayRow = {
  id: string;
  recorded_at: string;
  engine: string;
  mode: string;
  run_id: string | null;
  record: string;
};

export async function querySelfPlayGames(query: SelfPlayQuery) {
  const d1 = await database();
  const conditions: string[] = [];
  const values: string[] = [];
  if (query.engine) { conditions.push("engine = ?"); values.push(query.engine); }
  if (query.mode) { conditions.push("mode = ?"); values.push(query.mode); }
  if (query.agent) { conditions.push("(light_agent = ? OR dark_agent = ?)"); values.push(query.agent, query.agent); }
  if (query.winner) { conditions.push("winner = ?"); values.push(query.winner); }
  if (query.result) { conditions.push("result = ?"); values.push(query.result); }
  if (query.reason) { conditions.push("reason = ?"); values.push(query.reason); }
  if (query.runId) { conditions.push("run_id = ?"); values.push(query.runId); }
  const where = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";
  const rows = await d1.prepare(`SELECT id, recorded_at, engine, mode, run_id, record
    FROM selfplay_games ${where} ORDER BY recorded_at DESC, id DESC LIMIT ? OFFSET ?`)
    .bind(...values, query.limit, query.offset)
    .all<SelfPlayRow>();
  return rows.results.map(toArchiveGame);
}

export async function getSelfPlayGame(id: string) {
  const d1 = await database();
  const row = await d1.prepare(`SELECT id, recorded_at, engine, mode, run_id, record
    FROM selfplay_games WHERE id = ?`).bind(id).first<SelfPlayRow>();
  return row ? toArchiveGame(row) : null;
}

function toArchiveGame(row: SelfPlayRow) {
  return {
    id: row.id,
    recordedAt: row.recorded_at,
    engine: row.engine,
    mode: row.mode,
    runId: row.run_id,
    record: JSON.parse(row.record) as SelfPlayArchiveEntry["record"],
  };
}
