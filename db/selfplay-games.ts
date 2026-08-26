import type { SelfPlayGameRecord } from "../app/selfplay-record.ts";

export type SelfPlayArchiveEntry = {
  id: string;
  engine: string;
  mode: string;
  runId: string | null;
  record: SelfPlayGameRecord;
};

async function database() {
  const { env } = await import("cloudflare:workers");
  const d1 = env.DB;
  if (!d1) throw new Error("Self-play database is unavailable");
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

export async function deleteSelfPlayGames(ids: string[]) {
  if (!ids.length) return { deleted: 0 };
  const d1 = await database();
  const placeholders = ids.map(() => "?").join(", ");
  const result = await d1.prepare(`DELETE FROM selfplay_games WHERE id IN (${placeholders})`).bind(...ids).run();
  return { deleted: result.meta.changes ?? 0 };
}

export type SelfPlayFilters = {
  engine?: string;
  excludeEngine?: string;
  mode?: string;
  agent?: string;
  winner?: "light" | "dark";
  result?: "win" | "draw";
  reason?: SelfPlayArchiveEntry["record"]["reason"];
  runId?: string;
  pair?: [string, string];
};

export type SelfPlayQuery = SelfPlayFilters & {
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
  const { where, values } = buildWhere(query);
  const rows = await d1.prepare(`SELECT id, recorded_at, engine, mode, run_id, record
    FROM selfplay_games ${where} ORDER BY recorded_at DESC, id DESC LIMIT ? OFFSET ?`)
    .bind(...values, query.limit, query.offset)
    .all<SelfPlayRow>();
  return rows.results.map(toArchiveGame);
}

export async function countSelfPlayGames(query: SelfPlayFilters) {
  const d1 = await database();
  const { where, values } = buildWhere(query);
  const row = await d1.prepare(`SELECT COUNT(*) AS count FROM selfplay_games ${where}`)
    .bind(...values)
    .first<{ count: number | string }>();
  return Number(row?.count ?? 0);
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

function buildWhere(query: SelfPlayFilters) {
  const conditions: string[] = [];
  const values: string[] = [];
  if (query.engine) { conditions.push("engine = ?"); values.push(query.engine); }
  if (query.excludeEngine) { conditions.push("engine != ?"); values.push(query.excludeEngine); }
  if (query.mode) { conditions.push("mode = ?"); values.push(query.mode); }
  if (query.agent) { conditions.push("(light_agent = ? OR dark_agent = ?)"); values.push(query.agent, query.agent); }
  if (query.winner) { conditions.push("winner = ?"); values.push(query.winner); }
  if (query.result) { conditions.push("result = ?"); values.push(query.result); }
  if (query.reason) { conditions.push("reason = ?"); values.push(query.reason); }
  if (query.runId) { conditions.push("run_id = ?"); values.push(query.runId); }
  if (query.pair) {
    conditions.push("((light_agent = ? AND dark_agent = ?) OR (light_agent = ? AND dark_agent = ?))");
    values.push(query.pair[0], query.pair[1], query.pair[1], query.pair[0]);
  }
  return { where: conditions.length ? `WHERE ${conditions.join(" AND ")}` : "", values };
}
