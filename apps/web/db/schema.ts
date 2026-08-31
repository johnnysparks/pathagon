import { sql } from "drizzle-orm";
import { index, integer, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const humanGames = sqliteTable("human_games", {
  id: text("id").primaryKey(),
  schemaVersion: integer("schema_version").notNull(),
  recordedAt: text("recorded_at").notNull().default(sql`CURRENT_TIMESTAMP`),
  opponentId: text("opponent_id").notNull(),
  winner: text("winner").notNull(),
  plies: integer("plies").notNull(),
  actions: text("actions").notNull(),
  compact: text("compact").notNull(),
  metadata: text("metadata").notNull().default("{}"),
  validation: text("validation").notNull().default("replay-valid"),
  source: text("source").notNull().default("web-human-v1"),
}, (table) => [
  index("human_games_recorded_at_idx").on(table.recordedAt),
  index("human_games_opponent_idx").on(table.opponentId),
]);

export const selfplayGames = sqliteTable("selfplay_games", {
  id: text("id").primaryKey(),
  schemaVersion: integer("schema_version").notNull(),
  recordedAt: text("recorded_at").notNull().default(sql`CURRENT_TIMESTAMP`),
  engine: text("engine").notNull(),
  mode: text("mode").notNull(),
  runId: text("run_id"),
  seed: integer("seed").notNull(),
  lightAgent: text("light_agent").notNull(),
  darkAgent: text("dark_agent").notNull(),
  winner: text("winner"),
  result: text("result").notNull(),
  reason: text("reason").notNull(),
  plies: integer("plies").notNull(),
  record: text("record").notNull(),
  source: text("source").notNull().default("selfplay-v1"),
}, (table) => [
  index("selfplay_games_recorded_at_idx").on(table.recordedAt),
  index("selfplay_games_engine_mode_idx").on(table.engine, table.mode),
  index("selfplay_games_agents_idx").on(table.lightAgent, table.darkAgent),
  index("selfplay_games_result_idx").on(table.result, table.winner),
]);
