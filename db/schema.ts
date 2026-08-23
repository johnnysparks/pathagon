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
  validation: text("validation").notNull().default("replay-valid"),
  source: text("source").notNull().default("web-human-v1"),
}, (table) => [
  index("human_games_recorded_at_idx").on(table.recordedAt),
  index("human_games_opponent_idx").on(table.opponentId),
]);
