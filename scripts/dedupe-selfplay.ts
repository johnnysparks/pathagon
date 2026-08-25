import { createHash } from "node:crypto";

const args = parseArgs(process.argv.slice(2));
const url = args.url;
const agent = args.agent;
const mode = args.mode ?? "cross-play";
const token = args.token ?? process.env.PATHAGON_ARCHIVE_TOKEN;
if (!url || !agent) throw new Error("Usage: node --experimental-strip-types scripts/dedupe-selfplay.ts --url <site-url> --agent <agent-id> [--mode cross-play] [--apply]");

const endpoint = url.replace(/\/$/, "") + "/api/selfplay";
const games = await loadGames(endpoint, agent, mode, token);
const bySignature = new Map<string, ArchiveGame>();
const duplicateIds: string[] = [];
for (const game of games.sort((left, right) => left.recordedAt.localeCompare(right.recordedAt) || left.id.localeCompare(right.id))) {
  const signature = replaySignature(game.record);
  if (bySignature.has(signature)) duplicateIds.push(game.id);
  else bySignature.set(signature, game);
}

const summary = {
  agent,
  mode,
  fetched: games.length,
  uniqueTrajectories: bySignature.size,
  duplicateRecords: duplicateIds.length,
  applied: Boolean(args.apply),
};
if (!args.apply) {
  console.log(JSON.stringify(summary, null, 2));
  process.exit(0);
}

let deleted = 0;
for (let start = 0; start < duplicateIds.length; start += 80) {
  const ids = duplicateIds.slice(start, start + 80);
  const response = await fetch(endpoint, {
    method: "DELETE",
    headers: {
      "content-type": "application/json",
      ...(token ? { "OAI-Sites-Authorization": `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ ids }),
  });
  const body = await response.text();
  if (!response.ok) throw new Error(`Self-play deletion rejected: ${body}`);
  deleted += (JSON.parse(body) as { deleted?: number }).deleted ?? 0;
}
console.log(JSON.stringify({ ...summary, deleted }, null, 2));

type ArchiveGame = {
  id: string;
  recordedAt: string;
  record: {
    config?: unknown;
    winner: string | null;
    moves: Array<{ action: unknown }>;
  };
};

async function loadGames(endpoint: string, agent: string, mode: string, token: string | undefined) {
  const games: ArchiveGame[] = [];
  for (let offset = 0; ; offset += 500) {
    const query = new URLSearchParams({ mode, agent, limit: "500", offset: String(offset) });
    const response = await fetch(`${endpoint}?${query}`, {
      headers: token ? { "OAI-Sites-Authorization": `Bearer ${token}` } : {},
    });
    const body = await response.text();
    if (!response.ok) throw new Error(`Self-play query rejected: ${body}`);
    const page = JSON.parse(body) as { games?: ArchiveGame[] };
    const batch = page.games ?? [];
    games.push(...batch);
    if (batch.length < 500) return games;
  }
}

function replaySignature(record: ArchiveGame["record"]) {
  const payload = {
    config: record.config,
    winner: record.winner,
    moves: record.moves.map((move) => move.action),
  };
  return createHash("sha256").update(JSON.stringify(payload)).digest("hex");
}

function parseArgs(values: string[]) {
  const parsed: Record<string, string> = {};
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!value.startsWith("--")) continue;
    const [key, inline] = value.slice(2).split("=", 2);
    parsed[key] = inline ?? values[++index] ?? "true";
  }
  return parsed;
}
