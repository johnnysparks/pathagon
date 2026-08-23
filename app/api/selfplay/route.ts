import { validateSelfPlayRecord } from "../../selfplay-record";
import { querySelfPlayGames, storeSelfPlayGames } from "../../../db/selfplay-games";
import type { SelfPlayArchiveEntry } from "../../../db/selfplay-games";

const MAX_BATCH_SIZE = 100;
const ID_PATTERN = /^[a-zA-Z0-9._:-]{1,160}$/;
const FIELD_PATTERN = /^[a-zA-Z0-9._:-]{1,128}$/;

export async function POST(request: Request) {
  try {
    const payload = await request.json();
    if (!payload || typeof payload !== "object" || !Array.isArray((payload as Record<string, unknown>).games)) {
      throw new Error("Self-play upload must contain a games array");
    }
    const input = payload as Record<string, unknown>;
    const games = input.games as unknown[];
    if (games.length < 1 || games.length > MAX_BATCH_SIZE) throw new Error(`Upload must contain 1-${MAX_BATCH_SIZE} games`);
    const defaultEngine = validateField(input.engine, "typescript");
    const defaultMode = validateField(input.mode, "arena");
    const defaultRunId = input.runId === undefined || input.runId === null ? null : validateField(input.runId, "run");
    const entries: SelfPlayArchiveEntry[] = games.map((candidate, index) => {
      if (!candidate || typeof candidate !== "object") throw new Error(`Invalid self-play upload at index ${index}`);
      const item = candidate as Record<string, unknown>;
      const record = validateSelfPlayRecord(item.record ?? candidate);
      const id = item.id === undefined ? crypto.randomUUID() : validateId(item.id);
      const engine = item.engine === undefined ? defaultEngine : validateField(item.engine, defaultEngine);
      const mode = item.mode === undefined ? defaultMode : validateField(item.mode, defaultMode);
      const runId = item.runId === undefined || item.runId === null ? defaultRunId : validateField(item.runId, "run");
      return { id, engine, mode, runId, record };
    });
    const result = await storeSelfPlayGames(entries);
    return Response.json({ accepted: true, inserted: result.inserted, received: entries.length });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Invalid self-play upload";
    return Response.json({ accepted: false, error: message }, { status: 400 });
  }
}

export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const limit = boundedInteger(url.searchParams.get("limit"), 100, 1, 500);
    const offset = boundedInteger(url.searchParams.get("offset"), 0, 0, 1_000_000);
    const query = {
      engine: optionalField(url.searchParams.get("engine")),
      mode: optionalField(url.searchParams.get("mode")),
      agent: optionalField(url.searchParams.get("agent")),
      winner: optionalEnum(url.searchParams.get("winner"), ["light", "dark"] as const),
      result: optionalEnum(url.searchParams.get("result"), ["win", "draw"] as const),
      reason: optionalEnum(url.searchParams.get("reason"), ["path", "threefold-repetition", "max-plies", "no-legal-action"] as const),
      runId: optionalField(url.searchParams.get("runId")),
      limit,
      offset,
    };
    const games = await querySelfPlayGames(query);
    if (url.searchParams.get("format") === "jsonl") {
      return new Response(`${games.map((game) => JSON.stringify(game)).join("\n")}\n`, {
        headers: { "cache-control": "private, no-store", "content-type": "application/x-ndjson; charset=utf-8" },
      });
    }
    return Response.json({ count: games.length, limit, offset, games }, { headers: { "cache-control": "private, no-store" } });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to read self-play records";
    return Response.json({ found: false, error: message }, { status: 400 });
  }
}

function validateId(value: unknown) {
  if (typeof value !== "string" || !ID_PATTERN.test(value)) throw new Error("Invalid self-play record ID");
  return value;
}

function validateField(value: unknown, fallback: string) {
  if (value === undefined || value === null) return fallback;
  if (typeof value !== "string" || !FIELD_PATTERN.test(value)) throw new Error("Invalid self-play archive field");
  return value;
}

function optionalField(value: string | null) {
  return value === null ? undefined : validateField(value, "");
}

function boundedInteger(value: string | null, fallback: number, minimum: number, maximum: number) {
  if (value === null) return fallback;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < minimum || parsed > maximum) throw new Error("Invalid pagination value");
  return parsed;
}

function optionalEnum<const Values extends readonly string[]>(value: string | null, values: Values): Values[number] | undefined {
  if (value === null) return undefined;
  if (!(values as readonly string[]).includes(value)) throw new Error("Invalid self-play filter");
  return value as Values[number];
}
