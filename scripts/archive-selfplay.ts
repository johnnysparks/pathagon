import { readFile } from "node:fs/promises";
import { basename, resolve } from "node:path";
import { validateSelfPlayRecord } from "../app/selfplay-record.ts";
import type { SelfPlayGameRecord } from "../app/selfplay-record.ts";

const args = parseArgs(process.argv.slice(2));
const file = args.file;
const url = args.url;
if (!file || !url) {
  throw new Error("Usage: npm run selfplay:archive -- --file <json-or-jsonl> --url <site-url> [--engine rust|typescript] [--mode arena] [--run-id name]");
}

const sourcePath = resolve(file);
const records = await loadRecords(sourcePath);
if (!records.length) throw new Error(`No complete self-play records found in ${sourcePath}`);

const engine = args.engine ?? "typescript";
const mode = args.mode ?? inferMode(sourcePath);
const runId = sanitize(args.runId ?? basename(sourcePath).replace(/\.(json|jsonl)$/i, ""));
const endpoint = url.replace(/\/$/, "") + "/api/selfplay";
const token = args.token ?? process.env.PATHAGON_ARCHIVE_TOKEN;
let inserted = 0;
for (let start = 0; start < records.length; start += 100) {
  const games = records.slice(start, start + 100).map((record, offset) => ({
    id: `sp2-${engine}-${runId}-${record.seed}-${start + offset + 1}`.slice(0, 160),
    record,
  }));
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...(token ? { "OAI-Sites-Authorization": `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ engine, mode, runId, games }),
  });
  const body = await response.text();
  if (!response.ok) throw new Error(`Archive rejected batch ${start + 1}-${start + games.length}: ${body}`);
  const result = JSON.parse(body) as { inserted?: number };
  inserted += result.inserted ?? 0;
}

console.log(JSON.stringify({ archived: records.length, inserted, endpoint, engine, mode, runId }, null, 2));

async function loadRecords(path: string): Promise<SelfPlayGameRecord[]> {
  const source = await readFile(path, "utf8");
  try {
    const parsed = JSON.parse(source) as unknown;
    if (Array.isArray(parsed)) return parsed.map(validateSelfPlayRecord);
    if (parsed && typeof parsed === "object" && Array.isArray((parsed as Record<string, unknown>).games)) {
      return ((parsed as Record<string, unknown>).games as unknown[]).map(validateSelfPlayRecord);
    }
    return [validateSelfPlayRecord(parsed)];
  } catch (error) {
    if (error instanceof SyntaxError) {
      return source.split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => JSON.parse(line) as unknown)
        .filter((value): value is Record<string, unknown> => Boolean(value && typeof value === "object" && Array.isArray(value.moves)))
        .map(validateSelfPlayRecord);
    }
    throw error;
  }
}

function inferMode(path: string) {
  const name = basename(path).toLowerCase();
  if (name.includes("league")) return "league";
  if (name.includes("train")) return "train";
  return "arena";
}

function sanitize(value: string) {
  const result = value.replace(/[^a-zA-Z0-9._:-]/g, "-").replace(/^-+|-+$/g, "").slice(0, 80);
  return result || "run";
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
