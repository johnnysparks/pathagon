import { readFile } from "node:fs/promises";
import { basename, resolve } from "node:path";
import { validateSelfPlayRecord } from "../apps/web/app/selfplay-record.ts";
import type { SelfPlayGameRecord } from "../apps/web/app/selfplay-record.ts";

const args = parseArgs(process.argv.slice(2));
const file = args.file;
const url = args.url;
if (!file || !url) {
  throw new Error("Usage: npm run selfplay:archive -- --file <json-or-jsonl> --url <site-url> --engine <rust|python> [--mode arena] [--run-id name] [--replace-agent old-id=new-id] [--dry-run]");
}

const sourcePath = resolve(file);
const records = await loadRecords(sourcePath, parseAgentReplacement(args["replace-agent"]));
if (!records.length) throw new Error(`No complete self-play records found in ${sourcePath}`);

const engine = args.engine;
if (engine !== "rust" && engine !== "python") {
  throw new Error("Archive engine is required and must be rust or python");
}
const mode = args.mode ?? inferMode(sourcePath);
const runId = sanitize(args["run-id"] ?? args.runId ?? basename(sourcePath).replace(/\.(json|jsonl)$/i, ""));
const endpoint = url.replace(/\/$/, "") + "/api/selfplay";
const token = args.token ?? process.env.PATHAGON_ARCHIVE_TOKEN;
let inserted = 0;
if (!args["dry-run"]) {
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
}

console.log(JSON.stringify({ archived: records.length, inserted, dryRun: Boolean(args["dry-run"]), endpoint, engine, mode, runId }, null, 2));

async function loadRecords(path: string, replacement?: AgentReplacement): Promise<SelfPlayGameRecord[]> {
  const source = await readFile(path, "utf8");
  try {
    const parsed = JSON.parse(source) as unknown;
    if (Array.isArray(parsed)) return parsed.map((value) => normalizeRecord(value, replacement));
    if (parsed && typeof parsed === "object" && Array.isArray((parsed as Record<string, unknown>).games)) {
      return ((parsed as Record<string, unknown>).games as unknown[]).map((value) => normalizeRecord(value, replacement));
    }
    return [normalizeRecord(parsed, replacement)];
  } catch (error) {
    if (error instanceof SyntaxError) {
      return source.split("\n")
        .map((line) => line.trim())
        .filter(Boolean)
        .map((line) => JSON.parse(line) as unknown)
        .filter((value): value is Record<string, unknown> => Boolean(value && typeof value === "object" && Array.isArray(value.moves)))
        .map((value) => normalizeRecord(value, replacement));
    }
    throw error;
  }
}

type AgentReplacement = { from: string; to: string };

function normalizeRecord(value: unknown, replacement?: AgentReplacement) {
  const record = validateSelfPlayRecord(value);
  if (!replacement || !Object.values(record.agents).includes(replacement.from)) return record;
  const agents = {
    light: record.agents.light === replacement.from ? replacement.to : record.agents.light,
    dark: record.agents.dark === replacement.from ? replacement.to : record.agents.dark,
  } as const;
  const agentSpecifications = {
    light: record.agentSpecifications.light.id === replacement.from
      ? { ...record.agentSpecifications.light, id: replacement.to }
      : record.agentSpecifications.light,
    dark: record.agentSpecifications.dark.id === replacement.from
      ? { ...record.agentSpecifications.dark, id: replacement.to }
      : record.agentSpecifications.dark,
  } as const;
  return validateSelfPlayRecord({ ...record, agents, agentSpecifications });
}

function parseAgentReplacement(value: string | undefined): AgentReplacement | undefined {
  if (value === undefined) return undefined;
  const separator = value.indexOf("=");
  if (separator <= 0 || separator === value.length - 1) throw new Error("Agent replacement must use old-id=new-id");
  const from = value.slice(0, separator);
  const to = value.slice(separator + 1);
  if (!/^[a-zA-Z0-9._:-]{1,128}$/.test(from) || !/^[a-zA-Z0-9._:-]{1,128}$/.test(to)) {
    throw new Error("Agent replacement IDs are invalid");
  }
  return { from, to };
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
