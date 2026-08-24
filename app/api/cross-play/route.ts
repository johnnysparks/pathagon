import { PATHFINDER_SEARCH, SURVEYOR_SEARCH } from "../../ai";
import { defaultAgentSpecification, TYPESCRIPT_ENGINE } from "../../contract";
import { chooseLunaticAction } from "../../opponents";
import { querySelfPlayGames, storeSelfPlayGames } from "../../../db/selfplay-games";
import { createRandomAgent, createSearchAgent, mulberry32, playGame } from "../../../selfplay/core";
import type { SelfPlayAgent } from "../../../selfplay/core";

const TARGET_GAMES = 10;
const BRIDGED_GAMES = 4;
const MAX_QUERY_GAMES = 500;
const MAX_SEQUENCE = TARGET_GAMES - 1;
const ALL_CROSS_PLAY_RUN_ID = "all-cross-play";
const RUN_ID_PATTERN = /^[a-zA-Z0-9._:-]{1,120}$/;
const PLAYABLE_AGENTS = [
  { id: "pathfinder-v0.3.0", label: "The Pathfinder", kind: "heuristic" as const, tone: "green" },
  { id: "surveyor-v0.2.0", label: "The Surveyor", kind: "heuristic" as const, tone: "violet" },
  { id: "lunatic-v0.1.0", label: "Lunatic", kind: "heuristic" as const, tone: "gold" },
  { id: "coin-flip-v0.0.1", label: "Coin Flip", kind: "random" as const, tone: "muted" },
] as const;
const AGENTS = [
  ...PLAYABLE_AGENTS,
  { id: "gnn-warmstart-7x7", label: "GNN Learner", kind: "gnn" as const, tone: "green" },
] as const;

const BASELINE_RATINGS: Record<string, number> = {
  "pathfinder-v0.3.0": 1_142,
  "surveyor-v0.2.0": 1_085,
  "lunatic-v0.1.0": 1_059,
  "coin-flip-v0.0.1": 935,
  "gnn-warmstart-7x7": 957,
};

export async function POST(request: Request) {
  try {
    const payload = await request.json() as Record<string, unknown>;
    const runId = validateRunId(payload.runId);
    const sequence = validateInteger(payload.sequence, "sequence", 0, MAX_SEQUENCE);
    const seed = validateInteger(payload.seed, "seed", 0, 4_294_967_295);
    const existing = await queryRun(runId);
    if (existing.length >= TARGET_GAMES) {
      return Response.json({ accepted: false, error: "This cross-play run already has 10 games", games: existing.length }, { status: 409 });
    }

    const agents = choosePair(seed);
    const leftIsLight = sequence % 2 === 0;
    const light = leftIsLight ? agents.left : agents.right;
    const dark = leftIsLight ? agents.right : agents.left;
    const record = playGame(light.agent, dark.agent, {
      seed,
      maxPlies: 180,
      openingRandomPlies: 4,
    });
    const id = `cross-play-${runId}-${sequence}`;
    const result = await storeSelfPlayGames([{
      id,
      engine: "typescript-live-cross-play",
      mode: "cross-play",
      runId,
      record,
    }]);
    return Response.json({
      accepted: true,
      inserted: result.inserted,
      runId,
      sequence,
      game: summarizeGame(id, record),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to play cross-play game";
    return Response.json({ accepted: false, error: message }, { status: 400 });
  }
}

export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const requestedRunId = url.searchParams.get("runId");
    const aggregate = requestedRunId === ALL_CROSS_PLAY_RUN_ID || (!requestedRunId && url.searchParams.get("latest") === "1");
    const runId = aggregate ? ALL_CROSS_PLAY_RUN_ID : requestedRunId ? validateRunId(requestedRunId) : (() => { throw new Error("A cross-play run ID is required"); })();
    const records = aggregate ? await queryAllCrossPlayGames() : await queryRun(runId);
    if (!records.length) return Response.json({ found: false, error: "No cross-play runs yet" }, { status: 404 });
    const standings = buildStandings(records);
    const targetGames = aggregate ? records.length : targetGamesForRun(runId);
    return Response.json({
      runId,
      targetGames,
      games: records.length,
      status: records.length >= targetGames ? "complete" : records.length ? "running" : "ready",
      standings,
      latest: records.slice(-5).reverse().map(({ id, record }) => summarizeGame(id, record)),
    }, { headers: { "cache-control": "private, no-store" } });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to read cross-play run";
    return Response.json({ found: false, error: message }, { status: 400 });
  }
}

async function queryRun(runId: string) {
  const exactGames = await querySelfPlayGames({ runId, limit: MAX_QUERY_GAMES, offset: 0 });
  const exactRecords = exactGames.filter((game) => game.mode === "cross-play");
  return sortRecords(exactRecords);
}

async function queryAllCrossPlayGames() {
  const records = [];
  for (let offset = 0; ; offset += MAX_QUERY_GAMES) {
    const games = await querySelfPlayGames({ mode: "cross-play", limit: MAX_QUERY_GAMES, offset });
    records.push(...games.filter((game) => game.mode === "cross-play"));
    if (games.length < MAX_QUERY_GAMES) break;
  }
  return sortRecords(records);
}

function targetGamesForRun(runId: string) {
  return runId.startsWith("gnn-surveyor-") ? BRIDGED_GAMES : TARGET_GAMES;
}

function sortRecords<T extends { recordedAt: string; id: string; record: { seed: number } }>(records: T[]) {
  return records.sort((left, right) => left.recordedAt.localeCompare(right.recordedAt) || left.record.seed - right.record.seed || left.id.localeCompare(right.id));
}

function choosePair(seed: number) {
  const random = mulberry32(seed);
  const leftIndex = Math.floor(random() * PLAYABLE_AGENTS.length);
  let rightIndex = Math.floor(random() * (PLAYABLE_AGENTS.length - 1));
  if (rightIndex >= leftIndex) rightIndex += 1;
  const left = PLAYABLE_AGENTS[leftIndex];
  const right = PLAYABLE_AGENTS[rightIndex];
  return { left: { ...left, agent: createAgent(left.id) }, right: { ...right, agent: createAgent(right.id) } };
}

function createAgent(id: string): SelfPlayAgent {
  if (id === "pathfinder-v0.3.0") {
    return createSearchAgent(id, { ...PATHFINDER_SEARCH, depth: 2, maxNodes: 1_000, beamWidth: 8 });
  }
  if (id === "surveyor-v0.2.0") {
    return createSearchAgent(id, { ...SURVEYOR_SEARCH, depth: 1, maxNodes: 500, beamWidth: 12 });
  }
  if (id === "coin-flip-v0.0.1") return createRandomAgent(id);
  return {
    id,
    spec: defaultAgentSpecification(id, "search", TYPESCRIPT_ENGINE, { depth: 1, nodeBudget: 1 }),
    chooseAction(state) {
      return { action: chooseLunaticAction(state), nodes: 1, completedDepth: 1 };
    },
  };
}

function buildStandings(records: Awaited<ReturnType<typeof queryRun>>) {
  const ratings = new Map(Object.entries(BASELINE_RATINGS));
  const summaries = new Map(AGENTS.map((agent) => [agent.id, { games: 0, wins: 0, losses: 0, draws: 0, points: 0 }]));
  for (const { record } of records) {
    updateElo(ratings, record.agents.light, record.agents.dark, record.winner);
    for (const agent of [record.agents.light, record.agents.dark]) {
      const summary = summaries.get(agent);
      if (!summary) continue;
      summary.games += 1;
      if (!record.winner) summary.draws += 1;
      else if (record.agents[record.winner] === agent) summary.wins += 1;
      else summary.losses += 1;
      summary.points = summary.wins + summary.draws * 0.5;
    }
  }
  return AGENTS.map((agent) => ({
    id: agent.id,
    label: agent.label,
    tone: agent.tone,
    rating: Math.round(ratings.get(agent.id) ?? BASELINE_RATINGS[agent.id]),
    ...(summaries.get(agent.id) ?? { games: 0, wins: 0, losses: 0, draws: 0, points: 0 }),
  })).sort((left, right) => right.rating - left.rating || right.points - left.points);
}

function updateElo(ratings: Map<string, number>, light: string, dark: string, winner: "light" | "dark" | null) {
  const lightRating = ratings.get(light) ?? 1_000;
  const darkRating = ratings.get(dark) ?? 1_000;
  const expectedLight = 1 / (1 + 10 ** ((darkRating - lightRating) / 400));
  const actualLight = winner === "light" ? 1 : winner === "dark" ? 0 : 0.5;
  ratings.set(light, lightRating + 24 * (actualLight - expectedLight));
  ratings.set(dark, darkRating + 24 * ((1 - actualLight) - (1 - expectedLight)));
}

function summarizeGame(id: string, record: Awaited<ReturnType<typeof queryRun>>[number]["record"]) {
  return {
    id,
    seed: record.seed,
    light: labelFor(record.agents.light),
    dark: labelFor(record.agents.dark),
    winner: record.winner ? labelFor(record.agents[record.winner]) : null,
    result: record.result,
    reason: record.reason,
    plies: record.plies,
  };
}

function labelFor(id: string) {
  return AGENTS.find((agent) => agent.id === id)?.label ?? id;
}

function validateRunId(value: unknown): string {
  if (typeof value !== "string" || !RUN_ID_PATTERN.test(value)) throw new Error("Invalid cross-play run ID");
  return value;
}

function validateInteger(value: unknown, label: string, minimum: number, maximum: number): number {
  if (!Number.isSafeInteger(value) || Number(value) < minimum || Number(value) > maximum) throw new Error(`Invalid cross-play ${label}`);
  return Number(value);
}
