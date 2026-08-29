import { countSelfPlayGames, querySelfPlayGames, querySelfPlayResults } from "../../../db/selfplay-games";
import type { SelfPlayGameRecord } from "../../selfplay-record";
import type { SelfPlayResult } from "../../../db/selfplay-games";
import { applyAction, createGame } from "../../pathagon";

const MAX_QUERY_GAMES = 500;
const DEFAULT_HISTORY_LIMIT = 24;
const MAX_HISTORY_LIMIT = 50;
const ALL_CROSS_PLAY_RUN_ID = "all-cross-play";
const RUN_ID_PATTERN = /^[a-zA-Z0-9._:-]{1,120}$/;
const WEB_GENERATED_ENGINE = "typescript-live-cross-play";
const AGENTS = [
  { id: "pathfinder-v0.5.0-trained-evaluator", label: "The Pathfinder · Trained", kind: "heuristic" as const, tone: "green" },
  { id: "pathfinder-v0.3.0", label: "The Pathfinder", kind: "heuristic" as const, tone: "green" },
  { id: "surveyor-v0.2.0", label: "The Surveyor", kind: "heuristic" as const, tone: "violet" },
  { id: "lunatic-v0.1.0", label: "Lunatic", kind: "heuristic" as const, tone: "gold" },
  { id: "coin-flip-v0.0.1", label: "Coin Flip", kind: "random" as const, tone: "muted" },
  { id: "gnn-warmstart-7x7", label: "GNN Learner", kind: "gnn" as const, tone: "green" },
  { id: "qadv-arbiter-7x7-v0.1.0", label: "The Q-Arbiter", kind: "learned" as const, tone: "violet" },
  { id: "qadv-arbiter-guided-7x7-v0.2.0", label: "The Q-Arbiter · Guided Search", kind: "learned" as const, tone: "green" },
  { id: "gnn-reval30k-7x7", label: "Re-evaluated GNN 30k", kind: "gnn" as const, tone: "green" },
  { id: "cnn-baseline-7x7", label: "CNN baseline", kind: "cnn" as const, tone: "gold" },
  { id: "cnn-reval30k-7x7", label: "Re-evaluated CNN 30k", kind: "cnn" as const, tone: "gold" },
  { id: "gnn-scout-7x7", label: "GNN Scout", kind: "gnn" as const, tone: "violet" },
  { id: "gnn-scout-puct32-7x7", label: "Scout + PUCT", kind: "gnn" as const, tone: "violet" },
  { id: "gnn-scout-beam-7x7", label: "Scout + Neural Beam", kind: "gnn" as const, tone: "green" },
  { id: "gnn-scout-hybrid-beam-7x7", label: "Scout + Hybrid Beam", kind: "gnn" as const, tone: "gold" },
  { id: "pathfinder-deep-10k-7x7", label: "Pathfinder + Deep Search", kind: "heuristic" as const, tone: "green" },
  { id: "gnn-scout-beam10k-7x7", label: "Scout + 10k Beam", kind: "gnn" as const, tone: "violet" },
] as const;

const BASELINE_RATINGS: Record<string, number> = {
  "pathfinder-v0.5.0-trained-evaluator": 1_160,
  "pathfinder-v0.3.0": 1_142,
  "surveyor-v0.2.0": 1_085,
  "lunatic-v0.1.0": 1_059,
  "coin-flip-v0.0.1": 935,
  "gnn-warmstart-7x7": 957,
  "qadv-arbiter-7x7-v0.1.0": 1_000,
  "qadv-arbiter-guided-7x7-v0.2.0": 1_000,
  "gnn-reval30k-7x7": 1_000,
  "cnn-baseline-7x7": 950,
  "cnn-reval30k-7x7": 1_000,
  "gnn-scout-7x7": 940,
  "gnn-scout-puct32-7x7": 940,
  "gnn-scout-beam-7x7": 940,
  "gnn-scout-hybrid-beam-7x7": 940,
  "pathfinder-deep-10k-7x7": 940,
  "gnn-scout-beam10k-7x7": 940,
};

export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const requestedRunId = url.searchParams.get("runId");
    const aggregate = requestedRunId === ALL_CROSS_PLAY_RUN_ID || (!requestedRunId && url.searchParams.get("latest") === "1");
    const runId = aggregate ? ALL_CROSS_PLAY_RUN_ID : requestedRunId ? validateRunId(requestedRunId) : (() => { throw new Error("A cross-play run ID is required"); })();
    if (url.searchParams.get("history") === "1") {
      return Response.json(await readHistoryPage(aggregate, runId, url), { headers: { "cache-control": "private, no-store" } });
    }
    const filters = aggregate
      ? { mode: "cross-play", excludeEngine: WEB_GENERATED_ENGINE }
      : { mode: "cross-play", excludeEngine: WEB_GENERATED_ENGINE, runId };
    const results = await queryAllCrossPlayResults(filters);
    if (!results.length) return Response.json({ found: false, error: "No imported cross-play games yet" }, { status: 404 });
    const latestRecords = await querySelfPlayGames({ ...filters, limit: 5, offset: 0 });
    const standings = buildStandings(results);
    const headToHead = buildHeadToHead(results);
    return Response.json({
      runId,
      targetGames: results.length,
      games: results.length,
      status: "complete",
      standings,
      headToHead,
      latest: latestRecords.map(({ id, recordedAt, record }) => summarizeGame(id, recordedAt, record)),
    }, { headers: { "cache-control": "private, no-store" } });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to read cross-play run";
    return Response.json({ found: false, error: message }, { status: 400 });
  }
}

async function readHistoryPage(aggregate: boolean, runId: string, url: URL) {
  const limit = boundedInteger(url.searchParams.get("limit"), DEFAULT_HISTORY_LIMIT, 1, MAX_HISTORY_LIMIT);
  const offset = boundedInteger(url.searchParams.get("offset"), 0, 0, 1_000_000);
  const pairLeft = validateAgent(url.searchParams.get("pairLeft"));
  const pairRight = validateAgent(url.searchParams.get("pairRight"));
  if (Boolean(pairLeft) !== Boolean(pairRight)) throw new Error("Both pairwise agents are required");
  const filters = aggregate
    ? { mode: "cross-play", excludeEngine: WEB_GENERATED_ENGINE, pair: pairLeft && pairRight ? [pairLeft, pairRight] as [string, string] : undefined }
    : { mode: "cross-play", excludeEngine: WEB_GENERATED_ENGINE, runId, pair: pairLeft && pairRight ? [pairLeft, pairRight] as [string, string] : undefined };
  const [records, total] = await Promise.all([
    querySelfPlayGames({ ...filters, limit, offset }),
    countSelfPlayGames(filters),
  ]);
  return {
    runId,
    games: records.map(({ id, recordedAt, record }) => summarizeGame(id, recordedAt, record)),
    total,
    limit,
    offset,
    hasMore: offset + records.length < total,
  };
}

async function queryAllCrossPlayResults(filters: { mode: string; excludeEngine: string; runId?: string }) {
  const results: SelfPlayResult[] = [];
  for (let offset = 0; ; offset += MAX_QUERY_GAMES) {
    const page = await querySelfPlayResults({ ...filters, limit: MAX_QUERY_GAMES, offset });
    results.push(...page);
    if (page.length < MAX_QUERY_GAMES) break;
  }
  return sortResults(results);
}

function sortResults<T extends { recordedAt: string; id: string; seed: number }>(results: T[]) {
  return results.sort((left, right) => left.recordedAt.localeCompare(right.recordedAt) || left.seed - right.seed || left.id.localeCompare(right.id));
}

function buildStandings(records: Awaited<ReturnType<typeof queryAllCrossPlayResults>>) {
  const ratings = new Map(Object.entries(BASELINE_RATINGS));
  const summaries = new Map(AGENTS.map((agent) => [agent.id, { games: 0, wins: 0, losses: 0, draws: 0, points: 0 }]));
  for (const result of sortResults(records)) {
    updateElo(ratings, result.lightAgent, result.darkAgent, result.winner);
    for (const agent of [result.lightAgent, result.darkAgent]) {
      const summary = summaries.get(agent);
      if (!summary) continue;
      summary.games += 1;
      if (!result.winner) summary.draws += 1;
      else if ((result.winner === "light" ? result.lightAgent : result.darkAgent) === agent) summary.wins += 1;
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

function buildHeadToHead(records: Awaited<ReturnType<typeof queryAllCrossPlayResults>>) {
  const agentIndex = new Map(AGENTS.map((agent, index) => [agent.id, index]));
  const pairings = new Map<string, {
    leftId: string;
    rightId: string;
    leftLabel: string;
    rightLabel: string;
    games: number;
    leftWins: number;
    rightWins: number;
    draws: number;
    leftPoints: number;
    rightPoints: number;
    leftLightGames: number;
    rightLightGames: number;
  }>();

  for (let leftIndex = 0; leftIndex < AGENTS.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < AGENTS.length; rightIndex += 1) {
      const left = AGENTS[leftIndex];
      const right = AGENTS[rightIndex];
      pairings.set(`${left.id}|${right.id}`, {
        leftId: left.id,
        rightId: right.id,
        leftLabel: left.label,
        rightLabel: right.label,
        games: 0,
        leftWins: 0,
        rightWins: 0,
        draws: 0,
        leftPoints: 0,
        rightPoints: 0,
        leftLightGames: 0,
        rightLightGames: 0,
      });
    }
  }

  for (const result of records) {
    const lightIndex = agentIndex.get(result.lightAgent);
    const darkIndex = agentIndex.get(result.darkAgent);
    if (lightIndex === undefined || darkIndex === undefined || lightIndex === darkIndex) continue;
    const leftIndex = Math.min(lightIndex, darkIndex);
    const rightIndex = Math.max(lightIndex, darkIndex);
    const left = AGENTS[leftIndex];
    const right = AGENTS[rightIndex];
    const pairing = pairings.get(`${left.id}|${right.id}`);
    if (!pairing) continue;
    pairing.games += 1;
    if (result.lightAgent === left.id) pairing.leftLightGames += 1;
    else pairing.rightLightGames += 1;
    if (!result.winner) {
      pairing.draws += 1;
      pairing.leftPoints += 0.5;
      pairing.rightPoints += 0.5;
      continue;
    }
    const winner = result.winner === "light" ? result.lightAgent : result.darkAgent;
    if (winner === left.id) {
      pairing.leftWins += 1;
      pairing.leftPoints += 1;
    } else if (winner === right.id) {
      pairing.rightWins += 1;
      pairing.rightPoints += 1;
    }
  }

  return [...pairings.values()].sort((left, right) => right.games - left.games || left.leftLabel.localeCompare(right.leftLabel) || left.rightLabel.localeCompare(right.rightLabel));
}

function updateElo(ratings: Map<string, number>, light: string, dark: string, winner: "light" | "dark" | null) {
  const lightRating = ratings.get(light) ?? 1_000;
  const darkRating = ratings.get(dark) ?? 1_000;
  const expectedLight = 1 / (1 + 10 ** ((darkRating - lightRating) / 400));
  const actualLight = winner === "light" ? 1 : winner === "dark" ? 0 : 0.5;
  ratings.set(light, lightRating + 24 * (actualLight - expectedLight));
  ratings.set(dark, darkRating + 24 * ((1 - actualLight) - (1 - expectedLight)));
}

function summarizeGame(id: string, recordedAt: string, record: SelfPlayGameRecord) {
  const finalState = record.moves.reduce(
    (state, move) => applyAction(state, move.action),
    createGame(record.config),
  );
  return {
    id,
    recordedAt,
    seed: record.seed,
    light: labelFor(record.agents.light),
    dark: labelFor(record.agents.dark),
    winner: record.winner ? labelFor(record.agents[record.winner]) : null,
    result: record.result,
    reason: record.reason,
    plies: record.plies,
    finalBoard: finalState.board,
    winningPath: finalState.winningPath,
  };
}

function labelFor(id: string) {
  return AGENTS.find((agent) => agent.id === id)?.label ?? id;
}

function validateRunId(value: unknown): string {
  if (typeof value !== "string" || !RUN_ID_PATTERN.test(value)) throw new Error("Invalid cross-play run ID");
  return value;
}

function validateAgent(value: string | null) {
  if (value === null) return undefined;
  if (!AGENTS.some((agent) => agent.id === value)) throw new Error("Invalid pairwise agent");
  return value;
}

function boundedInteger(value: string | null, fallback: number, minimum: number, maximum: number) {
  const parsed = value === null || value.trim() === "" ? fallback : Number(value);
  if (!Number.isInteger(parsed)) throw new Error("Pagination values must be integers");
  return Math.min(maximum, Math.max(minimum, parsed));
}
