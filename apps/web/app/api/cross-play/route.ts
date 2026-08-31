import { countSelfPlayGames, querySelfPlayGames, querySelfPlayResults } from "../../../db/selfplay-games";
import type { SelfPlayGameRecord } from "../../selfplay-record";
import type { SelfPlayResult } from "../../../db/selfplay-games";
import { applyAction, createGame } from "../../pathagon";
import { LEAGUE_MODELS, RANKED_LEAGUE_MODELS, isRankedLeagueModel, leagueModel } from "../../league-models";

const MAX_QUERY_GAMES = 500;
const DEFAULT_HISTORY_LIMIT = 24;
const MAX_HISTORY_LIMIT = 50;
const ALL_CROSS_PLAY_RUN_ID = "all-cross-play";
const RUN_ID_PATTERN = /^[a-zA-Z0-9._:-]{1,120}$/;
const WEB_GENERATED_ENGINE = "typescript-live-cross-play";
const ARCHIVE_AGENTS = LEAGUE_MODELS;
const BASELINE_RATINGS = Object.fromEntries(RANKED_LEAGUE_MODELS.map((agent) => [agent.id, agent.initialRating]));

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
    const rankedResults = results.filter((result) => isRankedLeagueModel(result.lightAgent) && isRankedLeagueModel(result.darkAgent));
    const standings = buildStandings(rankedResults);
    const headToHead = buildHeadToHead(rankedResults);
    return Response.json({
      runId,
      targetGames: results.length,
      games: results.length,
      rankedGames: rankedResults.length,
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
  const summaries = new Map(RANKED_LEAGUE_MODELS.map((agent) => [agent.id, { games: 0, wins: 0, losses: 0, draws: 0, points: 0 }]));
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
  return RANKED_LEAGUE_MODELS.map((agent) => ({
    id: agent.id,
    label: agent.name,
    tone: agent.tone,
    rustEngine: true,
    rating: Math.round(ratings.get(agent.id) ?? BASELINE_RATINGS[agent.id]),
    ...(summaries.get(agent.id) ?? { games: 0, wins: 0, losses: 0, draws: 0, points: 0 }),
  })).sort((left, right) => right.rating - left.rating || right.points - left.points);
}

function buildHeadToHead(records: Awaited<ReturnType<typeof queryAllCrossPlayResults>>) {
  const agentIndex = new Map(RANKED_LEAGUE_MODELS.map((agent, index) => [agent.id, index]));
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

  for (let leftIndex = 0; leftIndex < RANKED_LEAGUE_MODELS.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < RANKED_LEAGUE_MODELS.length; rightIndex += 1) {
      const left = RANKED_LEAGUE_MODELS[leftIndex]!;
      const right = RANKED_LEAGUE_MODELS[rightIndex]!;
      pairings.set(`${left.id}|${right.id}`, {
        leftId: left.id,
        rightId: right.id,
        leftLabel: left.name,
        rightLabel: right.name,
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
    const left = RANKED_LEAGUE_MODELS[leftIndex]!;
    const right = RANKED_LEAGUE_MODELS[rightIndex]!;
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
  return leagueModel(id)?.name ?? id;
}

function validateRunId(value: unknown): string {
  if (typeof value !== "string" || !RUN_ID_PATTERN.test(value)) throw new Error("Invalid cross-play run ID");
  return value;
}

function validateAgent(value: string | null) {
  if (value === null) return undefined;
  if (!ARCHIVE_AGENTS.some((agent) => agent.id === value)) throw new Error("Invalid pairwise agent");
  return value;
}

function boundedInteger(value: string | null, fallback: number, minimum: number, maximum: number) {
  const parsed = value === null || value.trim() === "" ? fallback : Number(value);
  if (!Number.isInteger(parsed)) throw new Error("Pagination values must be integers");
  return Math.min(maximum, Math.max(minimum, parsed));
}
