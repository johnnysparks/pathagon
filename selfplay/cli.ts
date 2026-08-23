import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { DEFAULT_WEIGHTS, PATHFINDER_SEARCH, SURVEYOR_SEARCH } from "../app/ai.ts";
import type { EvaluationWeights, SearchConfig } from "../app/ai.ts";
import { createRandomAgent, createSearchAgent, mulberry32, mutateWeights, playGame } from "./core.ts";
import type { GameRecord } from "./core.ts";

type Champion = {
  schemaVersion: 2;
  id: string;
  generation: number;
  weights: EvaluationWeights;
  search: Omit<SearchConfig, "weights">;
  promotedAt: string;
  evidence: { games: number; wins: number; losses: number; draws: number };
};

type LeagueEntry = Pick<Champion, "id" | "generation" | "weights" | "search" | "promotedAt">;

const args = parseArgs(process.argv.slice(2));
const mode = args.mode ?? "arena";
const seed = integer(args.seed, 20_260_822);
const games = integer(args.games, mode === "train" ? 8 : 20);
const maxPlies = integer(args.maxPlies, 180);
const openingRandomPlies = integer(args.openingRandomPlies, 2);
const progressDir = resolve(args.out ?? "selfplay/progress");
await mkdir(resolve(progressDir, "runs"), { recursive: true });

if (mode === "train") {
  await train();
} else if (mode === "league") {
  await leagueArena();
} else {
  await arena();
}

async function arena() {
  const champion = await loadChampion(progressDir);
  const search = createSearchAgent(champion.id, { ...champion.search, weights: champion.weights });
  const opponentName = args.opponent ?? "random";
  const opponent = opponentName === "pathfinder"
    ? createSearchAgent("pathfinder-handcrafted-v0.3.0", PATHFINDER_SEARCH)
    : opponentName === "surveyor"
      ? createSearchAgent("surveyor-baseline-v0.3.0", SURVEYOR_SEARCH)
      : createRandomAgent("coin-flip-seeded");
  const records: GameRecord[] = [];
  for (let game = 0; game < games; game += 1) {
    const searchIsLight = game % 2 === 0;
    records.push(playGame(
      searchIsLight ? search : opponent,
      searchIsLight ? opponent : search,
      { seed: seed + game, maxPlies, openingRandomPlies },
    ));
  }
  const summary = summarize(records, champion.id);
  const output = { schemaVersion: 2, mode: "arena", seed, opponent: opponent.id, champion, summary, games: records };
  const path = resolve(progressDir, "runs", `arena-${seed}.json`);
  await writeJson(path, output);
  await writeJson(resolve(progressDir, "latest.json"), { mode: "arena", path, summary });
  console.log(JSON.stringify({ path, summary }, null, 2));
}

async function train() {
  let champion = await loadChampion(progressDir);
  const league = await loadLeague(progressDir, champion);
  const generations = integer(args.generations, 1);
  const population = integer(args.population, 4);
  const poolSize = integer(args.poolSize, 4);
  const promotionRate = decimal(args.promotionRate, 0.55);
  const random = mulberry32(seed);
  const history: unknown[] = [];
  for (let generation = 0; generation < generations; generation += 1) {
    const opponents = uniqueLeagueEntries([champion, ...league.slice(-poolSize)]);
    let best = {
      id: champion.id,
      weights: champion.weights,
      score: Number.NEGATIVE_INFINITY,
      records: [] as GameRecord[],
      matchups: [] as Array<{ opponent: string; summary: ReturnType<typeof summarize> }>,
    };
    for (let candidateIndex = 0; candidateIndex < population; candidateIndex += 1) {
      const weights = mutateWeights(champion.weights, random);
      const challenger = createSearchAgent(`candidate-${champion.generation + 1}-${candidateIndex}`, { ...champion.search, weights });
      const records: GameRecord[] = [];
      const matchups: Array<{ opponent: string; summary: ReturnType<typeof summarize> }> = [];
      for (let opponentIndex = 0; opponentIndex < opponents.length; opponentIndex += 1) {
        const opponent = opponents[opponentIndex];
        const opponentAgent = createSearchAgent(opponent.id, { ...opponent.search, weights: opponent.weights });
        const matchupRecords: GameRecord[] = [];
        for (let game = 0; game < games; game += 1) {
          const challengerIsLight = game % 2 === 0;
          const record = playGame(
            challengerIsLight ? challenger : opponentAgent,
            challengerIsLight ? opponentAgent : challenger,
            { seed: seed + generation * 100_000 + candidateIndex * 10_000 + opponentIndex * 1_000 + Math.floor(game / 2), maxPlies, openingRandomPlies },
          );
          matchupRecords.push(record);
          records.push(record);
        }
        matchups.push({ opponent: opponent.id, summary: summarize(matchupRecords, challenger.id) });
      }
      const score = candidateScore(records, challenger.id);
      history.push({ generation: champion.generation + 1, candidate: challenger.id, weights, score, summary: summarize(records, challenger.id), matchups });
      if (score > best.score) best = { id: challenger.id, weights, score, records, matchups };
    }
    const evidence = summarize(best.records, best.id);
    const decisiveGames = evidence.wins + evidence.losses;
    const winRate = decisiveGames ? evidence.wins / decisiveGames : 0;
    const clearsPool = best.matchups.every(({ summary }) => summary.wins >= summary.losses);
    const beatsIncumbent = best.matchups.some(({ opponent, summary }) => opponent === champion.id && summary.wins > summary.losses);
    if (best.score > 0 && winRate >= promotionRate && clearsPool && beatsIncumbent) {
      league.push(toLeagueEntry(champion));
      const evidence = summarize(best.records, best.id);
      champion = {
        ...champion,
        schemaVersion: 2,
        id: `surveyor-trained-g${champion.generation + 1}`,
        generation: champion.generation + 1,
        weights: best.weights,
        promotedAt: new Date().toISOString(),
        evidence,
      };
      league.push(toLeagueEntry(champion));
    }
  }
  const path = resolve(progressDir, "runs", `train-${seed}.json`);
  await writeJson(path, { schemaVersion: 2, mode: "train", seed, champion, history });
  await writeJson(resolve(progressDir, "champion.json"), champion);
  await writeJson(resolve(progressDir, "league.json"), { schemaVersion: 1, entries: uniqueLeagueEntries(league) });
  await writeJson(resolve(progressDir, "latest.json"), { mode: "train", path, champion });
  console.log(JSON.stringify({ path, champion, trials: history.length }, null, 2));
}

async function leagueArena() {
  const champion = await loadChampion(progressDir);
  const entries = uniqueLeagueEntries([...(await loadLeague(progressDir, champion)), champion]);
  const ratings = new Map(entries.map((entry) => [entry.id, 1_000]));
  const records: GameRecord[] = [];
  for (let left = 0; left < entries.length; left += 1) {
    for (let right = left + 1; right < entries.length; right += 1) {
      const leftAgent = createSearchAgent(entries[left].id, { ...entries[left].search, weights: entries[left].weights });
      const rightAgent = createSearchAgent(entries[right].id, { ...entries[right].search, weights: entries[right].weights });
      for (let game = 0; game < games; game += 1) {
        const leftIsLight = game % 2 === 0;
        const record = playGame(
          leftIsLight ? leftAgent : rightAgent,
          leftIsLight ? rightAgent : leftAgent,
          { seed: seed + left * 100_000 + right * 1_000 + Math.floor(game / 2), maxPlies, openingRandomPlies },
        );
        records.push(record);
        updateRatings(ratings, record);
      }
    }
  }
  const standings = entries.map((entry) => ({
    id: entry.id,
    generation: entry.generation,
    rating: Math.round(ratings.get(entry.id) ?? 1_000),
    ...summarize(records.filter((record) => Object.values(record.agents).includes(entry.id)), entry.id),
  })).sort((left, right) => right.rating - left.rating);
  const path = resolve(progressDir, "runs", `league-${seed}.json`);
  await writeJson(path, { schemaVersion: 1, mode: "league", seed, standings, games: records });
  await writeJson(resolve(progressDir, "latest.json"), { mode: "league", path, standings });
  console.log(JSON.stringify({ path, standings }, null, 2));
}

async function loadChampion(directory: string): Promise<Champion> {
  try {
    const saved = JSON.parse(await readFile(resolve(directory, "champion.json"), "utf8")) as Partial<Champion>;
    return {
      schemaVersion: 2,
      id: saved.id ?? "surveyor-handcrafted-v0.3.0",
      generation: saved.generation ?? 0,
      weights: { ...DEFAULT_WEIGHTS, ...saved.weights },
      search: { depth: saved.search?.depth ?? SURVEYOR_SEARCH.depth, maxNodes: saved.search?.maxNodes ?? SURVEYOR_SEARCH.maxNodes, beamWidth: saved.search?.beamWidth ?? SURVEYOR_SEARCH.beamWidth },
      promotedAt: saved.promotedAt ?? "2026-08-22T00:00:00.000Z",
      evidence: saved.evidence ?? { games: 0, wins: 0, losses: 0, draws: 0 },
    };
  } catch {
    return {
      schemaVersion: 2,
      id: "surveyor-handcrafted-v0.3.0",
      generation: 0,
      weights: DEFAULT_WEIGHTS,
      search: { depth: SURVEYOR_SEARCH.depth, maxNodes: SURVEYOR_SEARCH.maxNodes, beamWidth: SURVEYOR_SEARCH.beamWidth },
      promotedAt: "2026-08-22T00:00:00.000Z",
      evidence: { games: 0, wins: 0, losses: 0, draws: 0 },
    };
  }
}

async function loadLeague(directory: string, champion: Champion): Promise<LeagueEntry[]> {
  try {
    const saved = JSON.parse(await readFile(resolve(directory, "league.json"), "utf8")) as { entries?: Array<Partial<LeagueEntry>> };
    return uniqueLeagueEntries((saved.entries ?? []).map((entry) => ({
      id: entry.id ?? champion.id,
      generation: entry.generation ?? 0,
      weights: { ...DEFAULT_WEIGHTS, ...entry.weights },
      search: { ...champion.search, ...entry.search },
      promotedAt: entry.promotedAt ?? champion.promotedAt,
    })));
  } catch {
    return [toLeagueEntry(champion)];
  }
}

function toLeagueEntry(champion: Champion): LeagueEntry {
  return {
    id: champion.id,
    generation: champion.generation,
    weights: champion.weights,
    search: champion.search,
    promotedAt: champion.promotedAt,
  };
}

function uniqueLeagueEntries(entries: Array<LeagueEntry | Champion>) {
  const unique = new Map<string, LeagueEntry>();
  for (const entry of entries) unique.set(entry.id, toLeagueEntry(entry as Champion));
  return [...unique.values()];
}

function summarize(records: GameRecord[], targetId: string) {
  let wins = 0;
  let losses = 0;
  let draws = 0;
  for (const record of records) {
    if (!record.winner) { draws += 1; continue; }
    const targetColor = record.agents.light === targetId ? "light" : record.agents.dark === targetId ? "dark" : null;
    if (record.winner === targetColor) wins += 1;
    else losses += 1;
  }
  return { games: records.length, wins, losses, draws };
}

function candidateScore(records: GameRecord[], candidateId: string) {
  const result = summarize(records, candidateId);
  return result.wins - result.losses;
}

function updateRatings(ratings: Map<string, number>, record: GameRecord) {
  const light = record.agents.light;
  const dark = record.agents.dark;
  const lightRating = ratings.get(light) ?? 1_000;
  const darkRating = ratings.get(dark) ?? 1_000;
  const expectedLight = 1 / (1 + 10 ** ((darkRating - lightRating) / 400));
  const actualLight = record.winner === "light" ? 1 : record.winner === "dark" ? 0 : 0.5;
  ratings.set(light, lightRating + 24 * (actualLight - expectedLight));
  ratings.set(dark, darkRating + 24 * ((1 - actualLight) - (1 - expectedLight)));
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

function integer(value: string | undefined, fallback: number) {
  const parsed = Number.parseInt(value ?? "", 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function decimal(value: string | undefined, fallback: number) {
  const parsed = Number.parseFloat(value ?? "");
  return Number.isFinite(parsed) ? parsed : fallback;
}

async function writeJson(path: string, value: unknown) {
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}
