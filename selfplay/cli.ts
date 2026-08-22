import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { DEFAULT_WEIGHTS, SURVEYOR_SEARCH } from "../app/ai.ts";
import type { EvaluationWeights, SearchConfig } from "../app/ai.ts";
import { createRandomAgent, createSearchAgent, mulberry32, mutateWeights, playGame } from "./core.ts";
import type { GameRecord } from "./core.ts";

type Champion = {
  schemaVersion: 1;
  id: string;
  generation: number;
  weights: EvaluationWeights;
  search: Omit<SearchConfig, "weights">;
  promotedAt: string;
  evidence: { games: number; wins: number; losses: number; draws: number };
};

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
} else {
  await arena();
}

async function arena() {
  const champion = await loadChampion(progressDir);
  const search = createSearchAgent(champion.id, { ...champion.search, weights: champion.weights });
  const random = createRandomAgent("coin-flip-seeded");
  const records: GameRecord[] = [];
  for (let game = 0; game < games; game += 1) {
    const searchIsLight = game % 2 === 0;
    records.push(playGame(
      searchIsLight ? search : random,
      searchIsLight ? random : search,
      { seed: seed + game, maxPlies, openingRandomPlies },
    ));
  }
  const summary = summarize(records, champion.id);
  const output = { schemaVersion: 1, mode: "arena", seed, champion, summary, games: records };
  const path = resolve(progressDir, "runs", `arena-${seed}.json`);
  await writeJson(path, output);
  await writeJson(resolve(progressDir, "latest.json"), { mode: "arena", path, summary });
  console.log(JSON.stringify({ path, summary }, null, 2));
}

async function train() {
  let champion = await loadChampion(progressDir);
  const generations = integer(args.generations, 1);
  const population = integer(args.population, 4);
  const random = mulberry32(seed);
  const history: unknown[] = [];
  for (let generation = 0; generation < generations; generation += 1) {
    let best = { id: champion.id, weights: champion.weights, score: Number.NEGATIVE_INFINITY, records: [] as GameRecord[] };
    for (let candidateIndex = 0; candidateIndex < population; candidateIndex += 1) {
      const weights = mutateWeights(champion.weights, random);
      const challenger = createSearchAgent(`candidate-${champion.generation + 1}-${candidateIndex}`, { ...champion.search, weights });
      const incumbent = createSearchAgent(champion.id, { ...champion.search, weights: champion.weights });
      const records: GameRecord[] = [];
      for (let game = 0; game < games; game += 1) {
        const challengerIsLight = game % 2 === 0;
        records.push(playGame(
          challengerIsLight ? challenger : incumbent,
          challengerIsLight ? incumbent : challenger,
          { seed: seed + generation * 10_000 + candidateIndex * 1_000 + Math.floor(game / 2), maxPlies, openingRandomPlies },
        ));
      }
      const score = candidateScore(records, challenger.id);
      history.push({ generation: champion.generation + 1, candidate: challenger.id, weights, score, summary: summarize(records, challenger.id) });
      if (score > best.score) best = { id: challenger.id, weights, score, records };
    }
    if (best.score > 0) {
      const evidence = summarize(best.records, best.id);
      champion = {
        ...champion,
        id: `surveyor-trained-g${champion.generation + 1}`,
        generation: champion.generation + 1,
        weights: best.weights,
        promotedAt: new Date().toISOString(),
        evidence,
      };
    }
  }
  const path = resolve(progressDir, "runs", `train-${seed}.json`);
  await writeJson(path, { schemaVersion: 1, mode: "train", seed, champion, history });
  await writeJson(resolve(progressDir, "champion.json"), champion);
  await writeJson(resolve(progressDir, "latest.json"), { mode: "train", path, champion });
  console.log(JSON.stringify({ path, champion, trials: history.length }, null, 2));
}

async function loadChampion(directory: string): Promise<Champion> {
  try {
    return JSON.parse(await readFile(resolve(directory, "champion.json"), "utf8"));
  } catch {
    return {
      schemaVersion: 1,
      id: "surveyor-handcrafted-v0.2.0",
      generation: 0,
      weights: DEFAULT_WEIGHTS,
      search: { depth: SURVEYOR_SEARCH.depth, maxNodes: SURVEYOR_SEARCH.maxNodes, beamWidth: SURVEYOR_SEARCH.beamWidth },
      promotedAt: "2026-08-22T00:00:00.000Z",
      evidence: { games: 0, wins: 0, losses: 0, draws: 0 },
    };
  }
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

async function writeJson(path: string, value: unknown) {
  await writeFile(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}
