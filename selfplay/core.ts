import { searchBestAction } from "../app/ai.ts";
import type { EvaluationWeights, SearchConfig } from "../app/ai.ts";
import { applyAction, createGame, legalActions } from "../app/pathagon.ts";
import type { Action, GameState, Player } from "../app/pathagon.ts";
import type { SelfPlayGameRecord, SelfPlayMoveRecord } from "../app/selfplay-record.ts";

export type GameRecord = SelfPlayGameRecord;
export type MoveRecord = SelfPlayMoveRecord;

export type RandomSource = () => number;

export type SelfPlayAgent = {
  id: string;
  chooseAction(state: GameState, random: RandomSource): { action: Action | null; nodes: number; completedDepth?: number; tableHits?: number };
};

export type MatchOptions = {
  seed: number;
  maxPlies: number;
  openingRandomPlies: number;
};

export function createSearchAgent(id: string, config: SearchConfig): SelfPlayAgent {
  return {
    id,
    chooseAction(state) {
      const result = searchBestAction(state, config);
      return { action: result.action, nodes: result.nodes, completedDepth: result.completedDepth, tableHits: result.tableHits };
    },
  };
}

export function createRandomAgent(id = "random"): SelfPlayAgent {
  return {
    id,
    chooseAction(state, random) {
      const actions = legalActions(state);
      return { action: actions.length ? actions[Math.floor(random() * actions.length)] : null, nodes: 1 };
    },
  };
}

export function playGame(light: SelfPlayAgent, dark: SelfPlayAgent, options: MatchOptions): GameRecord {
  const random = mulberry32(options.seed);
  const agents = { light, dark };
  let state = createGame();
  const moves: MoveRecord[] = [];
  const repetitions = new Map<string, number>();
  while (!state.winner && state.ply < options.maxPlies) {
    const key = stateKey(state);
    const repeated = (repetitions.get(key) ?? 0) + 1;
    repetitions.set(key, repeated);
    if (repeated >= 3) return gameRecord(options.seed, agents, null, "threefold-repetition", moves);
    const player = state.turn;
    const legal = legalActions(state);
    if (!legal.length) return gameRecord(options.seed, agents, null, "no-legal-action", moves);
    const decision = state.ply < options.openingRandomPlies
      ? { action: legal[Math.floor(random() * legal.length)], nodes: 1 }
      : agents[player].chooseAction(state, random);
    if (!decision.action) return gameRecord(options.seed, agents, null, "no-legal-action", moves);
    state = applyAction(state, decision.action);
    moves.push({
      ply: state.ply,
      player,
      action: decision.action,
      captured: [...(state.lastAction?.captured ?? [])],
      nodes: decision.nodes,
      completedDepth: decision.completedDepth ?? 0,
      tableHits: decision.tableHits ?? 0,
    });
  }
  if (state.winner) return gameRecord(options.seed, agents, state.winner, "path", moves);
  return gameRecord(options.seed, agents, null, "max-plies", moves);
}

export function mutateWeights(weights: EvaluationWeights, random: RandomSource, scale = 0.2): EvaluationWeights {
  return {
    path: mutateWeight(weights.path, random, scale),
    material: mutateWeight(weights.material, random, scale),
    capture: mutateWeight(weights.capture, random, scale),
    structure: mutateWeight(weights.structure, random, scale),
    threat: mutateWeight(weights.threat, random, scale),
    edge: mutateWeight(weights.edge, random, scale),
  };
}

export function mulberry32(seed: number): RandomSource {
  let value = seed >>> 0;
  return () => {
    value += 0x6D2B79F5;
    let result = value;
    result = Math.imul(result ^ (result >>> 15), result | 1);
    result ^= result + Math.imul(result ^ (result >>> 7), result | 61);
    return ((result ^ (result >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function mutateWeight(value: number, random: RandomSource, scale: number) {
  const multiplier = 1 + (random() * 2 - 1) * scale;
  return Math.max(1, Math.round(value * multiplier));
}

function stateKey(state: GameState) {
  const board = state.board.map((piece) => piece === "light" ? "L" : piece === "dark" ? "D" : ".").join("");
  return [board, state.turn, state.reserve.light, state.reserve.dark, state.forbidden.join(","), state.lastRelocatedTo.light, state.lastRelocatedTo.dark].join("|");
}

function gameRecord(
  seed: number,
  agents: Record<Player, SelfPlayAgent>,
  winner: Player | null,
  reason: GameRecord["reason"],
  moves: MoveRecord[],
): GameRecord {
  return {
    schemaVersion: 2,
    seed,
    agents: { light: agents.light.id, dark: agents.dark.id },
    winner,
    result: winner ? "win" : "draw",
    reason,
    plies: moves.length,
    moves,
  };
}
