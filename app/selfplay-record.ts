import { applyAction, createGame, legalActions } from "./pathagon.ts";
import type { Action, Player } from "./pathagon.ts";
import {
  DEFAULT_GAME_CONFIG,
  PATHAGON_CONTRACT_VERSION,
  TYPESCRIPT_ENGINE,
  defaultAgentManifest,
  defaultAgentSpecification,
  validateAgentSpecification,
  validateContractReplay,
  validateEngineMetadata,
  validateGameConfig,
} from "./contract.ts";
import type { AgentSpecification, ContractReplayRecord, EngineMetadata, GameConfig } from "./contract.ts";

export const SELF_PLAY_SCHEMA_VERSION = 2;
export const SELF_PLAY_MAX_PLIES = 512;

export type SelfPlayMoveRecord = ContractReplayRecord["moves"][number];
export type SelfPlayGameRecord = ContractReplayRecord;

const TERMINATION_REASONS = new Set<SelfPlayGameRecord["reason"]>([
  "path",
  "threefold-repetition",
  "max-plies",
  "no-legal-action",
]);

export function validateSelfPlayRecord(value: unknown): SelfPlayGameRecord {
  if (!value || typeof value !== "object") throw new Error("Self-play record must be an object");
  const input = value as Record<string, unknown>;
  if (input.contractVersion !== PATHAGON_CONTRACT_VERSION && input.schemaVersion !== SELF_PLAY_SCHEMA_VERSION) throw new Error("Unsupported self-play contract version");
  if (!Number.isSafeInteger(input.seed) || Number(input.seed) < 0 || Number(input.seed) > 4_294_967_295) {
    throw new Error("Invalid self-play seed");
  }
  const agents = validateAgents(input.agents);
  const plies = input.plies;
  const config = normalizeConfig(input.config, input.boardSize, input.reservePerPlayer, plies);
  const engine = normalizeEngine(input.engine);
  const agentSpecifications = normalizeAgentSpecifications(input.agentSpecifications, agents, engine);
  const winner = input.winner === null ? null : validatePlayer(input.winner, "winner");
  if (input.result !== (winner ? "win" : "draw")) throw new Error("Self-play result does not match winner");
  if (typeof input.reason !== "string" || !TERMINATION_REASONS.has(input.reason as SelfPlayGameRecord["reason"])) {
    throw new Error("Invalid self-play termination reason");
  }
  if (!Number.isInteger(plies) || Number(plies) < 0 || Number(plies) > Math.min(SELF_PLAY_MAX_PLIES, config.maxPlies)) {
    throw new Error("Invalid self-play ply count");
  }
  if (!Array.isArray(input.moves) || input.moves.length !== plies) {
    throw new Error("Self-play plies do not match moves");
  }

  let state = createGame(config);
  const moves = input.moves.map((candidate, index) => {
    const move = validateMove(candidate, index, state.turn, config.boardSize * config.boardSize);
    let next: ReturnType<typeof applyAction>;
    try {
      next = applyAction(state, move.action);
    } catch {
      throw new Error(`Illegal self-play action at ply ${index + 1}`);
    }
    if (!sameNumbers(move.captured, next.lastAction?.captured ?? [])) {
      throw new Error(`Self-play capture data does not match replay at ply ${index + 1}`);
    }
    state = next;
    return move;
  });

  if (state.winner !== winner) throw new Error("Self-play winner does not match replay");
  if (input.reason === "path" && !winner) throw new Error("Path termination requires a winner");
  if (input.reason !== "path" && winner) throw new Error("Only path termination may have a winner");
  if (input.reason === "no-legal-action" && legalActions(state).length > 0) {
    throw new Error("No-legal-action termination is not proved by replay");
  }

  const normalized: SelfPlayGameRecord = {
    contractVersion: PATHAGON_CONTRACT_VERSION,
    seed: Number(input.seed),
    config,
    engine,
    agents,
    agentSpecifications,
    winner,
    result: winner ? "win" : "draw",
    reason: input.reason as SelfPlayGameRecord["reason"],
    plies: Number(plies),
    moves,
  };
  return validateContractReplay(normalized);
}

function normalizeConfig(value: unknown, boardSize: unknown, reserve: unknown, plies: unknown): GameConfig {
  if (value !== undefined) return validateGameConfig(value);
  const size = Number.isInteger(boardSize) ? Number(boardSize) : DEFAULT_GAME_CONFIG.boardSize;
  const pieces = Number.isInteger(reserve) ? Number(reserve) : DEFAULT_GAME_CONFIG.reservePerPlayer;
  const maxPlies = Math.max(DEFAULT_GAME_CONFIG.maxPlies, Number.isInteger(plies) ? Number(plies) : 0);
  return validateGameConfig({ ...DEFAULT_GAME_CONFIG, boardSize: size, reservePerPlayer: pieces, maxPlies });
}

function normalizeEngine(value: unknown): EngineMetadata {
  if (value && typeof value === "object") return validateEngineMetadata(value);
  if (typeof value === "string") {
    const runtime = value.includes("python") ? "python" : value.includes("rust") ? "rust" : "typescript";
    return validateEngineMetadata({ ...TYPESCRIPT_ENGINE, id: value, runtime });
  }
  return TYPESCRIPT_ENGINE;
}

function normalizeAgentSpecifications(value: unknown, agents: Record<Player, string>, engine: EngineMetadata): Record<Player, AgentSpecification> {
  if (value && typeof value === "object") {
    const input = value as Record<string, unknown>;
    const light = validateAgentSpecification(normalizeAgentSpecification(input.light, engine));
    const dark = validateAgentSpecification(normalizeAgentSpecification(input.dark, engine));
    if (light.id !== agents.light || dark.id !== agents.dark) throw new Error("Replay agent ID does not match specification");
    return { light, dark };
  }
  return {
    light: defaultAgentSpecification(agents.light, "search", engine),
    dark: defaultAgentSpecification(agents.dark, "search", engine),
  };
}

function normalizeAgentSpecification(value: unknown, engine: EngineMetadata): unknown {
  if (!value || typeof value !== "object" || Array.isArray(value)) return value;
  const input = value as Record<string, unknown>;
  return {
    ...input,
    manifest: input.manifest ?? defaultAgentManifest(engine),
  };
}

function validateAgents(value: unknown): Record<Player, string> {
  if (!value || typeof value !== "object") throw new Error("Invalid self-play agents");
  const input = value as Record<string, unknown>;
  return {
    light: validateAgentId(input.light),
    dark: validateAgentId(input.dark),
  };
}

function validateAgentId(value: unknown) {
  if (typeof value !== "string" || !/^[a-zA-Z0-9._:-]{1,128}$/.test(value)) {
    throw new Error("Invalid self-play agent ID");
  }
  return value;
}

function validateMove(value: unknown, index: number, expectedPlayer: Player, cellCount: number): SelfPlayMoveRecord {
  if (!value || typeof value !== "object") throw new Error(`Invalid self-play move at ply ${index + 1}`);
  const input = value as Record<string, unknown>;
  if (input.ply !== index + 1) throw new Error(`Invalid self-play move number at ply ${index + 1}`);
  if (input.player !== expectedPlayer) throw new Error(`Invalid self-play player at ply ${index + 1}`);
  if (!Array.isArray(input.captured) || input.captured.some((square) => !Number.isInteger(square) || Number(square) < 0 || Number(square) >= cellCount)) {
    throw new Error(`Invalid self-play captures at ply ${index + 1}`);
  }
  if (new Set(input.captured as number[]).size !== input.captured.length) {
    throw new Error(`Duplicate self-play capture at ply ${index + 1}`);
  }
  const action = validateAction(input.action, cellCount);
  const move: SelfPlayMoveRecord = {
    ply: index + 1,
    player: expectedPlayer,
    action,
    captured: [...(input.captured as number[])].map(Number),
    nodes: validateCounter(input.nodes, "nodes", index),
    completedDepth: validateCounter(input.completedDepth, "completed depth", index),
    tableHits: validateCounter(input.tableHits, "table hits", index),
  };
  if (input.score !== undefined) move.score = validateInteger(input.score, "score", index, "Invalid self-play score", false);
  if (input.bookHit !== undefined) {
    if (typeof input.bookHit !== "boolean") throw new Error(`Invalid self-play book hit at ply ${index + 1}`);
    move.bookHit = input.bookHit;
  }
  return move;
}

function validateAction(value: unknown, cellCount: number): Action {
  if (!value || typeof value !== "object") throw new Error("Invalid self-play action");
  const input = value as Record<string, unknown>;
  if (input.kind === "place" && Number.isInteger(input.to) && Number(input.to) >= 0 && Number(input.to) < cellCount) {
    return { kind: "place", to: Number(input.to) };
  }
  if (input.kind === "relocate" && Number.isInteger(input.from) && Number(input.from) >= 0 && Number(input.from) < cellCount && Number.isInteger(input.to) && Number(input.to) >= 0 && Number(input.to) < cellCount) {
    return { kind: "relocate", from: Number(input.from), to: Number(input.to) };
  }
  throw new Error("Invalid self-play action");
}

function validatePlayer(value: unknown, label: string): Player {
  if (value !== "light" && value !== "dark") throw new Error(`Invalid self-play ${label}`);
  return value;
}

function validateCounter(value: unknown, label: string, index: number) {
  return validateInteger(value, label, index, `Invalid self-play ${label} at ply ${index + 1}`);
}

function validateInteger(value: unknown, label: string, _index: number, message = `Invalid self-play ${label}`, nonNegative = true) {
  if (!Number.isSafeInteger(value) || (nonNegative && Number(value) < 0)) throw new Error(message);
  return Number(value);
}

function sameNumbers(left: number[], right: number[]) {
  if (left.length !== right.length) return false;
  const sortedLeft = [...left].sort((a, b) => a - b);
  const sortedRight = [...right].sort((a, b) => a - b);
  return sortedLeft.every((value, index) => value === sortedRight[index]);
}
