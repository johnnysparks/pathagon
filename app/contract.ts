/** The cross-runtime Pathagon interchange contract. Keep this module free of
 * game-engine imports so it can also validate records before a replay exists. */

export const PATHAGON_CONTRACT_VERSION = 1 as const;
export const PATHAGON_RULES_VERSION = "pathagon-rules-v1" as const;

export type Player = "light" | "dark";
export type TerminationReason = "path" | "threefold-repetition" | "max-plies" | "no-legal-action";

export type GameConfig = {
  rulesVersion: typeof PATHAGON_RULES_VERSION;
  boardSize: number;
  reservePerPlayer: number;
  maxPlies: number;
  repetitionLimit: 3;
};

export type ContractAction =
  | { kind: "place"; to: number }
  | { kind: "relocate"; from: number; to: number };

export type PositionContract = {
  contractVersion: typeof PATHAGON_CONTRACT_VERSION;
  config: GameConfig;
  board: Array<Player | null>;
  reserve: Record<Player, number>;
  turn: Player;
  forbidden: number[];
  lastRelocatedTo: Record<Player, number | null>;
  winner: Player | null;
  ply: number;
};

export type EngineMetadata = {
  id: string;
  runtime: "typescript" | "rust" | "python";
  version: string;
  rulesVersion: typeof PATHAGON_RULES_VERSION;
};

export type AgentSpecification = {
  id: string;
  name: string;
  version: string;
  kind: "random" | "heuristic" | "search" | "learned" | "puct";
  engineId: string;
  parameters?: Record<string, unknown>;
};

export type ContractMove = {
  ply: number;
  player: Player;
  action: ContractAction;
  captured: number[];
  nodes: number;
  completedDepth: number;
  tableHits: number;
  score?: number;
  bookHit?: boolean;
};

export type ContractReplayRecord = {
  contractVersion: typeof PATHAGON_CONTRACT_VERSION;
  seed: number;
  config: GameConfig;
  engine: EngineMetadata;
  agents: Record<Player, string>;
  agentSpecifications: Record<Player, AgentSpecification>;
  winner: Player | null;
  result: "win" | "draw";
  reason: TerminationReason;
  plies: number;
  moves: ContractMove[];
};

export const DEFAULT_GAME_CONFIG: GameConfig = {
  rulesVersion: PATHAGON_RULES_VERSION,
  boardSize: 7,
  reservePerPlayer: 14,
  maxPlies: 180,
  repetitionLimit: 3,
};

export const TYPESCRIPT_ENGINE: EngineMetadata = {
  id: "typescript-reference",
  runtime: "typescript",
  version: "1.0.0",
  rulesVersion: PATHAGON_RULES_VERSION,
};

export function defaultAgentSpecification(id: string, kind: AgentSpecification["kind"], engine = TYPESCRIPT_ENGINE): AgentSpecification {
  return { id, name: id, version: "1.0.0", kind, engineId: engine.id };
}

export function validateGameConfig(value: unknown): GameConfig {
  if (!isRecord(value) || value.rulesVersion !== PATHAGON_RULES_VERSION) throw new Error("Invalid Pathagon rules version");
  const boardSize = integer(value.boardSize, "board size");
  const reservePerPlayer = integer(value.reservePerPlayer, "reserve per player");
  const maxPlies = integer(value.maxPlies, "maximum plies");
  if (boardSize < 3 || boardSize > 8) throw new Error("Board size must be between 3 and 8");
  if (reservePerPlayer < 1 || reservePerPlayer > 64) throw new Error("Reserve must be between 1 and 64");
  if (maxPlies < 1 || maxPlies > 4096) throw new Error("Maximum plies must be between 1 and 4096");
  if (value.repetitionLimit !== 3) throw new Error("Pathagon repetition limit must be 3");
  return { rulesVersion: PATHAGON_RULES_VERSION, boardSize, reservePerPlayer, maxPlies, repetitionLimit: 3 };
}

export function validateContractAction(value: unknown, boardSize = 8): ContractAction {
  if (!isRecord(value) || typeof value.kind !== "string") throw new Error("Invalid contract action");
  if (value.kind === "place" && integerInRange(value.to, 0, boardSize * boardSize - 1)) return { kind: "place", to: Number(value.to) };
  if (value.kind === "relocate" && integerInRange(value.from, 0, boardSize * boardSize - 1) && integerInRange(value.to, 0, boardSize * boardSize - 1)) {
    return { kind: "relocate", from: Number(value.from), to: Number(value.to) };
  }
  throw new Error("Invalid contract action");
}

export function validatePosition(value: unknown): PositionContract {
  if (!isRecord(value) || value.contractVersion !== PATHAGON_CONTRACT_VERSION) throw new Error("Unsupported position contract version");
  const config = validateGameConfig(value.config);
  const cellCount = config.boardSize * config.boardSize;
  if (!Array.isArray(value.board) || value.board.length !== cellCount || value.board.some((piece) => piece !== null && piece !== "light" && piece !== "dark")) throw new Error("Invalid contract board");
  const reserve = validateReserve(value.reserve);
  const turn = validatePlayer(value.turn, "turn");
  const forbidden = validateSquares(value.forbidden, cellCount, "forbidden");
  if (!isRecord(value.lastRelocatedTo)) throw new Error("Invalid last relocation markers");
  const lastRelocatedTo = { light: optionalSquare(value.lastRelocatedTo.light, cellCount), dark: optionalSquare(value.lastRelocatedTo.dark, cellCount) };
  const winner = value.winner === null ? null : validatePlayer(value.winner, "winner");
  const ply = integer(value.ply, "position ply");
  if (ply < 0 || ply > config.maxPlies) throw new Error("Position ply is outside the configured limit");
  return { contractVersion: PATHAGON_CONTRACT_VERSION, config, board: [...value.board] as Array<Player | null>, reserve, turn, forbidden, lastRelocatedTo, winner, ply };
}

export function validateEngineMetadata(value: unknown): EngineMetadata {
  if (!isRecord(value) || !field(value.id) || !field(value.version) || !field(value.runtime) || value.rulesVersion !== PATHAGON_RULES_VERSION) throw new Error("Invalid engine metadata");
  if (value.runtime !== "typescript" && value.runtime !== "rust" && value.runtime !== "python") throw new Error("Invalid engine runtime");
  return { id: value.id, runtime: value.runtime, version: value.version, rulesVersion: PATHAGON_RULES_VERSION };
}

export function validateAgentSpecification(value: unknown): AgentSpecification {
  if (!isRecord(value) || !field(value.id) || typeof value.name !== "string" || value.name.length < 1 || value.name.length > 128 || !field(value.version) || !field(value.engineId)) throw new Error("Invalid agent specification");
  if (!["random", "heuristic", "search", "learned", "puct"].includes(String(value.kind))) throw new Error("Invalid agent kind");
  if (value.parameters !== undefined && !isRecord(value.parameters)) throw new Error("Invalid agent parameters");
  return { id: value.id, name: value.name, version: value.version, kind: value.kind as AgentSpecification["kind"], engineId: value.engineId, ...(value.parameters === undefined ? {} : { parameters: value.parameters }) };
}

export function validateContractReplay(value: unknown): ContractReplayRecord {
  if (!isRecord(value) || value.contractVersion !== PATHAGON_CONTRACT_VERSION) throw new Error("Unsupported replay contract version");
  const config = validateGameConfig(value.config);
  const engine = validateEngineMetadata(value.engine);
  if (!isRecord(value.agents) || !field(value.agents.light) || !field(value.agents.dark)) throw new Error("Invalid replay agents");
  if (!isRecord(value.agentSpecifications)) throw new Error("Missing replay agent specifications");
  const agentSpecifications = { light: validateAgentSpecification(value.agentSpecifications.light), dark: validateAgentSpecification(value.agentSpecifications.dark) };
  const agents = { light: value.agents.light, dark: value.agents.dark };
  if (agents.light !== agentSpecifications.light.id || agents.dark !== agentSpecifications.dark.id) throw new Error("Replay agent ID does not match its specification");
  const winner = value.winner === null ? null : validatePlayer(value.winner, "winner");
  if (value.result !== (winner ? "win" : "draw")) throw new Error("Replay result does not match winner");
  if (!isTerminationReason(value.reason)) throw new Error("Invalid replay termination reason");
  const plies = integer(value.plies, "replay plies");
  if (plies < 0 || plies > config.maxPlies || !Array.isArray(value.moves) || value.moves.length !== plies) throw new Error("Replay plies do not match moves");
  const moves = value.moves.map((move, index) => validateContractMove(move, index, config.boardSize));
  return { contractVersion: 1, seed: integerInRange(value.seed, 0, 4_294_967_295) ? Number(value.seed) : (() => { throw new Error("Invalid replay seed"); })(), config, engine, agents, agentSpecifications, winner, result: value.result as "win" | "draw", reason: value.reason, plies, moves };
}

function validateContractMove(value: unknown, index: number, boardSize: number): ContractMove {
  if (!isRecord(value) || value.ply !== index + 1 || (value.player !== "light" && value.player !== "dark")) throw new Error(`Invalid contract move at ply ${index + 1}`);
  const captured = validateSquares(value.captured, boardSize * boardSize, "captured");
  const move: ContractMove = { ply: index + 1, player: value.player, action: validateContractAction(value.action, boardSize), captured, nodes: nonNegativeInteger(value.nodes, "nodes"), completedDepth: nonNegativeInteger(value.completedDepth, "completed depth"), tableHits: nonNegativeInteger(value.tableHits, "table hits") };
  if (value.score !== undefined) move.score = integer(value.score, "score");
  if (value.bookHit !== undefined) { if (typeof value.bookHit !== "boolean") throw new Error("Invalid book hit"); move.bookHit = value.bookHit; }
  return move;
}

function validateReserve(value: unknown): Record<Player, number> {
  if (!isRecord(value)) throw new Error("Invalid reserve");
  return { light: nonNegativeInteger(value.light, "light reserve"), dark: nonNegativeInteger(value.dark, "dark reserve") };
}
function validateSquares(value: unknown, cellCount: number, label: string) {
  if (!Array.isArray(value) || value.some((square) => !integerInRange(square, 0, cellCount - 1)) || new Set(value).size !== value.length) throw new Error(`Invalid ${label} squares`);
  return value.map(Number);
}
function optionalSquare(value: unknown, cellCount: number) { return value === null ? null : integerInRange(value, 0, cellCount - 1) ? Number(value) : (() => { throw new Error("Invalid relocation square"); })(); }
function validatePlayer(value: unknown, label: string): Player { if (value !== "light" && value !== "dark") throw new Error(`Invalid ${label}`); return value; }
function isTerminationReason(value: unknown): value is TerminationReason { return value === "path" || value === "threefold-repetition" || value === "max-plies" || value === "no-legal-action"; }
function field(value: unknown): value is string { return typeof value === "string" && /^[A-Za-z0-9._:-]{1,128}$/.test(value); }
function integer(value: unknown, label: string) { if (!Number.isSafeInteger(value)) throw new Error(`Invalid ${label}`); return Number(value); }
function nonNegativeInteger(value: unknown, label: string) { const number = integer(value, label); if (number < 0) throw new Error(`Invalid ${label}`); return number; }
function integerInRange(value: unknown, minimum: number, maximum: number): value is number { return Number.isSafeInteger(value) && Number(value) >= minimum && Number(value) <= maximum; }
function isRecord(value: unknown): value is Record<string, any> { return Boolean(value) && typeof value === "object" && !Array.isArray(value); }
