import { DEFAULT_GAME_CONFIG } from "./contract.ts";
import type { ContractAction, GameConfig, Player as ContractPlayer, PositionContract } from "./contract.ts";

export type Player = ContractPlayer;

export type Action = ContractAction;

export type ResolvedAction = Action & { player: Player; captured: number[] };

export type GameState = {
  config: GameConfig;
  board: Array<Player | null>;
  reserve: Record<Player, number>;
  turn: Player;
  forbidden: number[];
  lastRelocatedTo: Record<Player, number | null>;
  winner: Player | null;
  winningPath: number[];
  lastAction: ResolvedAction | null;
  ply: number;
};

const DIRECTIONS = [[-1, 0], [1, 0], [0, -1], [0, 1]] as const;

export function createGame(config: GameConfig = DEFAULT_GAME_CONFIG): GameState {
  const cellCount = config.boardSize * config.boardSize;
  return {
    config,
    board: Array<Player | null>(cellCount).fill(null),
    reserve: { light: config.reservePerPlayer, dark: config.reservePerPlayer },
    turn: "light",
    forbidden: [],
    lastRelocatedTo: { light: null, dark: null },
    winner: null,
    winningPath: [],
    lastAction: null,
    ply: 0,
  };
}

export function createGameFromPosition(position: PositionContract): GameState {
  return {
    config: position.config,
    board: [...position.board],
    reserve: { ...position.reserve },
    turn: position.turn,
    forbidden: [...position.forbidden],
    lastRelocatedTo: { ...position.lastRelocatedTo },
    winner: position.winner,
    winningPath: [],
    lastAction: null,
    ply: position.ply,
  };
}

export function createNearWinFixture(): GameState {
  const state = createGame();
  for (const square of [42, 35, 28, 21, 14, 7]) state.board[square] = "light";
  state.reserve.light = 8;
  return state;
}

export function legalActions(state: GameState): Action[] {
  if (state.winner) return [];
  const forbidden = new Set(state.forbidden);
  const destinations = state.board.map((piece, index) => ({ piece, index }))
    .filter(({ piece, index }) => piece === null && !forbidden.has(index)).map(({ index }) => index);
  if (state.reserve[state.turn] > 0) return destinations.map((to) => ({ kind: "place" as const, to }));
  const sources = state.board.map((piece, index) => ({ piece, index }))
    .filter(({ piece, index }) => piece === state.turn && state.lastRelocatedTo[state.turn] !== index).map(({ index }) => index);
  return sources.flatMap((from) => destinations.map((to) => ({ kind: "relocate" as const, from, to })));
}

export function applyAction(state: GameState, action: Action): GameState {
  // A turn must change the board. Relocation is never a pick-up-and-replace pass.
  if (action.kind === "relocate" && action.from === action.to) {
    throw new Error("A relocated piece must move to a different square");
  }
  if (!legalActions(state).some((candidate) => sameAction(candidate, action))) throw new Error("Illegal Pathagon action");
  return applyLegalAction(state, action);
}

// Search and self-play enumerate legal actions before applying them. Keeping that
// hot path separate avoids rebuilding the entire legal move list at every node.
export function applyLegalAction(state: GameState, action: Action): GameState {
  const player = state.turn;
  const opponent = otherPlayer(player);
  const board = [...state.board];
  const reserve = { ...state.reserve };
  const lastRelocatedTo = { ...state.lastRelocatedTo };
  if (action.kind === "place") {
    reserve[player] -= 1;
    lastRelocatedTo[player] = null;
  } else {
    board[action.from] = null;
    lastRelocatedTo[player] = action.to;
  }
  board[action.to] = player;
  const captured = capturesFrom(board, action.to, player, state.config.boardSize);
  for (const square of captured) board[square] = null;
  reserve[opponent] += captured.length;
  const winningPath = findWinningPath(board, player, state.config.boardSize);
  return { config: state.config, board, reserve, turn: opponent, forbidden: captured, lastRelocatedTo, winner: winningPath.length ? player : null, winningPath, lastAction: { ...action, player, captured }, ply: state.ply + 1 };
}

function capturesFrom(board: Array<Player | null>, origin: number, player: Player, boardSize: number) {
  const opponent = otherPlayer(player);
  const row = Math.floor(origin / boardSize);
  const column = origin % boardSize;
  const captured: number[] = [];
  for (const [rowDelta, columnDelta] of DIRECTIONS) {
    const nearRow = row + rowDelta;
    const nearColumn = column + columnDelta;
    const farRow = row + rowDelta * 2;
    const farColumn = column + columnDelta * 2;
    if (!inBounds(farRow, farColumn, boardSize)) continue;
    const near = nearRow * boardSize + nearColumn;
    const far = farRow * boardSize + farColumn;
    if (board[near] === opponent && board[far] === player) captured.push(near);
  }
  return captured;
}

function findWinningPath(board: Array<Player | null>, player: Player, boardSize: number) {
  const starts: number[] = [];
  for (let i = 0; i < boardSize; i += 1) {
    const square = player === "light" ? (boardSize - 1) * boardSize + i : i * boardSize;
    if (board[square] === player) starts.push(square);
  }
  const queue = [...starts];
  const visited = new Set(starts);
  const parent = new Map<number, number>();
  while (queue.length) {
    const square = queue.shift()!;
    const row = Math.floor(square / boardSize);
    const column = square % boardSize;
    const reached = player === "light" ? row === 0 : column === boardSize - 1;
    if (reached) {
      const path = [square];
      let cursor = square;
      while (parent.has(cursor)) { cursor = parent.get(cursor)!; path.push(cursor); }
      return path.reverse();
    }
    for (const [rowDelta, columnDelta] of DIRECTIONS) {
      const nextRow = row + rowDelta;
      const nextColumn = column + columnDelta;
      if (!inBounds(nextRow, nextColumn, boardSize)) continue;
      const next = nextRow * boardSize + nextColumn;
      if (board[next] !== player || visited.has(next)) continue;
      visited.add(next); parent.set(next, square); queue.push(next);
    }
  }
  return [];
}

function sameAction(left: Action, right: Action) {
  return left.kind === right.kind && left.to === right.to && (left.kind === "place" || (right.kind === "relocate" && left.from === right.from));
}
function inBounds(row: number, column: number, boardSize: number) { return row >= 0 && row < boardSize && column >= 0 && column < boardSize; }
export function otherPlayer(player: Player): Player { return player === "light" ? "dark" : "light"; }
export function playerLabel(player: Player) { return player === "light" ? "light" : "dark"; }
export function describeAction(action: ResolvedAction) {
  const destination = coordinate(action.to);
  const verb = action.kind === "place" ? `placed at ${destination}` : `moved ${coordinate(action.from)} → ${destination}`;
  const capture = action.captured.length ? ` and trapped ${action.captured.length} ${action.captured.length === 1 ? "piece" : "pieces"}` : "";
  return `${playerLabel(action.player)[0].toUpperCase()}${playerLabel(action.player).slice(1)} ${verb}${capture}.`;
}
function coordinate(index: number) { return `${String.fromCharCode(65 + (index % 7))}${Math.floor(index / 7) + 1}`; }
