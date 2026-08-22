export type Player = "light" | "dark";

export type Action =
  | { kind: "place"; to: number }
  | { kind: "relocate"; from: number; to: number };

export type ResolvedAction = Action & { player: Player; captured: number[] };

export type GameState = {
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

const BOARD_SIZE = 7;
const CELL_COUNT = BOARD_SIZE * BOARD_SIZE;
const DIRECTIONS = [[-1, 0], [1, 0], [0, -1], [0, 1]] as const;

export function createGame(): GameState {
  return {
    board: Array<Player | null>(CELL_COUNT).fill(null),
    reserve: { light: 14, dark: 14 },
    turn: "light",
    forbidden: [],
    lastRelocatedTo: { light: null, dark: null },
    winner: null,
    winningPath: [],
    lastAction: null,
    ply: 0,
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
  const captured = capturesFrom(board, action.to, player);
  for (const square of captured) board[square] = null;
  reserve[opponent] += captured.length;
  const winningPath = findWinningPath(board, player);
  return { board, reserve, turn: opponent, forbidden: captured, lastRelocatedTo, winner: winningPath.length ? player : null, winningPath, lastAction: { ...action, player, captured }, ply: state.ply + 1 };
}

function capturesFrom(board: Array<Player | null>, origin: number, player: Player) {
  const opponent = otherPlayer(player);
  const row = Math.floor(origin / BOARD_SIZE);
  const column = origin % BOARD_SIZE;
  const captured: number[] = [];
  for (const [rowDelta, columnDelta] of DIRECTIONS) {
    const nearRow = row + rowDelta;
    const nearColumn = column + columnDelta;
    const farRow = row + rowDelta * 2;
    const farColumn = column + columnDelta * 2;
    if (!inBounds(farRow, farColumn)) continue;
    const near = nearRow * BOARD_SIZE + nearColumn;
    const far = farRow * BOARD_SIZE + farColumn;
    if (board[near] === opponent && board[far] === player) captured.push(near);
  }
  return captured;
}

function findWinningPath(board: Array<Player | null>, player: Player) {
  const starts: number[] = [];
  for (let i = 0; i < BOARD_SIZE; i += 1) {
    const square = player === "light" ? (BOARD_SIZE - 1) * BOARD_SIZE + i : i * BOARD_SIZE;
    if (board[square] === player) starts.push(square);
  }
  const queue = [...starts];
  const visited = new Set(starts);
  const parent = new Map<number, number>();
  while (queue.length) {
    const square = queue.shift()!;
    const row = Math.floor(square / BOARD_SIZE);
    const column = square % BOARD_SIZE;
    const reached = player === "light" ? row === 0 : column === BOARD_SIZE - 1;
    if (reached) {
      const path = [square];
      let cursor = square;
      while (parent.has(cursor)) { cursor = parent.get(cursor)!; path.push(cursor); }
      return path.reverse();
    }
    for (const [rowDelta, columnDelta] of DIRECTIONS) {
      const nextRow = row + rowDelta;
      const nextColumn = column + columnDelta;
      if (!inBounds(nextRow, nextColumn)) continue;
      const next = nextRow * BOARD_SIZE + nextColumn;
      if (board[next] !== player || visited.has(next)) continue;
      visited.add(next); parent.set(next, square); queue.push(next);
    }
  }
  return [];
}

function sameAction(left: Action, right: Action) {
  return left.kind === right.kind && left.to === right.to && (left.kind === "place" || (right.kind === "relocate" && left.from === right.from));
}
function inBounds(row: number, column: number) { return row >= 0 && row < BOARD_SIZE && column >= 0 && column < BOARD_SIZE; }
export function otherPlayer(player: Player): Player { return player === "light" ? "dark" : "light"; }
export function playerLabel(player: Player) { return player === "light" ? "light" : "dark"; }
export function describeAction(action: ResolvedAction) {
  const destination = coordinate(action.to);
  const verb = action.kind === "place" ? `placed at ${destination}` : `moved ${coordinate(action.from)} → ${destination}`;
  const capture = action.captured.length ? ` and trapped ${action.captured.length} ${action.captured.length === 1 ? "piece" : "pieces"}` : "";
  return `${playerLabel(action.player)[0].toUpperCase()}${playerLabel(action.player).slice(1)} ${verb}${capture}.`;
}
function coordinate(index: number) { return `${String.fromCharCode(65 + (index % 7))}${Math.floor(index / 7) + 1}`; }
