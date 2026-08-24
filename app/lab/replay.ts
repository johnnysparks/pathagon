export type ReplayPlayer = "light" | "dark";

export type ReplayAction =
  | { kind: "place"; to: number }
  | { kind: "relocate"; from: number; to: number };

export type ReplayMove = {
  ply: number;
  player: ReplayPlayer;
  action: ReplayAction;
  captured: number[];
};

export type ReplayGame = {
  schemaVersion: number;
  seed: number;
  boardSize: number;
  reservePerPlayer: number;
  agents: Record<ReplayPlayer, string>;
  winner: ReplayPlayer | null;
  result: "win" | "draw";
  reason: string;
  plies: number;
  moves: ReplayMove[];
};

export type ReplayPosition = {
  board: Array<ReplayPlayer | null>;
  reserve: Record<ReplayPlayer, number>;
  turn: ReplayPlayer;
  forbidden: number[];
  lastRelocatedTo: Record<ReplayPlayer, number | null>;
  winner: ReplayPlayer | null;
  winningPath: number[];
  lastMove: ReplayMove | null;
  ply: number;
};

export function parseReplayArchive(text: string, fallbackBoardSize: number, fallbackReserve: number): ReplayGame[] {
  return text.split("\n").filter((line) => line.trim()).map((line) => {
    const raw = JSON.parse(line) as Record<string, unknown>;
    const moves = Array.isArray(raw.moves) ? raw.moves.map(normalizeMove) : [];
    const config = raw.config && typeof raw.config === "object" ? raw.config as Record<string, unknown> : {};
    return {
      schemaVersion: Number(raw.schemaVersion ?? raw.contractVersion ?? 2),
      seed: Number(raw.seed),
      boardSize: Number(config.boardSize ?? raw.boardSize ?? fallbackBoardSize),
      reservePerPlayer: Number(config.reservePerPlayer ?? raw.reservePerPlayer ?? fallbackReserve),
      agents: (raw.agents ?? { light: "unknown", dark: "unknown" }) as Record<ReplayPlayer, string>,
      winner: raw.winner === "light" || raw.winner === "dark" ? raw.winner : null,
      result: raw.result === "win" ? "win" : "draw",
      reason: String(raw.reason ?? "unknown"),
      plies: Number(raw.plies ?? moves.length),
      moves,
    };
  });
}

export function buildReplayPositions(game: ReplayGame): ReplayPosition[] {
  const initial: ReplayPosition = {
    board: Array<ReplayPlayer | null>(game.boardSize * game.boardSize).fill(null),
    reserve: { light: game.reservePerPlayer, dark: game.reservePerPlayer },
    turn: "light",
    forbidden: [],
    lastRelocatedTo: { light: null, dark: null },
    winner: null,
    winningPath: [],
    lastMove: null,
    ply: 0,
  };
  const positions = [initial];
  let position = initial;
  for (const move of game.moves) {
    const board = [...position.board];
    const reserve = { ...position.reserve };
    const lastRelocatedTo = { ...position.lastRelocatedTo };
    if (move.action.kind === "place") {
      reserve[move.player] -= 1;
      lastRelocatedTo[move.player] = null;
    } else {
      board[move.action.from] = null;
      lastRelocatedTo[move.player] = move.action.to;
    }
    board[move.action.to] = move.player;
    for (const square of move.captured) {
      board[square] = null;
      reserve[otherPlayer(move.player)] += 1;
    }
    const winningPath = findWinningPath(board, game.boardSize, move.player);
    position = {
      board,
      reserve,
      turn: otherPlayer(move.player),
      forbidden: [...move.captured],
      lastRelocatedTo,
      winner: winningPath.length ? move.player : null,
      winningPath,
      lastMove: move,
      ply: move.ply,
    };
    positions.push(position);
  }
  return positions;
}

export function formatReplayAction(action: ReplayAction, boardSize: number) {
  return action.kind === "place"
    ? coordinate(action.to, boardSize)
    : `${coordinate(action.from, boardSize)} → ${coordinate(action.to, boardSize)}`;
}

export function coordinate(square: number, boardSize: number) {
  return `${String.fromCharCode(65 + (square % boardSize))}${Math.floor(square / boardSize) + 1}`;
}

function normalizeMove(raw: unknown): ReplayMove {
  const value = raw as Record<string, unknown>;
  const action = value.action as Record<string, unknown>;
  const normalizedAction: ReplayAction = action.kind === "relocate"
    ? { kind: "relocate", from: Number(action.from), to: Number(action.to) }
    : { kind: "place", to: Number(action.to) };
  return {
    ply: Number(value.ply),
    player: value.player === "dark" ? "dark" : "light",
    action: normalizedAction,
    captured: Array.isArray(value.captured) ? value.captured.map(Number) : [],
  };
}

function otherPlayer(player: ReplayPlayer): ReplayPlayer {
  return player === "light" ? "dark" : "light";
}

function findWinningPath(board: Array<ReplayPlayer | null>, size: number, player: ReplayPlayer) {
  const starts = player === "light"
    ? Array.from({ length: size }, (_, index) => (size - 1) * size + index)
    : Array.from({ length: size }, (_, index) => index * size);
  const queue = starts.filter((square) => board[square] === player);
  const visited = new Set(queue);
  const parent = new Map<number, number>();
  while (queue.length) {
    const square = queue.shift()!;
    const row = Math.floor(square / size);
    const column = square % size;
    const reached = player === "light" ? row === 0 : column === size - 1;
    if (reached) {
      const path = [square];
      let cursor = square;
      while (parent.has(cursor)) {
        cursor = parent.get(cursor)!;
        path.push(cursor);
      }
      return path.reverse();
    }
    for (const neighbor of neighbors(square, size)) {
      if (board[neighbor] !== player || visited.has(neighbor)) continue;
      visited.add(neighbor);
      parent.set(neighbor, square);
      queue.push(neighbor);
    }
  }
  return [];
}

function neighbors(square: number, size: number) {
  const row = Math.floor(square / size);
  const column = square % size;
  const result: number[] = [];
  if (row > 0) result.push(square - size);
  if (row + 1 < size) result.push(square + size);
  if (column > 0) result.push(square - 1);
  if (column + 1 < size) result.push(square + 1);
  return result;
}
