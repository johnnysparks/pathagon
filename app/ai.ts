import { applyAction, legalActions, otherPlayer } from "./pathagon.ts";
import type { Action, GameState, Player } from "./pathagon.ts";

export type EvaluationWeights = {
  path: number;
  material: number;
  capture: number;
};

export type SearchConfig = {
  depth: number;
  maxNodes: number;
  beamWidth: number;
  weights: EvaluationWeights;
};

export type SearchResult = {
  action: Action | null;
  score: number;
  nodes: number;
  exhausted: boolean;
};

export const DEFAULT_WEIGHTS: EvaluationWeights = {
  path: 240,
  material: 110,
  capture: 700,
};

export const SURVEYOR_SEARCH: SearchConfig = {
  depth: 2,
  maxNodes: 12_000,
  beamWidth: 64,
  weights: DEFAULT_WEIGHTS,
};

export function searchBestAction(state: GameState, config: SearchConfig): SearchResult {
  const rootPlayer = state.turn;
  const actions = orderActions(state, rootPlayer, config.weights);
  if (!actions.length) return { action: null, score: evaluatePosition(state, rootPlayer, config.weights), nodes: 0, exhausted: false };
  const budget = { nodes: 0, exhausted: false };
  let bestAction = actions[0];
  let bestScore = Number.NEGATIVE_INFINITY;
  for (const action of actions) {
    const next = applyAction(state, action);
    budget.nodes += 1;
    const score = minimax(next, rootPlayer, config.depth - 1, Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY, config, budget);
    if (score > bestScore || (score === bestScore && actionOrder(action) < actionOrder(bestAction))) {
      bestAction = action;
      bestScore = score;
    }
  }
  return { action: bestAction, score: bestScore, nodes: budget.nodes, exhausted: budget.exhausted };
}

export function evaluatePosition(state: GameState, player: Player, weights: EvaluationWeights) {
  if (state.winner === player) return 1_000_000_000 - state.ply;
  const opponent = otherPlayer(player);
  if (state.winner === opponent) return -1_000_000_000 + state.ply;
  const ownDistance = connectionDistance(state.board, player);
  const opponentDistance = connectionDistance(state.board, opponent);
  const ownPieces = state.board.filter((piece) => piece === player).length;
  const opponentPieces = state.board.filter((piece) => piece === opponent).length;
  const lastCapture = state.lastAction?.captured.length ?? 0;
  const captureDirection = state.lastAction?.player === player ? 1 : -1;
  return (opponentDistance - ownDistance) * weights.path
    + (ownPieces - opponentPieces) * weights.material
    + captureDirection * lastCapture * weights.capture;
}

function minimax(
  state: GameState,
  rootPlayer: Player,
  depth: number,
  alpha: number,
  beta: number,
  config: SearchConfig,
  budget: { nodes: number; exhausted: boolean },
): number {
  if (state.winner || depth <= 0) return evaluatePosition(state, rootPlayer, config.weights);
  if (budget.nodes >= config.maxNodes) {
    budget.exhausted = true;
    return evaluatePosition(state, rootPlayer, config.weights);
  }
  const maximizing = state.turn === rootPlayer;
  const actions = orderActions(state, rootPlayer, config.weights).slice(0, config.beamWidth);
  if (!actions.length) return evaluatePosition(state, rootPlayer, config.weights);
  let best = maximizing ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY;
  for (const action of actions) {
    const next = applyAction(state, action);
    budget.nodes += 1;
    const score = minimax(next, rootPlayer, depth - 1, alpha, beta, config, budget);
    if (maximizing) {
      best = Math.max(best, score);
      alpha = Math.max(alpha, best);
    } else {
      best = Math.min(best, score);
      beta = Math.min(beta, best);
    }
    if (beta <= alpha || budget.nodes >= config.maxNodes) break;
  }
  return best;
}

function orderActions(state: GameState, rootPlayer: Player, weights: EvaluationWeights) {
  const maximizing = state.turn === rootPlayer;
  return legalActions(state)
    .map((action) => {
      const next = applyAction(state, action);
      const tactical = next.winner === state.turn ? 2_000_000_000 : (next.lastAction?.captured.length ?? 0) * 10_000;
      const score = tactical + evaluatePosition(next, rootPlayer, weights);
      return { action, score };
    })
    .sort((left, right) => {
      const difference = maximizing ? right.score - left.score : left.score - right.score;
      return difference || actionOrder(left.action) - actionOrder(right.action);
    })
    .map(({ action }) => action);
}

// Dijkstra over the board: our pieces cost 0, empty cells cost 1,
// and opposing pieces are walls. Lower distance means a more complete path.
export function connectionDistance(board: GameState["board"], player: Player) {
  const distances = Array<number>(49).fill(Number.POSITIVE_INFINITY);
  const frontier: Array<{ square: number; distance: number }> = [];
  for (let i = 0; i < 7; i += 1) {
    const square = player === "light" ? 42 + i : i * 7;
    if (board[square] === otherPlayer(player)) continue;
    const distance = board[square] === player ? 0 : 1;
    distances[square] = distance;
    frontier.push({ square, distance });
  }
  while (frontier.length) {
    frontier.sort((left, right) => left.distance - right.distance);
    const current = frontier.shift()!;
    if (current.distance !== distances[current.square]) continue;
    const row = Math.floor(current.square / 7);
    const column = current.square % 7;
    if ((player === "light" && row === 0) || (player === "dark" && column === 6)) return current.distance;
    for (const [rowDelta, columnDelta] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
      const nextRow = row + rowDelta;
      const nextColumn = column + columnDelta;
      if (nextRow < 0 || nextRow >= 7 || nextColumn < 0 || nextColumn >= 7) continue;
      const next = nextRow * 7 + nextColumn;
      if (board[next] === otherPlayer(player)) continue;
      const distance = current.distance + (board[next] === player ? 0 : 1);
      if (distance >= distances[next]) continue;
      distances[next] = distance;
      frontier.push({ square: next, distance });
    }
  }
  return 49;
}

function actionOrder(action: Action) {
  return action.kind === "place" ? action.to : action.from * 49 + action.to;
}
