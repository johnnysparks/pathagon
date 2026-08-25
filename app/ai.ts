import { applyLegalAction, legalActions, otherPlayer } from "./pathagon.ts";
import type { Action, GameState, Player } from "./pathagon.ts";
import { DEFAULT_EVALUATOR_WEIGHTS } from "./contract.ts";
import type { EvaluatorWeights } from "./contract.ts";

export type EvaluationWeights = EvaluatorWeights;

export type SearchConfig = {
  depth: number;
  maxNodes: number;
  beamWidth: number;
  weights: EvaluationWeights;
  /** Rust-only opt-in proof search for 4x4-or-smaller tactical positions. */
  tacticalProofHorizon?: number;
};

export type SearchResult = {
  action: Action | null;
  score: number;
  nodes: number;
  exhausted: boolean;
  completedDepth: number;
  tableHits: number;
};

export type MoveEvaluation = {
  action: Action;
  beforeScore: number;
  score: number;
  delta: number;
  nodes: number;
  exhausted: boolean;
  completedDepth: number;
  tableHits: number;
};

type Budget = { nodes: number; exhausted: boolean; tableHits: number };
type TableEntry = { depth: number; score: number; flag: "exact" | "lower" | "upper" };

export const DEFAULT_WEIGHTS: EvaluationWeights = DEFAULT_EVALUATOR_WEIGHTS;

export const SURVEYOR_SEARCH: SearchConfig = {
  depth: 2,
  maxNodes: 12_000,
  beamWidth: 64,
  weights: DEFAULT_WEIGHTS,
};

export const PATHFINDER_SEARCH: SearchConfig = {
  depth: 4,
  maxNodes: 90_000,
  beamWidth: 40,
  weights: DEFAULT_WEIGHTS,
};

// Coaching is a single bounded reference search. Deeper search belongs to the
// Rust engine rather than running an unbounded refinement loop in the browser.
export const COACHING_SEARCH: SearchConfig = {
  depth: 3,
  maxNodes: 18_000,
  beamWidth: 36,
  weights: DEFAULT_WEIGHTS,
};

export function searchBestAction(state: GameState, config: SearchConfig): SearchResult {
  const rootPlayer = state.turn;
  const initialActions = orderActions(state, rootPlayer, config.weights);
  if (!initialActions.length) {
    return { action: null, score: evaluatePosition(state, rootPlayer, config.weights), nodes: 0, exhausted: false, completedDepth: 0, tableHits: 0 };
  }

  const budget: Budget = { nodes: 0, exhausted: false, tableHits: 0 };
  const table = new Map<string, TableEntry>();
  let bestAction = initialActions[0];
  let bestScore = Number.NEGATIVE_INFINITY;
  let completedDepth = 0;

  // Preserve the best move from the last fully completed depth when a mobile-
  // friendly node budget expires.
  for (let depth = 1; depth <= config.depth; depth += 1) {
    const actions = putFirst(orderActions(state, rootPlayer, config.weights), bestAction);
    let iterationAction = actions[0];
    let iterationScore = Number.NEGATIVE_INFINITY;
    let complete = true;
    let alpha = Number.NEGATIVE_INFINITY;
    for (const action of actions) {
      if (budget.nodes >= config.maxNodes) { budget.exhausted = true; complete = false; break; }
      const next = applyLegalAction(state, action);
      budget.nodes += 1;
      const score = minimax(next, rootPlayer, depth - 1, alpha, Number.POSITIVE_INFINITY, config, budget, table);
      if (score > iterationScore || (score === iterationScore && actionOrder(action) < actionOrder(iterationAction))) {
        iterationAction = action;
        iterationScore = score;
      }
      alpha = Math.max(alpha, iterationScore);
      if (budget.exhausted) { complete = false; break; }
    }
    if (!complete) break;
    bestAction = iterationAction;
    bestScore = iterationScore;
    completedDepth = depth;
  }

  if (completedDepth === 0) {
    bestScore = evaluatePosition(applyLegalAction(state, bestAction), rootPlayer, config.weights);
  }
  return { action: bestAction, score: bestScore, nodes: budget.nodes, exhausted: budget.exhausted, completedDepth, tableHits: budget.tableHits };
}

export function analyzeAction(state: GameState, action: Action, config: SearchConfig = COACHING_SEARCH): MoveEvaluation {
  const rootPlayer = state.turn;
  const baseline = evaluatePosition(state, rootPlayer, config.weights);
  const budget: Budget = { nodes: 0, exhausted: false, tableHits: 0 };
  const table = new Map<string, TableEntry>();
  const next = applyLegalAction(state, action);
  budget.nodes += 1;
  const score = next.winner
    ? evaluatePosition(next, rootPlayer, config.weights)
    : minimax(next, rootPlayer, Math.max(0, config.depth - 1), Number.NEGATIVE_INFINITY, Number.POSITIVE_INFINITY, config, budget, table);
  return {
    action,
    beforeScore: baseline,
    score,
    delta: score - baseline,
    nodes: budget.nodes,
    exhausted: budget.exhausted,
    completedDepth: config.depth,
    tableHits: budget.tableHits,
  };
}

export function analyzeActions(state: GameState, config: SearchConfig = COACHING_SEARCH, maxActions = 48): MoveEvaluation[] {
  const rootPlayer = state.turn;
  const baseline = evaluatePosition(state, rootPlayer, config.weights);
  const budget: Budget = { nodes: 0, exhausted: false, tableHits: 0 };
  const table = new Map<string, TableEntry>();
  const results: MoveEvaluation[] = [];
  let alpha = Number.NEGATIVE_INFINITY;
  const actions = orderActions(state, rootPlayer, config.weights).slice(0, maxActions);

  for (const action of actions) {
    if (budget.nodes >= config.maxNodes) {
      budget.exhausted = true;
      break;
    }
    const next = applyLegalAction(state, action);
    budget.nodes += 1;
    const score = next.winner
      ? evaluatePosition(next, rootPlayer, config.weights)
      : minimax(next, rootPlayer, Math.max(0, config.depth - 1), alpha, Number.POSITIVE_INFINITY, config, budget, table);
    results.push({
      action,
      beforeScore: baseline,
      score,
      delta: score - baseline,
      nodes: budget.nodes,
      exhausted: budget.exhausted,
      completedDepth: config.depth,
      tableHits: budget.tableHits,
    });
    alpha = Math.max(alpha, score);
    if (budget.exhausted) break;
  }

  return results.sort((left, right) => right.score - left.score || actionOrder(left.action) - actionOrder(right.action));
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
  const structure = largestComponent(state.board, player) - largestComponent(state.board, opponent);
  const threats = captureOpportunities(state, player) - captureOpportunities(state, opponent);
  const edges = edgePresence(state.board, player) - edgePresence(state.board, opponent);
  return (opponentDistance - ownDistance) * weights.path
    + (ownPieces - opponentPieces) * weights.material
    + captureDirection * lastCapture * weights.capture
    + structure * weights.structure
    + threats * weights.threat
    + edges * weights.edge;
}

function minimax(
  state: GameState,
  rootPlayer: Player,
  depth: number,
  alpha: number,
  beta: number,
  config: SearchConfig,
  budget: Budget,
  table: Map<string, TableEntry>,
): number {
  if (state.winner || depth <= 0) return evaluatePosition(state, rootPlayer, config.weights);
  if (budget.nodes >= config.maxNodes) {
    budget.exhausted = true;
    return evaluatePosition(state, rootPlayer, config.weights);
  }

  const key = searchKey(state, rootPlayer);
  const cached = table.get(key);
  const originalAlpha = alpha;
  const originalBeta = beta;
  if (cached && cached.depth >= depth) {
    budget.tableHits += 1;
    if (cached.flag === "exact") return cached.score;
    if (cached.flag === "lower") alpha = Math.max(alpha, cached.score);
    else beta = Math.min(beta, cached.score);
    if (alpha >= beta) return cached.score;
  }

  const maximizing = state.turn === rootPlayer;
  const actions = orderActions(state, rootPlayer, config.weights).slice(0, config.beamWidth);
  if (!actions.length) return evaluatePosition(state, rootPlayer, config.weights);
  let best = maximizing ? Number.NEGATIVE_INFINITY : Number.POSITIVE_INFINITY;
  for (const action of actions) {
    const next = applyLegalAction(state, action);
    budget.nodes += 1;
    const score = minimax(next, rootPlayer, depth - 1, alpha, beta, config, budget, table);
    if (maximizing) {
      best = Math.max(best, score);
      alpha = Math.max(alpha, best);
    } else {
      best = Math.min(best, score);
      beta = Math.min(beta, best);
    }
    if (beta <= alpha || budget.nodes >= config.maxNodes) break;
  }
  if (!budget.exhausted) {
    const flag = best <= originalAlpha ? "upper" : best >= originalBeta ? "lower" : "exact";
    table.set(key, { depth, score: best, flag });
  }
  if (budget.nodes >= config.maxNodes) budget.exhausted = true;
  return best;
}

function orderActions(state: GameState, rootPlayer: Player, weights: EvaluationWeights) {
  const maximizing = state.turn === rootPlayer;
  return legalActions(state)
    .map((action) => {
      const next = applyLegalAction(state, action);
      const tactical = next.winner === state.turn ? 2_000_000_000 : (next.lastAction?.captured.length ?? 0) * 10_000;
      return { action, score: tactical + evaluatePosition(next, rootPlayer, weights) };
    })
    .sort((left, right) => {
      const difference = maximizing ? right.score - left.score : left.score - right.score;
      return difference || actionOrder(left.action) - actionOrder(right.action);
    })
    .map(({ action }) => action);
}

// Dijkstra: our pieces cost 0, empty cells cost 1, opponents are walls.
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
    for (const next of neighbors(current.square)) {
      if (board[next] === otherPlayer(player)) continue;
      const distance = current.distance + (board[next] === player ? 0 : 1);
      if (distance >= distances[next]) continue;
      distances[next] = distance;
      frontier.push({ square: next, distance });
    }
  }
  return 49;
}

function largestComponent(board: GameState["board"], player: Player) {
  const remaining = new Set(board.flatMap((piece, square) => piece === player ? [square] : []));
  let largest = 0;
  while (remaining.size) {
    const first = remaining.values().next().value as number;
    const stack = [first];
    remaining.delete(first);
    let size = 0;
    while (stack.length) {
      const square = stack.pop()!;
      size += 1;
      for (const next of neighbors(square)) {
        if (!remaining.has(next)) continue;
        remaining.delete(next);
        stack.push(next);
      }
    }
    largest = Math.max(largest, size);
  }
  return largest;
}

function captureOpportunities(state: GameState, player: Player) {
  const forbidden = new Set(state.forbidden);
  const victims = new Set<number>();
  for (let origin = 0; origin < 49; origin += 1) {
    if (state.board[origin] || forbidden.has(origin)) continue;
    const row = Math.floor(origin / 7);
    const column = origin % 7;
    for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) {
      const farRow = row + dr * 2;
      const farColumn = column + dc * 2;
      if (farRow < 0 || farRow >= 7 || farColumn < 0 || farColumn >= 7) continue;
      const near = (row + dr) * 7 + column + dc;
      const far = farRow * 7 + farColumn;
      if (state.board[near] === otherPlayer(player) && state.board[far] === player) victims.add(near);
    }
  }
  return victims.size;
}

function edgePresence(board: GameState["board"], player: Player) {
  let near = false;
  let far = false;
  for (let index = 0; index < 7; index += 1) {
    const nearSquare = player === "light" ? 42 + index : index * 7;
    const farSquare = player === "light" ? index : index * 7 + 6;
    near ||= board[nearSquare] === player;
    far ||= board[farSquare] === player;
  }
  return Number(near) + Number(far);
}

function neighbors(square: number) {
  const row = Math.floor(square / 7);
  const column = square % 7;
  const result: number[] = [];
  if (row > 0) result.push(square - 7);
  if (row < 6) result.push(square + 7);
  if (column > 0) result.push(square - 1);
  if (column < 6) result.push(square + 1);
  return result;
}

function searchKey(state: GameState, rootPlayer: Player) {
  const board = state.board.map((piece) => piece === "light" ? "L" : piece === "dark" ? "D" : ".").join("");
  return `${rootPlayer}|${board}|${state.turn}|${state.reserve.light},${state.reserve.dark}|${state.forbidden.join(",")}|${state.lastRelocatedTo.light},${state.lastRelocatedTo.dark}`;
}

function putFirst(actions: Action[], preferred: Action) {
  const index = actions.findIndex((action) => actionOrder(action) === actionOrder(preferred));
  if (index <= 0) return actions;
  return [actions[index], ...actions.slice(0, index), ...actions.slice(index + 1)];
}

function actionOrder(action: Action) {
  return action.kind === "place" ? action.to : action.from * 49 + action.to;
}
