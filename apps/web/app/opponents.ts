import { connectionDistance, PATHFINDER_SEARCH, searchBestAction, SURVEYOR_SEARCH, type SearchConfig } from "./ai.ts";
import { PATHFINDER_TACTICAL_FILTER_ID, TRANSITION_PATHFINDER_ID, TRAINED_PATHFINDER_ID } from "./agent-ids.ts";
import { applyLegalAction, legalActions, otherPlayer } from "./pathagon.ts";
import type { Action, GameState } from "./pathagon.ts";
import type { CnnEngine } from "./cnn-engine.ts";
import type { RustEngine, TransitionPolicyEngine } from "./rust-engine.ts";

export type Opponent = {
  id: string;
  name: string;
  /** Short label used where the selector has limited horizontal space. */
  shortName?: string;
  version: string;
  engine: string;
  elo: string;
  personality: string;
  searchDepth: number | null;
  chooseAction(state: GameState): Action | null;
};

/**
 * Browser-safe Pathfinder horizons. The centre value preserves the shipped
 * baseline; the long-horizon values make the speed/strength trade-off explicit
 * while keeping the experiment bounded by both nodes and wall-clock time.
 */
export const PATHFINDER_DEPTH_OPTIONS = [
  2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
  21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
  41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60,
  61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
  81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
] as const;
export type PathfinderDepth = typeof PATHFINDER_DEPTH_OPTIONS[number];

/** Maximum browser search budget. The Rust runtime enforces this again. */
export const PATHFINDER_MAX_NODES_HARD_CAP = 10_000_000;
const PATHFINDER_ROLLBACK_MAX_NODES = 32_000;
export const PATHFINDER_MAX_NODES_DEFAULT = PATHFINDER_SEARCH.maxNodes;
export const PATHFINDER_MAX_NODES_OPTIONS = [PATHFINDER_ROLLBACK_MAX_NODES, 64_000, 250_000, 256_000, 500_000, 1_000_000, 2_000_000, 5_000_000, PATHFINDER_MAX_NODES_HARD_CAP] as const;
export const PATHFINDER_BEAM_OPTIONS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1_024, 2_048, 4_096] as const;
const PATHFINDER_BEAM_MAX = PATHFINDER_BEAM_OPTIONS[PATHFINDER_BEAM_OPTIONS.length - 1];

/** Default control value for the long-horizon browser experiments. */
export function pathfinderMaxNodesForDepth(depth: number) {
  if (depth <= 4) return PATHFINDER_ROLLBACK_MAX_NODES;
  if (depth === 5) return PATHFINDER_SEARCH.maxNodes;
  if (depth >= 100) return PATHFINDER_MAX_NODES_HARD_CAP;
  if (depth >= 50) return 5_000_000;
  return 1_000_000;
}

function defaultPathfinderMaxNodes(depth: number) {
  if (depth <= 2) return 512;
  if (depth <= 3) return 1_024;
  if (depth <= 4) return PATHFINDER_ROLLBACK_MAX_NODES;
  if (depth <= 5) return PATHFINDER_SEARCH.maxNodes;
  if (depth <= 6) return 8_000;
  if (depth <= 7) return 16_000;
  if (depth <= 8) return 32_000;
  if (depth <= 9) return 64_000;
  if (depth <= 10) return 128_000;
  if (depth <= 11) return 256_000;
  if (depth <= 12) return 512_000;
  if (depth < 50) return PATHFINDER_MAX_NODES_DEFAULT;
  if (depth < 100) return 5_000_000;
  return PATHFINDER_MAX_NODES_HARD_CAP;
}

function defaultPathfinderBeamWidth(depth: number) {
  if (depth <= 2) return 16;
  if (depth <= 3) return 12;
  if (depth <= 5) return PATHFINDER_SEARCH.beamWidth;
  if (depth <= 7) return 4;
  if (depth <= 9) return 3;
  return 2;
}

function clampPathfinderMaxNodes(maxNodes: number) {
  if (!Number.isFinite(maxNodes)) return PATHFINDER_MAX_NODES_DEFAULT;
  return Math.max(1, Math.min(PATHFINDER_MAX_NODES_HARD_CAP, Math.floor(maxNodes)));
}

function clampPathfinderBeamWidth(beamWidth: number) {
  if (!Number.isFinite(beamWidth)) return PATHFINDER_SEARCH.beamWidth;
  return Math.max(1, Math.min(PATHFINDER_BEAM_MAX, Math.floor(beamWidth)));
}

export function pathfinderSearchAtDepth(depth: number, maxNodes?: number, beamWidth?: number): SearchConfig {
  return pathfinderSearchAtDepthWithWeights(depth, PATHFINDER_SEARCH.weights, maxNodes, beamWidth);
}

export const TRAINED_PATHFINDER_WEIGHTS = {
  path: 241,
  material: 112,
  capture: 887,
  structure: 40,
  threat: 154,
  edge: 74,
} as const;

export function trainedPathfinderSearchAtDepth(depth: number, maxNodes?: number, beamWidth?: number): SearchConfig {
  return pathfinderSearchAtDepthWithWeights(depth, TRAINED_PATHFINDER_WEIGHTS, maxNodes, beamWidth);
}

function pathfinderSearchAtDepthWithWeights(depth: number, weights: SearchConfig["weights"], maxNodes?: number, beamWidth?: number): SearchConfig {
  const safeDepth = PATHFINDER_DEPTH_OPTIONS.reduce<PathfinderDepth>(
    (closest, option) => Math.abs(option - depth) < Math.abs(closest - depth) ? option : closest,
    PATHFINDER_SEARCH.depth as PathfinderDepth,
  );
  return {
    ...PATHFINDER_SEARCH,
    depth: safeDepth,
    maxNodes: clampPathfinderMaxNodes(maxNodes ?? defaultPathfinderMaxNodes(safeDepth)),
    beamWidth: clampPathfinderBeamWidth(beamWidth ?? defaultPathfinderBeamWidth(safeDepth)),
    weights,
  };
}

export const RANDOM_OPPONENT: Opponent = {
  id: "coin-flip-v0",
  name: "Coin Flip",
  version: "0.0.1",
  engine: "Random legal",
  elo: "≈ 100",
  personality: "No plans. No grudges. Pure impulse.",
  searchDepth: 0,
  chooseAction(state) {
    const actions = legalActions(state);
    return actions.length ? actions[Math.floor(Math.random() * actions.length)] : null;
  },
};

export const SURVEYOR_OPPONENT: Opponent = {
  id: "surveyor-v0",
  name: "The Surveyor",
  version: "0.2.0",
  engine: "2-ply minimax",
  elo: "Provisional",
  personality: "Measures twice. Connects once.",
  searchDepth: 2,
  chooseAction(state) {
    return searchBestAction(state, SURVEYOR_SEARCH).action;
  },
};

// Lunatic is intentionally a shallow, explainable baseline. It notices the
// same local patterns a human might spot first, then commits to them without
// checking the opponent's reply. That makes it useful as a breadth opponent
// and as a control when evaluating whether deeper search is earning its cost.
export const LUNATIC_OPPONENT: Opponent = {
  id: "lunatic-v0",
  name: "Lunatic",
  version: "0.1.0",
  engine: "1-ply pattern heuristic",
  elo: "Unrated · heuristic",
  personality: "Sees traps everywhere. Sometimes it is right.",
  searchDepth: 1,
  chooseAction(state) {
    return chooseLunaticAction(state);
  },
};

export const PATHFINDER_OPPONENT: Opponent = {
  id: PATHFINDER_TACTICAL_FILTER_ID,
  name: "The Pathfinder",
  shortName: "Pathfinder · Tactical",
  version: "0.4.0",
  engine: "5-ply iterative · tactical-safe",
  elo: "Unrated · expert",
  personality: "Builds quietly. Punishes shortcuts.",
  searchDepth: 5,
  chooseAction(state) {
    return searchBestAction(state, pathfinderSearchAtDepth(PATHFINDER_SEARCH.depth)).action;
  },
};

export const TRAINED_PATHFINDER_OPPONENT: Opponent = {
  id: TRAINED_PATHFINDER_ID,
  name: "The Pathfinder · Trained",
  shortName: "Pathfinder · Trained",
  version: "0.5.0",
  engine: "5-ply iterative · trained evaluator",
  elo: "Provisional · trained",
  personality: "Keeps the path, weighs the traps more carefully.",
  searchDepth: 5,
  chooseAction(state) {
    return searchBestAction(state, trainedPathfinderSearchAtDepth(PATHFINDER_SEARCH.depth)).action;
  },
};

/**
 * The strongest validated transition-policy opponent. Its model is loaded by
 * the Rust search worker; chooseOpponentAction retains the trained evaluator
 * as a safe fallback if the model asset cannot be fetched.
 */
export const TRANSITION_PATHFINDER_OPPONENT: Opponent = {
  id: TRANSITION_PATHFINDER_ID,
  name: "The Pathfinder · Transition v4",
  shortName: "Pathfinder · v4",
  version: "4.0.0",
  engine: "5-ply iterative · action-transition policy",
  elo: "Provisional · scaled research",
  personality: "Learns which moves change the board, then searches the consequences.",
  searchDepth: 5,
  chooseAction(state) {
    return searchBestAction(state, trainedPathfinderSearchAtDepth(PATHFINDER_SEARCH.depth)).action;
  },
};

export const CNN_OPPONENT: Opponent = {
  id: "cnn-puct-v0",
  name: "The Convolutionist",
  version: "0.1.0",
  engine: "CNN policy/value · 64 PUCT",
  elo: "Unrated · learned",
  personality: "Reads the whole board, then trusts the branches it has visited.",
  searchDepth: null,
  chooseAction(state) {
    return legalActions(state)[0] ?? null;
  },
};

export const CNN_SEARCH = { simulations: 64, cpuct: 1.5 } as const;
export const PATHFINDER_DEADLINE_MS = 2_800;
export const PATHFINDER_DEADLINE_OPTIONS = [2_800, 5_000, 10_000, 20_000, 30_000, 60_000] as const;

export const OPPONENTS = [CNN_OPPONENT, TRANSITION_PATHFINDER_OPPONENT, TRAINED_PATHFINDER_OPPONENT, PATHFINDER_OPPONENT, LUNATIC_OPPONENT, SURVEYOR_OPPONENT, RANDOM_OPPONENT] as const;

export function getOpponent(id: string): Opponent {
  return OPPONENTS.find((opponent) => opponent.id === id) ?? SURVEYOR_OPPONENT;
}

/** Choose the browser opponent move through the Rust/WASM engine boundary. */
export function chooseOpponentAction(
  engine: RustEngine,
  opponent: Opponent,
  state: GameState,
  cnnEngine?: CnnEngine,
  pathfinderDepth: number = PATHFINDER_SEARCH.depth,
  deadlineMs?: number,
  transitionPolicyEngine?: TransitionPolicyEngine,
  pathfinderMaxNodes?: number,
  pathfinderBeamWidth?: number,
): Action | null {
  if (opponent.id === RANDOM_OPPONENT.id) {
    const actions = engine.legalActions(state);
    return actions.length ? actions[Math.floor(Math.random() * actions.length)] : null;
  }
  if (opponent.id === CNN_OPPONENT.id) {
    return cnnEngine?.selectAction(state, CNN_SEARCH).action ?? null;
  }
  if (opponent.id === LUNATIC_OPPONENT.id) return engine.lunaticAction(state).action;
  const isTransitionPathfinder = opponent.id === TRANSITION_PATHFINDER_OPPONENT.id;
  const isPathfinder = opponent.id === PATHFINDER_OPPONENT.id || opponent.id === TRAINED_PATHFINDER_OPPONENT.id || isTransitionPathfinder;
  const config = isPathfinder
    ? opponent.id === TRAINED_PATHFINDER_OPPONENT.id || isTransitionPathfinder
      ? trainedPathfinderSearchAtDepth(pathfinderDepth, pathfinderMaxNodes, pathfinderBeamWidth)
      : pathfinderSearchAtDepth(pathfinderDepth, pathfinderMaxNodes, pathfinderBeamWidth)
    : SURVEYOR_SEARCH;
  if (isTransitionPathfinder && transitionPolicyEngine) {
    return transitionPolicyEngine.searchBestAction(state, config, deadlineMs).action;
  }
  return isPathfinder
    ? deadlineMs === undefined
      ? engine.searchBestTacticalAction(state, config).action
      : engine.searchBestTacticalActionWithDeadline(state, config, deadlineMs).action
    : engine.searchBestAction(state, config).action;
}

export function chooseLunaticAction(state: GameState): Action | null {
  const actions = legalActions(state);
  if (!actions.length) return null;
  const player = state.turn;
  const beforeOwnDistance = connectionDistance(state.board, player);
  const beforeOpponentDistance = connectionDistance(state.board, otherPlayer(player));
  return actions
    .map((action) => {
      const next = applyLegalAction(state, action);
      const captured = next.lastAction?.captured.length ?? 0;
      const ownDistance = connectionDistance(next.board, player);
      const opponentDistance = connectionDistance(next.board, otherPlayer(player));
      const score = next.winner === player
        ? 1_000_000_000
        : captured * 10_000
          + (beforeOwnDistance - ownDistance) * 500
          + (opponentDistance - beforeOpponentDistance) * 350
          + (action.kind === "relocate" ? 10 : 0);
      return { action, score };
    })
    .sort((left, right) => right.score - left.score || actionOrder(left.action) - actionOrder(right.action))[0].action;
}

function actionOrder(action: Action) {
  return action.kind === "place" ? action.to : action.from * 49 + action.to;
}
