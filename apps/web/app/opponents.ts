import { connectionDistance, PATHFINDER_SEARCH, searchBestAction, SURVEYOR_SEARCH, type SearchConfig } from "./ai.ts";
import { applyLegalAction, legalActions, otherPlayer } from "./pathagon.ts";
import type { Action, GameState } from "./pathagon.ts";
import type { CnnEngine } from "./cnn-engine.ts";
import type { RustEngine } from "./rust-engine.ts";

export type Opponent = {
  id: string;
  name: string;
  version: string;
  engine: string;
  elo: string;
  personality: string;
  searchDepth: number | null;
  chooseAction(state: GameState): Action | null;
};

/**
 * Browser-safe Pathfinder horizons. The centre value preserves the shipped
 * baseline; the outer values make the speed/strength trade-off explicit
 * without allowing an accidental unbounded search in the UI.
 */
export const PATHFINDER_DEPTH_OPTIONS = [2, 3, 4, 5, 6] as const;
export type PathfinderDepth = typeof PATHFINDER_DEPTH_OPTIONS[number];

const PATHFINDER_BUDGETS: Record<PathfinderDepth, number> = {
  2: 32_000,
  3: 58_000,
  4: PATHFINDER_SEARCH.maxNodes,
  5: 124_000,
  6: 164_000,
};

const PATHFINDER_BEAMS: Record<PathfinderDepth, number> = {
  2: 48,
  3: 44,
  4: PATHFINDER_SEARCH.beamWidth,
  5: 36,
  6: 32,
};

export function pathfinderSearchAtDepth(depth: number): SearchConfig {
  const safeDepth = PATHFINDER_DEPTH_OPTIONS.reduce<PathfinderDepth>(
    (closest, option) => Math.abs(option - depth) < Math.abs(closest - depth) ? option : closest,
    PATHFINDER_SEARCH.depth as PathfinderDepth,
  );
  return {
    ...PATHFINDER_SEARCH,
    depth: safeDepth,
    maxNodes: PATHFINDER_BUDGETS[safeDepth],
    beamWidth: PATHFINDER_BEAMS[safeDepth],
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
  id: "pathfinder-v0",
  name: "The Pathfinder",
  version: "0.4.0",
  engine: "4-ply iterative · tactical-safe",
  elo: "Unrated · expert",
  personality: "Builds quietly. Punishes shortcuts.",
  searchDepth: 4,
  chooseAction(state) {
    return searchBestAction(state, pathfinderSearchAtDepth(PATHFINDER_SEARCH.depth)).action;
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

export const OPPONENTS = [CNN_OPPONENT, PATHFINDER_OPPONENT, LUNATIC_OPPONENT, SURVEYOR_OPPONENT, RANDOM_OPPONENT] as const;

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
): Action | null {
  if (opponent.id === RANDOM_OPPONENT.id) {
    const actions = engine.legalActions(state);
    return actions.length ? actions[Math.floor(Math.random() * actions.length)] : null;
  }
  if (opponent.id === CNN_OPPONENT.id) {
    return cnnEngine?.selectAction(state, CNN_SEARCH).action ?? null;
  }
  if (opponent.id === LUNATIC_OPPONENT.id) return engine.lunaticAction(state).action;
  const config = opponent.id === PATHFINDER_OPPONENT.id
    ? pathfinderSearchAtDepth(pathfinderDepth)
    : SURVEYOR_SEARCH;
  return opponent.id === PATHFINDER_OPPONENT.id
    ? engine.searchBestTacticalAction(state, config).action
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
