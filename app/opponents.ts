import { PATHFINDER_SEARCH, searchBestAction, SURVEYOR_SEARCH } from "./ai.ts";
import { legalActions } from "./pathagon.ts";
import type { Action, GameState } from "./pathagon.ts";

export type Opponent = {
  id: string;
  name: string;
  version: string;
  engine: string;
  elo: string;
  personality: string;
  searchDepth: number;
  chooseAction(state: GameState): Action | null;
};

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

export const PATHFINDER_OPPONENT: Opponent = {
  id: "pathfinder-v0",
  name: "The Pathfinder",
  version: "0.3.0",
  engine: "4-ply iterative",
  elo: "Unrated · expert",
  personality: "Builds quietly. Punishes shortcuts.",
  searchDepth: 4,
  chooseAction(state) {
    return searchBestAction(state, PATHFINDER_SEARCH).action;
  },
};

export const OPPONENTS = [PATHFINDER_OPPONENT, SURVEYOR_OPPONENT, RANDOM_OPPONENT] as const;

export function getOpponent(id: string): Opponent {
  return OPPONENTS.find((opponent) => opponent.id === id) ?? SURVEYOR_OPPONENT;
}
