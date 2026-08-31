import type { MoveEvaluation } from "./ai";
import type { Opponent } from "./opponents";
import type { Action, GameState } from "./pathagon";
import type { SearchProgress } from "./rust-search-client";
import type { PathfinderMoveTelemetry } from "./archive-metadata";

export type GameDebugPayload = {
  format: "pathagon-game-debug-v1";
  gameId: string;
  page: {
    url?: string;
    userAgent?: string;
  };
  opponent: Pick<Opponent, "id" | "name" | "version" | "engine">;
  settings: {
    depth: number;
    maxNodes: number;
    deadlineMs: number;
  };
  result: {
    winner: GameState["winner"];
    ply: number;
    turn: GameState["turn"];
    board: GameState["board"];
    reserve: GameState["reserve"];
    forbidden: number[];
    lastRelocatedTo: GameState["lastRelocatedTo"];
    winningPath: number[];
    lastAction: GameState["lastAction"];
  };
  actions: Action[];
  pathfinder: {
    searches: PathfinderMoveTelemetry[];
    lastSearch: PathfinderMoveTelemetry | null;
    progress: SearchProgress | null;
  };
  coach: {
    status: "idle" | "searching" | "ready";
    action: Action | null;
    evaluation: MoveEvaluation | null;
  };
  runtime: {
    rustEngineReady: boolean;
    cnnEngineReady: boolean;
    engineError: string | null;
    cnnError: string | null;
    archiveStatus: "idle" | "saving" | "saved" | "error";
    archiveError: string | null;
  };
};

type BuildGameDebugPayloadInput = {
  gameId: string;
  game: GameState;
  opponent: Pick<Opponent, "id" | "name" | "version" | "engine">;
  depth: number;
  maxNodes: number;
  deadlineMs: number;
  actions: Action[];
  pathfinderSearches: PathfinderMoveTelemetry[];
  lastPathfinderSearch: PathfinderMoveTelemetry | null;
  pathfinderProgress: SearchProgress | null;
  coachingStatus: "idle" | "searching" | "ready";
  coachingAction: Action | null;
  coachingEvaluation: MoveEvaluation | null;
  rustEngineReady: boolean;
  cnnEngineReady: boolean;
  engineError: string | null;
  cnnError: string | null;
  archiveStatus: "idle" | "saving" | "saved" | "error";
  archiveError: string | null;
  pageUrl?: string;
  userAgent?: string;
};

export function buildGameDebugPayload(input: BuildGameDebugPayloadInput): GameDebugPayload {
  const { game } = input;
  return {
    format: "pathagon-game-debug-v1",
    gameId: input.gameId,
    page: {
      ...(input.pageUrl ? { url: input.pageUrl } : {}),
      ...(input.userAgent ? { userAgent: input.userAgent } : {}),
    },
    opponent: {
      id: input.opponent.id,
      name: input.opponent.name,
      version: input.opponent.version,
      engine: input.opponent.engine,
    },
    settings: {
      depth: input.depth,
      maxNodes: input.maxNodes,
      deadlineMs: input.deadlineMs,
    },
    result: {
      winner: game.winner,
      ply: game.ply,
      turn: game.turn,
      board: [...game.board],
      reserve: { ...game.reserve },
      forbidden: [...game.forbidden],
      lastRelocatedTo: { ...game.lastRelocatedTo },
      winningPath: [...game.winningPath],
      lastAction: game.lastAction
        ? { ...game.lastAction, captured: [...game.lastAction.captured] }
        : null,
    },
    actions: input.actions.map((action) => ({ ...action })),
    pathfinder: {
      searches: input.pathfinderSearches,
      lastSearch: input.lastPathfinderSearch,
      progress: input.pathfinderProgress,
    },
    coach: {
      status: input.coachingStatus,
      action: input.coachingAction,
      evaluation: input.coachingEvaluation,
    },
    runtime: {
      rustEngineReady: input.rustEngineReady,
      cnnEngineReady: input.cnnEngineReady,
      engineError: input.engineError,
      cnnError: input.cnnError,
      archiveStatus: input.archiveStatus,
      archiveError: input.archiveError,
    },
  };
}

export function formatGameDebugPayload(payload: GameDebugPayload) {
  return JSON.stringify(payload, null, 2);
}
