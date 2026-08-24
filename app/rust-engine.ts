import type { MoveEvaluation, SearchConfig, SearchResult } from "./ai.ts";
import type { Action, GameState, Player } from "./pathagon.ts";

type RuntimePosition = {
  contractVersion: 1;
  config: GameState["config"];
  board: Array<Player | null>;
  reserve: Record<Player, number>;
  turn: Player;
  forbidden: number[];
  lastRelocatedTo: Record<Player, number | null>;
  lastCapture: number;
  lastPlayer: Player | null;
  winner: Player | null;
  winningPath: number[];
  ply: number;
};

type RuntimeSearchResult = Omit<SearchResult, "action"> & { action: Action | null };

type RuntimeMoveEvaluation = Omit<MoveEvaluation, "beforeScore" | "completedDepth" | "tableHits"> & {
  beforeScore: number;
  completedDepth: number;
  tableHits: number;
};

type RustWasmModule = {
  default(input?: string | URL): Promise<unknown>;
  pathagon_legal_actions(position: string): string;
  pathagon_apply_action(position: string, action: string): string;
  pathagon_search_best_action(position: string, config: string): string;
  pathagon_lunatic_action(position: string): string;
  pathagon_analyze_action(position: string, action: string, config: string): string;
  pathagon_analyze_actions(position: string, config: string, maxActions: number): string;
};

let modulePromise: Promise<RustWasmModule> | null = null;

export type RustEngine = {
  legalActions(state: GameState): Action[];
  applyAction(state: GameState, action: Action): GameState;
  searchBestAction(state: GameState, config: SearchConfig): RuntimeSearchResult;
  lunaticAction(state: GameState): RuntimeSearchResult;
  analyzeAction(state: GameState, action: Action, config: SearchConfig): MoveEvaluation;
  analyzeActions(state: GameState, config: SearchConfig, maxActions: number): MoveEvaluation[];
};

export function loadRustEngine(): Promise<RustEngine> {
  const moduleUrl = ["/engine/pathagon_engine", ".js"].join("");
  modulePromise ??= import(/* @vite-ignore */ moduleUrl).then(async (module) => {
    const wasm = module as unknown as RustWasmModule;
    await wasm.default();
    return wasm;
  });
  return modulePromise.then((wasm) => ({
    legalActions(state) {
      return JSON.parse(wasm.pathagon_legal_actions(JSON.stringify(toRuntimePosition(state)))) as Action[];
    },
    applyAction(state, action) {
      const next = JSON.parse(wasm.pathagon_apply_action(JSON.stringify(toRuntimePosition(state)), JSON.stringify(action))) as RuntimePosition;
      return fromRuntimePosition(state, next, action);
    },
    searchBestAction(state, config) {
      return JSON.parse(wasm.pathagon_search_best_action(JSON.stringify(toRuntimePosition(state)), JSON.stringify(config))) as RuntimeSearchResult;
    },
    lunaticAction(state) {
      return JSON.parse(wasm.pathagon_lunatic_action(JSON.stringify(toRuntimePosition(state)))) as RuntimeSearchResult;
    },
    analyzeAction(state, action, config) {
      return JSON.parse(wasm.pathagon_analyze_action(
        JSON.stringify(toRuntimePosition(state)),
        JSON.stringify(action),
        JSON.stringify(config),
      )) as RuntimeMoveEvaluation;
    },
    analyzeActions(state, config, maxActions) {
      return JSON.parse(wasm.pathagon_analyze_actions(
        JSON.stringify(toRuntimePosition(state)),
        JSON.stringify(config),
        maxActions,
      )) as RuntimeMoveEvaluation[];
    },
  }));
}

function toRuntimePosition(state: GameState): RuntimePosition {
  return {
    contractVersion: 1,
    config: state.config,
    board: [...state.board],
    reserve: { ...state.reserve },
    turn: state.turn,
    forbidden: [...state.forbidden].sort((left, right) => left - right),
    lastRelocatedTo: { ...state.lastRelocatedTo },
    lastCapture: state.lastAction?.captured.length ?? 0,
    lastPlayer: state.lastAction?.player ?? null,
    winner: state.winner,
    winningPath: [...state.winningPath],
    ply: state.ply,
  };
}

function fromRuntimePosition(before: GameState, next: RuntimePosition, action: Action): GameState {
  return {
    config: before.config,
    board: [...next.board],
    reserve: { ...next.reserve },
    turn: next.turn,
    forbidden: [...next.forbidden],
    lastRelocatedTo: { ...next.lastRelocatedTo },
    winner: next.winner,
    winningPath: [...next.winningPath],
    lastAction: {
      ...action,
      player: before.turn,
      captured: [...next.forbidden],
    },
    ply: next.ply,
  };
}
