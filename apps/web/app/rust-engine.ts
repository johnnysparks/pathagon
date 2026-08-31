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

export type TransitionPolicyRankedAction = {
  action: Action;
  safe: boolean;
  immediateWin: boolean;
  score: number;
};

export type TransitionPolicySearchResult = RuntimeSearchResult;

type TransitionPolicyModelHandle = {
  modelName(): string;
  encoding(): string;
  score(position: string, action: string, safe: boolean): number;
  rankActions(position: string, maxActions: number): string;
  searchBestAction(position: string, config: string, deadlineMs: number): string;
};

type RustWasmModule = {
  default(input?: string | URL): Promise<unknown>;
  pathagon_legal_actions(position: string): string;
  pathagon_apply_action(position: string, action: string): string;
  pathagon_search_best_action(position: string, config: string): string;
  pathagon_search_best_action_with_tactical_filter(position: string, config: string): string;
  pathagon_search_best_action_with_tactical_filter_deadline(position: string, config: string, deadlineMs: number): string;
  pathagon_lunatic_action(position: string): string;
  pathagon_analyze_action(position: string, action: string, config: string): string;
  pathagon_analyze_actions(position: string, config: string, maxActions: number): string;
  PathagonTransitionPolicyModel: new (bytes: Uint8Array) => TransitionPolicyModelHandle;
};

let modulePromise: Promise<RustWasmModule> | null = null;

function loadRustWasmModule(): Promise<RustWasmModule> {
  const moduleUrl = ["/engine/pathagon_engine", ".js"].join("");
  modulePromise ??= import(/* @vite-ignore */ moduleUrl).then(async (module) => {
    const wasm = module as unknown as RustWasmModule;
    await wasm.default();
    return wasm;
  });
  return modulePromise;
}

export type RustEngine = {
  legalActions(state: GameState): Action[];
  applyAction(state: GameState, action: Action): GameState;
  searchBestAction(state: GameState, config: SearchConfig): RuntimeSearchResult;
  searchBestTacticalAction(state: GameState, config: SearchConfig): RuntimeSearchResult;
  searchBestTacticalActionWithDeadline(state: GameState, config: SearchConfig, deadlineMs: number): RuntimeSearchResult;
  lunaticAction(state: GameState): RuntimeSearchResult;
  analyzeAction(state: GameState, action: Action, config: SearchConfig): MoveEvaluation;
  analyzeActions(state: GameState, config: SearchConfig, maxActions: number): MoveEvaluation[];
};

export type TransitionPolicyEngine = {
  modelName: string;
  encoding: string;
  score(state: GameState, action: Action, safe?: boolean): number;
  rankActions(state: GameState, maxActions?: number): TransitionPolicyRankedAction[];
  searchBestAction(state: GameState, config: SearchConfig, deadlineMs?: number): TransitionPolicySearchResult;
};

export function loadRustEngine(): Promise<RustEngine> {
  return loadRustWasmModule().then((wasm) => ({
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
    searchBestTacticalAction(state, config) {
      return JSON.parse(wasm.pathagon_search_best_action_with_tactical_filter(JSON.stringify(toRuntimePosition(state)), JSON.stringify(config))) as RuntimeSearchResult;
    },
    searchBestTacticalActionWithDeadline(state, config, deadlineMs) {
      return JSON.parse(wasm.pathagon_search_best_action_with_tactical_filter_deadline(
        JSON.stringify(toRuntimePosition(state)),
        JSON.stringify(config),
        deadlineMs,
      )) as RuntimeSearchResult;
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

/** Load the versioned explicit transition scorer through the Rust/WASM ABI. */
export function loadTransitionPolicyEngine(
  modelUrl = "/models/pathfinder-action-transition-v4-xent.json",
): Promise<TransitionPolicyEngine> {
  return loadRustWasmModule().then(async (wasm) => {
    const response = await fetch(modelUrl);
    if (!response.ok) throw new Error(`transition-policy model request failed (${response.status})`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    const model = new wasm.PathagonTransitionPolicyModel(bytes);
    return {
      modelName: model.modelName(),
      encoding: model.encoding(),
      score(state: GameState, action: Action, safe = true) {
        return model.score(JSON.stringify(toRuntimePosition(state)), JSON.stringify(action), safe);
      },
      rankActions(state: GameState, maxActions = 0) {
        return JSON.parse(model.rankActions(JSON.stringify(toRuntimePosition(state)), maxActions)) as TransitionPolicyRankedAction[];
      },
      searchBestAction(state: GameState, config: SearchConfig, deadlineMs = 2_800) {
        return JSON.parse(model.searchBestAction(
          JSON.stringify(toRuntimePosition(state)),
          JSON.stringify(config),
          deadlineMs,
        )) as TransitionPolicySearchResult;
      },
    };
  });
}

export function toRuntimePosition(state: GameState): RuntimePosition {
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
