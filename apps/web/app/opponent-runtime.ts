import type { SearchConfig } from "./ai.ts";
import type { Action, GameState } from "./pathagon.ts";
import type { CnnEngine } from "./cnn-engine.ts";
import type { GnnEngine, JepaEngine, QAdvEngine } from "./learned-engine.ts";
import type { RustEngine, SearchTrace, TransitionPolicyEngine } from "./rust-engine.ts";

/** The single browser contract shared by every player-facing opponent card. */
export type OpponentCapability = "rankMoves" | "search" | "evaluateBoard";

export type OpponentStatus = "ready" | "artifact-pending" | "legacy";

export type OpponentControl = {
  id: string;
  label: string;
  info: string;
  values: readonly [number, number, number, number, number];
  defaultIndex: number;
  format: (value: number) => string;
};

export type RankedAction = {
  action: Action;
  /** Relative preference, never a probability. */
  preference: number;
  policyPrior?: number;
  value?: number;
  visits?: number;
  randomPriority?: number;
};

export type SearchBudget = {
  maxNodes: number;
  maxTimeMs: number;
  targetDepth: number;
  simulations?: number;
  samples?: number;
};

export type SearchTelemetry = {
  budget: SearchBudget;
  elapsedMs: number;
  nodes: number;
  simulations?: number;
  depth: number;
  exhausted: boolean;
  cancelled: boolean;
  trace: SearchTrace[];
};

export type OpponentSearchResult = {
  action: Action | null;
  ranked: RankedAction[];
  telemetry: SearchTelemetry;
  /** Rando's output is deliberately not described as goodness. */
  interpretation: "relative preference" | "random priority/order";
};

export type BoardEvaluation = {
  ranked: RankedAction[];
  value?: number;
  telemetry: SearchTelemetry;
  interpretation: "relative preference" | "random priority/order";
};

export type OpponentRuntimeContext = {
  rustEngine: RustEngine;
  cnnEngine?: CnnEngine;
  gnnEngine?: GnnEngine;
  qadvEngine?: QAdvEngine;
  jepaEngine?: JepaEngine;
  transitionPolicyEngine?: TransitionPolicyEngine;
};

export type OpponentRuntimeConfig = {
  controls: Record<string, number>;
  seed: number;
  signal?: AbortSignal;
};

export type OpponentRuntime = {
  rankMoves: (state: GameState, context: OpponentRuntimeContext, config: OpponentRuntimeConfig) => RankedAction[];
  search: (state: GameState, context: OpponentRuntimeContext, config: OpponentRuntimeConfig) => OpponentSearchResult;
  evaluateBoard: (state: GameState, context: OpponentRuntimeContext, config: OpponentRuntimeConfig) => BoardEvaluation;
};

export const FIVE_STEP_INDEXES = [0, 1, 2, 3, 4] as const;

export function numericControl(
  id: string,
  label: string,
  info: string,
  values: readonly [number, number, number, number, number],
  format: (value: number) => string = (value) => String(value),
  defaultIndex = 2,
): OpponentControl {
  return { id, label, info, values, format, defaultIndex };
}

export function searchConfigFromControls(controls: Record<string, number>, fallback: SearchConfig): SearchConfig {
  return {
    ...fallback,
    depth: Math.max(1, Math.floor(controls.depth ?? fallback.depth)),
    maxNodes: Math.max(1, Math.floor(controls.maxNodes ?? fallback.maxNodes)),
    beamWidth: Math.max(1, Math.floor(controls.beamWidth ?? fallback.beamWidth)),
  };
}
