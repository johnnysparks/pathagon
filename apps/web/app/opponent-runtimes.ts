import { searchBestAction, type SearchConfig } from "./ai.ts";
import {
  DOUBLE_DRAGON_ID,
  PATHMAN_ID,
  RANDO_RACCON_ID,
  SEER_ID,
  TILE_DRIVER_ID,
  YANN_TILESON_ID,
} from "./agent-ids.ts";
import type {
  BoardEvaluation,
  OpponentRuntime,
  OpponentRuntimeConfig,
  OpponentRuntimeContext,
  OpponentSearchResult,
  RankedAction,
  SearchTelemetry,
} from "./opponent-runtime.ts";
import { applyLegalAction, legalActions } from "./pathagon.ts";
import type { Action, GameState } from "./pathagon.ts";
import type { SearchTrace } from "./rust-engine.ts";

const EMPTY_TRACE: SearchTrace[] = [];

function telemetry(config: OpponentRuntimeConfig, overrides: Partial<SearchTelemetry> = {}): SearchTelemetry {
  const controls = config.controls;
  const maxNodes = Math.max(1, Math.floor(controls.maxNodes ?? 1));
  const maxTimeMs = Math.max(1, Math.floor(controls.maxTimeMs ?? 1));
  return {
    budget: {
      maxNodes,
      maxTimeMs,
      targetDepth: Math.max(1, Math.floor(controls.depth ?? 1)),
      ...(controls.simulations === undefined ? {} : { simulations: Math.floor(controls.simulations) }),
      ...(controls.samples === undefined ? {} : { samples: Math.floor(controls.samples) }),
    },
    elapsedMs: 0,
    nodes: 0,
    depth: 0,
    exhausted: false,
    cancelled: Boolean(config.signal?.aborted),
    trace: EMPTY_TRACE,
    ...overrides,
  };
}

function puctTelemetry(config: OpponentRuntimeConfig, result: { nodes: number; simulations: number; evaluations: Array<{ action: Action; value: number }> }, startedAt: number): SearchTelemetry {
  const elapsedMs = typeof performance === "undefined" ? 0 : Math.round(performance.now() - startedAt);
  return telemetry(config, {
    elapsedMs,
    nodes: result.nodes,
    simulations: result.simulations,
    depth: 1,
    trace: [{
      depth: 1,
      nodes: result.nodes,
      tableHits: 0,
      candidates: result.evaluations.map((evaluation) => ({ action: evaluation.action, score: evaluation.value })),
    }],
    exhausted: result.simulations < Math.max(1, Math.floor(config.controls.simulations ?? result.simulations)),
  });
}

function legalRanked(state: GameState, ranked: RankedAction[]): RankedAction[] {
  const legal = new Set(legalActions(state).map(actionKey));
  return ranked.filter((candidate) => legal.has(actionKey(candidate.action)));
}

function actionKey(action: Action) {
  return action.kind === "place" ? `p:${action.to}` : `m:${action.from}:${action.to}`;
}

function pathmanConfig(config: OpponentRuntimeConfig): SearchConfig {
  const controls = config.controls;
  return {
    depth: Math.max(1, Math.floor(controls.depth ?? 5)),
    maxNodes: Math.max(1, Math.floor(controls.maxNodes ?? 256_000)),
    beamWidth: Math.max(1, Math.floor(controls.beamWidth ?? 256)),
    weights: {
      path: 240,
      material: 110,
      capture: 700,
      structure: 55,
      threat: 130,
      edge: 80,
    },
  };
}

function rankPathman(state: GameState, context: OpponentRuntimeContext, config: OpponentRuntimeConfig): RankedAction[] {
  const results = context.rustEngine.analyzeActions(state, pathmanConfig(config), legalActions(state).length);
  return legalRanked(state, results.map((result) => ({ action: result.action, preference: result.score })));
}

function searchPathman(state: GameState, context: OpponentRuntimeContext, config: OpponentRuntimeConfig): OpponentSearchResult {
  const traces: SearchTrace[] = [];
  const startedAt = typeof performance === "undefined" ? 0 : performance.now();
  const search = context.rustEngine.searchBestTacticalActionWithDeadlineTrace(
    state,
    pathmanConfig(config),
    Math.max(1, Math.floor(config.controls.maxTimeMs ?? 2_800)),
    () => undefined,
    (trace) => traces.push(trace),
  );
  const ranked = traces.at(-1)?.candidates.map((candidate) => ({ action: candidate.action, preference: candidate.score }))
    ?? (search.action ? [{ action: search.action, preference: search.score }] : []);
  return {
    action: legalRanked(state, [{ action: search.action, preference: search.score }])[0]?.action ?? null,
    ranked: legalRanked(state, ranked),
    telemetry: telemetry(config, {
      elapsedMs: typeof performance === "undefined" ? 0 : Math.round(performance.now() - startedAt),
      nodes: search.nodes,
      depth: search.completedDepth,
      exhausted: search.exhausted,
      trace: traces,
    }),
    interpretation: "relative preference",
  };
}

function evaluatePathman(state: GameState, context: OpponentRuntimeContext, config: OpponentRuntimeConfig): BoardEvaluation {
  const ranked = rankPathman(state, context, config);
  return { ranked, value: ranked[0]?.preference, telemetry: telemetry(config, { nodes: ranked.length, depth: pathmanConfig(config).depth }), interpretation: "relative preference" };
}

function cnnRank(state: GameState, context: OpponentRuntimeContext, config: OpponentRuntimeConfig): RankedAction[] {
  if (config.signal?.aborted) return [];
  if (!context.cnnEngine) throw new Error("Seer requires the promoted CNN browser artifact");
  const evaluation = context.cnnEngine.evaluate(state);
  return legalRanked(state, evaluation.actions.map((action, index) => ({
    action,
    preference: evaluation.policyLogits[index] ?? Number.NEGATIVE_INFINITY,
    policyPrior: softmaxAt(evaluation.policyLogits, index),
    value: evaluation.value,
  }))).sort((left, right) => right.preference - left.preference || actionKey(left.action).localeCompare(actionKey(right.action)));
}

function searchSeer(state: GameState, context: OpponentRuntimeContext, config: OpponentRuntimeConfig): OpponentSearchResult {
  if (!context.cnnEngine) throw new Error("Seer requires the promoted CNN browser artifact");
  const startedAt = typeof performance === "undefined" ? 0 : performance.now();
  const result = context.cnnEngine.selectAction(state, {
    simulations: Math.max(1, Math.floor(config.controls.simulations ?? 64)),
    cpuct: config.controls.cpuct ?? 1.5,
    maxNodes: Math.max(1, Math.floor(config.controls.maxNodes ?? 128_000)),
    maxTimeMs: Math.max(1, Math.floor(config.controls.maxTimeMs ?? 2_000)),
  });
  const ranked = legalRanked(state, result.evaluations.map((evaluation) => ({
    action: evaluation.action,
    preference: evaluation.value,
    policyPrior: evaluation.prior,
    visits: evaluation.visits,
    value: evaluation.value,
  }))).sort((left, right) => (right.visits ?? 0) - (left.visits ?? 0) || right.preference - left.preference);
  return {
    action: legalRanked(state, result.action ? [{ action: result.action, preference: result.value }] : [])[0]?.action ?? ranked[0]?.action ?? null,
    ranked,
    telemetry: puctTelemetry(config, result, startedAt),
    interpretation: "relative preference",
  };
}

function evaluateSeer(state: GameState, context: OpponentRuntimeContext, config: OpponentRuntimeConfig): BoardEvaluation {
  if (!context.cnnEngine) throw new Error("Seer requires the promoted CNN browser artifact");
  const ranked = cnnRank(state, context, config);
  return { ranked, value: ranked[0]?.value, telemetry: telemetry(config, { nodes: ranked.length, depth: 1 }), interpretation: "relative preference" };
}

function gnnRank(state: GameState, context: OpponentRuntimeContext, config: OpponentRuntimeConfig): RankedAction[] {
  if (config.signal?.aborted) return [];
  if (!context.gnnEngine) throw new Error("Tile Driver requires the promoted GNN browser artifact");
  const evaluation = context.gnnEngine.evaluate(state);
  return legalRanked(state, evaluation.actions.map((action, index) => ({
    action,
    preference: evaluation.policyLogits[index] ?? Number.NEGATIVE_INFINITY,
    policyPrior: softmaxAt(evaluation.policyLogits, index),
    value: evaluation.value,
  }))).sort((left, right) => right.preference - left.preference || actionKey(left.action).localeCompare(actionKey(right.action)));
}

function searchTileDriver(state: GameState, context: OpponentRuntimeContext, config: OpponentRuntimeConfig): OpponentSearchResult {
  if (!context.gnnEngine) throw new Error("Tile Driver requires the promoted GNN browser artifact");
  const startedAt = typeof performance === "undefined" ? 0 : performance.now();
  const result = context.gnnEngine.selectAction(state, {
    simulations: Math.max(1, Math.floor(config.controls.simulations ?? 32)),
    cpuct: config.controls.cpuct ?? 1.5,
    maxNodes: Math.max(1, Math.floor(config.controls.maxNodes ?? 32_000)),
    maxTimeMs: Math.max(1, Math.floor(config.controls.maxTimeMs ?? 2_000)),
  });
  const ranked = legalRanked(state, result.evaluations.map((evaluation) => ({
    action: evaluation.action,
    preference: evaluation.value,
    policyPrior: evaluation.prior,
    visits: evaluation.visits,
    value: evaluation.value,
  }))).sort((left, right) => (right.visits ?? 0) - (left.visits ?? 0) || right.preference - left.preference);
  return {
    action: legalRanked(state, result.action ? [{ action: result.action, preference: result.value }] : [])[0]?.action ?? ranked[0]?.action ?? null,
    ranked,
    telemetry: puctTelemetry(config, result, startedAt),
    interpretation: "relative preference",
  };
}

function evaluateTileDriver(state: GameState, context: OpponentRuntimeContext, config: OpponentRuntimeConfig): BoardEvaluation {
  const ranked = gnnRank(state, context, config);
  return { ranked, value: ranked[0]?.value, telemetry: telemetry(config, { nodes: ranked.length, depth: 1 }), interpretation: "relative preference" };
}

function qadvRank(state: GameState, context: OpponentRuntimeContext, config: OpponentRuntimeConfig): RankedAction[] {
  if (config.signal?.aborted) return [];
  if (!context.qadvEngine) throw new Error("Double Dragon requires the promoted Q/Advantage browser artifact");
  const evaluation = context.qadvEngine.evaluate(state);
  const qWeight = Math.max(0, Math.min(1, config.controls.qAdvWeight ?? 0.5));
  const maximum = Math.max(...evaluation.policyLogits);
  const minimum = Math.min(...evaluation.policyLogits);
  const range = Math.max(1e-6, maximum - minimum);
  return legalRanked(state, evaluation.actions.map((action, index) => {
    const qValue = evaluation.qValues[index] ?? -1;
    const policySignal = ((evaluation.policyLogits[index] ?? minimum) - minimum) / range;
    return {
      action,
      preference: qWeight * qValue + (1 - qWeight) * policySignal,
      policyPrior: softmaxAt(evaluation.policyLogits, index),
      value: qValue,
    };
  })).sort((left, right) => right.preference - left.preference || actionKey(left.action).localeCompare(actionKey(right.action)));
}

function searchDoubleDragon(state: GameState, context: OpponentRuntimeContext, config: OpponentRuntimeConfig): OpponentSearchResult {
  if (!context.qadvEngine) throw new Error("Double Dragon requires the promoted Q/Advantage browser artifact");
  const startedAt = typeof performance === "undefined" ? 0 : performance.now();
  const result = context.qadvEngine.selectAction(state, {
    simulations: Math.max(1, Math.floor(config.controls.simulations ?? 32)),
    cpuct: config.controls.cpuct ?? 1.5,
    qAdvWeight: Math.max(0, Math.min(1, config.controls.qAdvWeight ?? 0.5)),
    maxNodes: Math.max(1, Math.floor(config.controls.maxNodes ?? 32_000)),
    maxTimeMs: Math.max(1, Math.floor(config.controls.maxTimeMs ?? 2_000)),
  });
  const ranked = qadvRank(state, context, config);
  const searched = legalRanked(state, result.evaluations.map((evaluation) => ({
    action: evaluation.action,
    preference: evaluation.value,
    policyPrior: evaluation.prior,
    visits: evaluation.visits,
    value: evaluation.value,
  }))).sort((left, right) => (right.visits ?? 0) - (left.visits ?? 0) || right.preference - left.preference);
  return {
    action: legalRanked(state, result.action ? [{ action: result.action, preference: result.value }] : [])[0]?.action ?? searched[0]?.action ?? ranked[0]?.action ?? null,
    ranked: searched.length ? searched : ranked,
    telemetry: puctTelemetry(config, result, startedAt),
    interpretation: "relative preference",
  };
}

function evaluateDoubleDragon(state: GameState, context: OpponentRuntimeContext, config: OpponentRuntimeConfig): BoardEvaluation {
  const ranked = qadvRank(state, context, config);
  return { ranked, value: ranked[0]?.value, telemetry: telemetry(config, { nodes: ranked.length, depth: 1 }), interpretation: "relative preference" };
}

function jepaRank(state: GameState, context: OpponentRuntimeContext, config: OpponentRuntimeConfig): RankedAction[] {
  if (config.signal?.aborted) return [];
  if (!context.jepaEngine) throw new Error("Yann Tileson requires the promoted JEPA browser artifact");
  const evaluation = context.jepaEngine.evaluate(state);
  return legalRanked(state, evaluation.actions.map((action, index) => ({
    action,
    preference: evaluation.rankLogits[index] ?? Number.NEGATIVE_INFINITY,
    value: evaluation.actionValues[index],
  }))).sort((left, right) => right.preference - left.preference || actionKey(left.action).localeCompare(actionKey(right.action)));
}

function searchYannTileson(state: GameState, context: OpponentRuntimeContext, config: OpponentRuntimeConfig): OpponentSearchResult {
  if (!context.jepaEngine) throw new Error("Yann Tileson requires the promoted JEPA browser artifact");
  const startedAt = typeof performance === "undefined" ? 0 : performance.now();
  const targetDepth = Math.max(1, Math.floor(config.controls.depth ?? 2));
  const beamWidth = Math.max(1, Math.floor(config.controls.beamWidth ?? 8));
  const maxNodes = Math.max(1, Math.floor(config.controls.maxNodes ?? 4_000));
  const maxTimeMs = Math.max(1, Math.floor(config.controls.maxTimeMs ?? 1_000));
  const root = jepaRank(state, context, config);
  type BeamNode = { state: GameState; rootAction: Action; score: number; depth: number };
  let frontier: BeamNode[] = root.slice(0, beamWidth).map((candidate) => ({
    state: context.rustEngine.applyAction(state, candidate.action),
    rootAction: candidate.action,
    score: candidate.value ?? candidate.preference,
    depth: 1,
  }));
  let nodes = frontier.length;
  let completedDepth = frontier.length ? 1 : 0;
  const traces: SearchTrace[] = [{
    depth: completedDepth,
    nodes,
    tableHits: 0,
    candidates: root.slice(0, beamWidth).map((candidate) => ({ action: candidate.action, score: candidate.preference })),
  }];
  while (frontier.length && completedDepth < targetDepth && nodes < maxNodes) {
    if (config.signal?.aborted) break;
    if (typeof performance !== "undefined" && performance.now() - startedAt >= maxTimeMs) break;
    const next: BeamNode[] = [];
    for (const node of frontier) {
      if (nodes >= maxNodes) break;
      const ranked = jepaRank(node.state, context, config).slice(0, beamWidth);
      for (const candidate of ranked) {
        if (nodes >= maxNodes || config.signal?.aborted) break;
        const nextState = context.rustEngine.applyAction(node.state, candidate.action);
        next.push({
          state: nextState,
          rootAction: node.rootAction,
          score: node.score + (candidate.value ?? candidate.preference) * 0.7 ** node.depth,
          depth: node.depth + 1,
        });
        nodes += 1;
      }
    }
    if (!next.length) break;
    next.sort((left, right) => right.score - left.score || actionKey(left.rootAction).localeCompare(actionKey(right.rootAction)));
    frontier = next.slice(0, beamWidth);
    completedDepth += 1;
    const scores = new Map<string, { action: Action; score: number }>();
    for (const node of frontier) {
      const key = actionKey(node.rootAction);
      const previous = scores.get(key);
      if (!previous || node.score > previous.score) scores.set(key, { action: node.rootAction, score: node.score });
    }
    traces.push({
      depth: completedDepth,
      nodes,
      tableHits: 0,
      candidates: [...scores.values()].sort((left, right) => right.score - left.score || actionKey(left.action).localeCompare(actionKey(right.action))),
    });
  }
  const bestByAction = new Map<string, RankedAction>();
  for (const node of frontier) {
    const key = actionKey(node.rootAction);
    const current = bestByAction.get(key);
    if (!current || node.score > current.preference) bestByAction.set(key, { action: node.rootAction, preference: node.score });
  }
  const ranked = [...bestByAction.values()].sort((left, right) => right.preference - left.preference || actionKey(left.action).localeCompare(actionKey(right.action)));
  const elapsedMs = typeof performance === "undefined" ? 0 : Math.round(performance.now() - startedAt);
  return {
    action: ranked[0]?.action ?? root[0]?.action ?? null,
    ranked: ranked.length ? ranked : root,
    telemetry: telemetry(config, {
      budget: { maxNodes, maxTimeMs, targetDepth },
      elapsedMs,
      nodes,
      depth: completedDepth,
      exhausted: nodes >= maxNodes || elapsedMs >= maxTimeMs || completedDepth < targetDepth,
      cancelled: Boolean(config.signal?.aborted),
      trace: traces,
    }),
    interpretation: "relative preference",
  };
}

function evaluateYannTileson(state: GameState, context: OpponentRuntimeContext, config: OpponentRuntimeConfig): BoardEvaluation {
  const ranked = jepaRank(state, context, config);
  return { ranked, value: ranked[0]?.value, telemetry: telemetry(config, { nodes: ranked.length, depth: 1 }), interpretation: "relative preference" };
}

function seededUnit(seed: number) {
  let value = (seed | 0) ^ 0x9e3779b9;
  return () => {
    value |= 0;
    value = Math.imul(value ^ (value >>> 16), 0x21f0aaad);
    value = Math.imul(value ^ (value >>> 15), 0x735a2d97);
    value ^= value >>> 15;
    return (value >>> 0) / 4_294_967_296;
  };
}

function randomRank(state: GameState, _context: OpponentRuntimeContext, config: OpponentRuntimeConfig): RankedAction[] {
  const random = seededUnit(config.seed ^ state.ply ^ state.board.reduce((hash, piece, index) => hash ^ ((piece ? piece.charCodeAt(0) : 0) + index * 17), 0));
  return legalActions(state).map((action) => {
    const priority = random();
    return { action, preference: priority, randomPriority: priority };
  }).sort((left, right) => (right.randomPriority ?? 0) - (left.randomPriority ?? 0) || actionKey(left.action).localeCompare(actionKey(right.action)));
}

function searchRando(state: GameState, context: OpponentRuntimeContext, config: OpponentRuntimeConfig): OpponentSearchResult {
  const ranked = randomRank(state, context, config);
  const requestedSamples = Math.min(ranked.length, Math.max(1, Math.floor(config.controls.samples ?? 8)));
  const maxNodes = Math.max(1, Math.floor(config.controls.maxNodes ?? 128));
  const maxTimeMs = Math.max(1, Math.floor(config.controls.maxTimeMs ?? 250));
  const targetDepth = Math.max(1, Math.floor(config.controls.depth ?? 1));
  const startedAt = typeof performance === "undefined" ? 0 : performance.now();
  const sampled: RankedAction[] = [];
  let nodes = 0;
  let completedDepth = 0;
  let timedOut = false;
  for (let rootIndex = 0; rootIndex < requestedSamples; rootIndex += 1) {
    if (nodes >= maxNodes || config.signal?.aborted) break;
    if (typeof performance !== "undefined" && performance.now() - startedAt >= maxTimeMs) {
      timedOut = true;
      break;
    }
    const root = ranked[rootIndex];
    if (!root) break;
    let continuation = context.rustEngine.applyAction(state, root.action);
    nodes += 1;
    let depth = 1;
    while (depth < targetDepth && nodes < maxNodes) {
      if (config.signal?.aborted) break;
      if (typeof performance !== "undefined" && performance.now() - startedAt >= maxTimeMs) {
        timedOut = true;
        break;
      }
      const actions = context.rustEngine.legalActions(continuation);
      if (!actions.length) break;
      const random = seededUnit(config.seed ^ (rootIndex + 1) * 0x45d9f3b ^ depth * 0x27d4eb2d);
      const action = actions[Math.floor(random() * actions.length)];
      if (!action) break;
      continuation = context.rustEngine.applyAction(continuation, action);
      nodes += 1;
      depth += 1;
    }
    completedDepth = Math.max(completedDepth, depth);
    sampled.push(root);
    if (timedOut) break;
  }
  const elapsedMs = typeof performance === "undefined" ? 0 : Math.round(performance.now() - startedAt);
  return {
    action: sampled[0]?.action ?? null,
    ranked: sampled,
    telemetry: telemetry(config, {
      budget: { maxNodes, maxTimeMs, targetDepth, samples: requestedSamples },
      elapsedMs,
      nodes,
      depth: completedDepth,
      exhausted: sampled.length < ranked.length || nodes >= maxNodes || timedOut || completedDepth < targetDepth,
      cancelled: Boolean(config.signal?.aborted),
      trace: [{
        depth: 1,
        nodes,
        tableHits: 0,
        candidates: sampled.map((candidate) => ({ action: candidate.action, score: candidate.randomPriority ?? candidate.preference })),
      }],
    }),
    interpretation: "random priority/order",
  };
}

function evaluateRando(state: GameState, _context: OpponentRuntimeContext, config: OpponentRuntimeConfig): BoardEvaluation {
  const ranked = randomRank(state, _context, config);
  return { ranked, telemetry: telemetry(config, { nodes: ranked.length }), interpretation: "random priority/order" };
}

export const OPPONENT_RUNTIMES: Readonly<Record<string, OpponentRuntime>> = {
  [PATHMAN_ID]: { rankMoves: rankPathman, search: searchPathman, evaluateBoard: evaluatePathman },
  [TILE_DRIVER_ID]: { rankMoves: gnnRank, search: searchTileDriver, evaluateBoard: evaluateTileDriver },
  [SEER_ID]: { rankMoves: cnnRank, search: searchSeer, evaluateBoard: evaluateSeer },
  [DOUBLE_DRAGON_ID]: { rankMoves: qadvRank, search: searchDoubleDragon, evaluateBoard: evaluateDoubleDragon },
  [RANDO_RACCON_ID]: { rankMoves: randomRank, search: searchRando, evaluateBoard: evaluateRando },
  [YANN_TILESON_ID]: { rankMoves: jepaRank, search: searchYannTileson, evaluateBoard: evaluateYannTileson },
};

export function opponentRuntime(id: string): OpponentRuntime {
  return OPPONENT_RUNTIMES[id] ?? OPPONENT_RUNTIMES[PATHMAN_ID]!;
}

function softmaxAt(values: readonly number[], index: number) {
  const maximum = Math.max(...values);
  const denominator = values.reduce((sum, value) => sum + Math.exp(value - maximum), 0);
  return denominator ? Math.exp((values[index] ?? maximum) - maximum) / denominator : 0;
}

// Keep the pure fallback imported for older callers and for environments
// without WASM; it remains a reference implementation, not a model card.
export function referencePathmanAction(state: GameState, config: SearchConfig) {
  return searchBestAction(state, config).action;
}

// Compile-time use of the rule transition here also documents the runtime
// contract's legality boundary for adapter authors.
export function applyOpponentAction(state: GameState, action: Action) {
  return applyLegalAction(state, action);
}
