import assert from "node:assert/strict";
import test from "node:test";
import { COACHING_SEARCH, DEFAULT_WEIGHTS, analyzeAction, analyzeActions, searchBestAction } from "../app/ai.ts";
import {
  LUNATIC_OPPONENT,
  PATHFINDER_BEAM_OPTIONS,
  PATHFINDER_DEPTH_OPTIONS,
  PATHFINDER_MAX_NODES_HARD_CAP,
  PATHFINDER_OPPONENT,
  TRANSITION_PATHFINDER_OPPONENT,
  TRAINED_PATHFINDER_OPPONENT,
  TRAINED_PATHFINDER_WEIGHTS,
  pathfinderMaxNodesForDepth,
  pathfinderSearchAtDepth,
  trainedPathfinderSearchAtDepth,
  SURVEYOR_OPPONENT,
  OPPONENTS,
  DEFAULT_OPPONENT_ID,
  getOpponent,
} from "../app/opponents.ts";
import { applyAction, createGame, legalActions } from "../app/pathagon.ts";
import type { GameState, Player } from "../app/pathagon.ts";
import { DOUBLE_DRAGON_ID, PATHFINDER_TACTICAL_FILTER_ID, PATHMAN_ID, RANDO_RACCON_ID, SEER_ID, TILE_DRIVER_ID, TRANSITION_PATHFINDER_ID, TRAINED_PATHFINDER_ID, YANN_TILESON_ID } from "../app/agent-ids.ts";
import type { OpponentRuntimeContext } from "../app/opponent-runtime.ts";

function position(pieces: Partial<Record<number, Player>>, options: Partial<GameState> = {}) {
  const state = createGame();
  for (const [square, player] of Object.entries(pieces)) state.board[Number(square)] = player!;
  return { ...state, ...options, reserve: { light: 14, dark: 14, ...options.reserve } };
}

test("The Surveyor takes an immediate win", () => {
  const state = position(
    { 21: "dark", 22: "dark", 23: "dark", 24: "dark", 25: "dark", 26: "dark" },
    { turn: "dark", reserve: { light: 14, dark: 8 } },
  );
  const action = SURVEYOR_OPPONENT.chooseAction(state);
  assert.deepEqual(action, { kind: "place", to: 27 });
  assert.equal(applyAction(state, action!).winner, "dark");
});

test("The Surveyor blocks an immediate human win", () => {
  const state = position(
    { 42: "light", 35: "light", 28: "light", 21: "light", 14: "light", 7: "light" },
    { turn: "dark", reserve: { light: 8, dark: 14 } },
  );
  assert.deepEqual(SURVEYOR_OPPONENT.chooseAction(state), { kind: "place", to: 0 });
});

test("The Surveyor values a free automatic capture", () => {
  const state = position(
    { 21: "dark", 22: "light" },
    { turn: "dark", reserve: { light: 13, dark: 13 } },
  );
  const action = SURVEYOR_OPPONENT.chooseAction(state);
  assert.deepEqual(action, { kind: "place", to: 23 });
  assert.deepEqual(applyAction(state, action!).lastAction?.captured, [22]);
});

test("The Surveyor sees the right-edge rush and blocks before the capture ladder", () => {
  const state = position(
    { 48: "light", 41: "light", 34: "light", 27: "dark", 26: "dark" },
    { turn: "dark", reserve: { light: 11, dark: 12 }, ply: 5 },
  );
  const action = SURVEYOR_OPPONENT.chooseAction(state);
  assert.deepEqual(action, { kind: "place", to: 20 });
});

test("Pathfinder look-ahead stays within the browser-safe experiment envelope", () => {
  const quick = pathfinderSearchAtDepth(PATHFINDER_DEPTH_OPTIONS[0]);
  const balanced = pathfinderSearchAtDepth(PATHFINDER_OPPONENT.searchDepth!);
  const deep = pathfinderSearchAtDepth(PATHFINDER_DEPTH_OPTIONS.at(-1)!);
  const long = pathfinderSearchAtDepth(20);
  const longTwentyOne = pathfinderSearchAtDepth(21);
  const longTwentyTwo = pathfinderSearchAtDepth(22);
  const longTwentyThree = pathfinderSearchAtDepth(23);
  const extreme = pathfinderSearchAtDepth(50);
  assert.equal(quick.depth, 2);
  assert.equal(balanced.depth, 5);
  assert.equal(deep.depth, 100);
  assert.equal(long.depth, 20);
  assert.equal(longTwentyOne.depth, 21);
  assert.equal(longTwentyTwo.depth, 22);
  assert.equal(longTwentyThree.depth, 23);
  assert.equal(extreme.depth, 50);
  assert.ok(quick.maxNodes < balanced.maxNodes);
  assert.ok(balanced.maxNodes < deep.maxNodes);
  assert.ok(quick.beamWidth > deep.beamWidth);
  assert.equal(balanced.maxNodes, 256_000);
  assert.equal(balanced.beamWidth, 256);
  const rollback = pathfinderSearchAtDepth(4);
  assert.equal(rollback.depth, 4);
  assert.equal(rollback.maxNodes, 32_000);
  assert.equal(rollback.beamWidth, 256);
  assert.equal(pathfinderMaxNodesForDepth(4), 32_000);
  assert.equal(pathfinderMaxNodesForDepth(5), 256_000);
  assert.equal(pathfinderSearchAtDepth(99).depth, 99);
  assert.equal(pathfinderSearchAtDepth(-10).depth, 2);
  assert.equal(pathfinderSearchAtDepth(23, 50_000_000).maxNodes, PATHFINDER_MAX_NODES_HARD_CAP);
  assert.equal(pathfinderMaxNodesForDepth(23), 1_000_000);
  assert.equal(pathfinderMaxNodesForDepth(50), 5_000_000);
  assert.equal(pathfinderMaxNodesForDepth(100), PATHFINDER_MAX_NODES_HARD_CAP);
});

test("trained Pathfinder keeps its promoted search envelope and evaluator weights", () => {
  const config = trainedPathfinderSearchAtDepth(TRAINED_PATHFINDER_OPPONENT.searchDepth!);
  assert.equal(PATHFINDER_OPPONENT.id, PATHFINDER_TACTICAL_FILTER_ID);
  assert.equal(TRAINED_PATHFINDER_OPPONENT.id, TRAINED_PATHFINDER_ID);
  assert.deepEqual(config.weights, TRAINED_PATHFINDER_WEIGHTS);
  assert.equal(config.depth, 5);
  assert.equal(config.maxNodes, 256_000);
  assert.equal(config.beamWidth, 256);
});

test("Pathfinder exposes a bounded beam-width override for user play", () => {
  assert.equal(pathfinderSearchAtDepth(5, 256_000, PATHFINDER_BEAM_OPTIONS[0]).beamWidth, 2);
  assert.equal(pathfinderSearchAtDepth(5, 256_000, 512).beamWidth, 512);
  assert.equal(trainedPathfinderSearchAtDepth(5, 256_000, PATHFINDER_BEAM_OPTIONS.at(-1)).beamWidth, 4_096);
  assert.equal(pathfinderSearchAtDepth(5, 256_000, 50_000).beamWidth, 4_096);
});

test("transition-policy v4 is the strongest user-facing Pathfinder identity", () => {
  assert.equal(TRANSITION_PATHFINDER_OPPONENT.id, TRANSITION_PATHFINDER_ID);
  assert.equal(TRANSITION_PATHFINDER_OPPONENT.shortName, "Pathfinder · v4");
  assert.equal(TRANSITION_PATHFINDER_OPPONENT.version, "4.0.0");
  assert.equal(TRANSITION_PATHFINDER_OPPONENT.searchDepth, 5);
  const action = TRANSITION_PATHFINDER_OPPONENT.chooseAction(createGame());
  assert.ok(action);
  assert.ok(legalActions(createGame()).some((candidate) => JSON.stringify(candidate) === JSON.stringify(action)));
});

test("Lunatic takes an obvious automatic capture", () => {
  const state = position(
    { 21: "dark", 22: "light" },
    { turn: "dark", reserve: { light: 13, dark: 13 } },
  );
  const action = LUNATIC_OPPONENT.chooseAction(state);
  assert.deepEqual(action, { kind: "place", to: 23 });
  assert.deepEqual(applyAction(state, action!).lastAction?.captured, [22]);
});

test("Lunatic takes an immediate win before chasing local patterns", () => {
  const state = position(
    { 21: "dark", 22: "dark", 23: "dark", 24: "dark", 25: "dark", 26: "dark" },
    { turn: "dark", reserve: { light: 14, dark: 8 } },
  );
  const action = LUNATIC_OPPONENT.chooseAction(state);
  assert.deepEqual(action, { kind: "place", to: 27 });
  assert.equal(applyAction(state, action!).winner, "dark");
});

test("Lunatic always returns a legal move in movement phase", () => {
  const state = position(
    { 0: "dark", 2: "dark", 4: "dark", 6: "dark", 42: "light", 44: "light", 46: "light", 48: "light" },
    { turn: "dark", reserve: { light: 0, dark: 0 } },
  );
  const action = LUNATIC_OPPONENT.chooseAction(state);
  assert.ok(action);
  assert.ok(legalActions(state).some((candidate) => JSON.stringify(candidate) === JSON.stringify(action)));
});

test("move coaching evaluates a legal preview and reports its balance shift", () => {
  const state = createGame();
  const evaluation = analyzeAction(state, { kind: "place", to: 24 }, { ...COACHING_SEARCH, depth: 1, maxNodes: 100 });
  assert.deepEqual(evaluation.action, { kind: "place", to: 24 });
  assert.equal(evaluation.delta, evaluation.score);
  assert.ok(evaluation.nodes > 0);
});

test("move coaching sorts the visible heatmap from best to worst", () => {
  const moves = analyzeActions(createGame(), { ...COACHING_SEARCH, depth: 1, maxNodes: 500 }, 12);
  assert.equal(moves.length, 12);
  assert.ok(moves.every((move, index) => index === 0 || moves[index - 1].score >= move.score));
});

test("iterative search returns the last completed depth inside its node budget", () => {
  const result = searchBestAction(createGame(), { depth: 5, maxNodes: 120, beamWidth: 49, weights: DEFAULT_WEIGHTS });
  assert.ok(result.action);
  assert.ok(result.nodes <= 120);
  assert.equal(result.exhausted, true);
  assert.ok(result.completedDepth >= 1 && result.completedDepth < 5);
});

test("player-facing roster has six cute names and Pathman is the default", () => {
  assert.deepEqual(OPPONENTS.map((opponent) => opponent.name), ["Pathman", "Tile Driver", "Seer", "Double Dragon", "Yann Tileson", "Rando Raccon"]);
  assert.equal(DEFAULT_OPPONENT_ID, PATHMAN_ID);
  assert.equal(getOpponent("pathfinder-action-transition-v4-xent").id, PATHMAN_ID);
  assert.equal(getOpponent("coin-flip-v0.0.1").id, RANDO_RACCON_ID);
  assert.ok(OPPONENTS.every((opponent) => opponent.capabilities.length === 3));
  assert.ok(OPPONENTS.every((opponent) => opponent.controls.every((control) => control.values.length === 5)));
  assert.deepEqual(OPPONENTS.map((opponent) => opponent.id), [PATHMAN_ID, TILE_DRIVER_ID, SEER_ID, DOUBLE_DRAGON_ID, YANN_TILESON_ID, RANDO_RACCON_ID]);
});

test("ready opponent runtimes rank legal actions and return a legal action", () => {
  const state = createGame();
  const actions = legalActions(state);
  const fakeRust = {
    legalActions: (current: GameState) => legalActions(current),
    applyAction: (current: GameState, action: import("../app/pathagon.ts").Action) => applyAction(current, action),
    analyzeActions: () => actions.map((action, index) => ({ action, beforeScore: 0, score: actions.length - index, delta: 0, nodes: 1, exhausted: false, completedDepth: 1, tableHits: 0 })),
    searchBestTacticalActionWithDeadlineTrace: () => ({ action: actions[0] ?? null, score: 1, nodes: 3, exhausted: false, completedDepth: 1, tableHits: 0 }),
  } as unknown as OpponentRuntimeContext["rustEngine"];
  const fakeCnn = {
    evaluate: () => ({ actions, policyLogits: actions.map((_, index) => actions.length - index), value: 0.25 }),
    selectAction: () => ({ action: actions[0] ?? null, value: 0.25, simulations: 64, evaluations: actions.map((action, index) => ({ action, prior: 1 / actions.length, visits: actions.length - index, value: 0.25 })) }),
  } as unknown as NonNullable<OpponentRuntimeContext["cnnEngine"]>;
  const fakeGnn = {
    evaluate: () => ({ actions, policyLogits: actions.map((_, index) => actions.length - index), value: 0.2 }),
    selectAction: () => ({ action: actions[0] ?? null, value: 0.2, simulations: 32, evaluations: actions.map((action, index) => ({ action, prior: 1 / actions.length, visits: actions.length - index, value: 0.2 })) }),
  } as unknown as NonNullable<OpponentRuntimeContext["gnnEngine"]>;
  const fakeQadv = {
    evaluate: () => ({ actions, policyLogits: actions.map((_, index) => actions.length - index), qValues: actions.map((_, index) => (actions.length - index) / actions.length), value: 0.15 }),
    selectAction: () => ({ action: actions[0] ?? null, value: 0.15, simulations: 32, evaluations: actions.map((action, index) => ({ action, prior: 1 / actions.length, visits: actions.length - index, value: (actions.length - index) / actions.length })) }),
  } as unknown as NonNullable<OpponentRuntimeContext["qadvEngine"]>;
  const fakeJepa = {
    evaluate: () => ({ actions, rankLogits: actions.map((_, index) => actions.length - index), actionValues: actions.map((_, index) => (actions.length - index) / actions.length) }),
  } as unknown as NonNullable<OpponentRuntimeContext["jepaEngine"]>;
  const context = { rustEngine: fakeRust, cnnEngine: fakeCnn, gnnEngine: fakeGnn, qadvEngine: fakeQadv, jepaEngine: fakeJepa };
  for (const opponent of OPPONENTS) {
    const config = { controls: Object.fromEntries(opponent.controls.map((control) => [control.id, control.values[control.defaultIndex]])), seed: 17 };
    const runtimeContext = context as OpponentRuntimeContext;
    const ranked = opponent.runtime.rankMoves(state, runtimeContext, config);
    assert.ok(ranked.every((candidate) => actions.some((action) => JSON.stringify(action) === JSON.stringify(candidate.action))), opponent.name);
    const result = opponent.runtime.search(state, runtimeContext, config);
    assert.ok(result.action && actions.some((action) => JSON.stringify(action) === JSON.stringify(result.action)), opponent.name);
  }
});

test("Rando Raccon is seeded, bounded, and honestly labeled", () => {
  const state = createGame();
  const opponent = getOpponent(RANDO_RACCON_ID);
  const config = { controls: Object.fromEntries(opponent.controls.map((control) => [control.id, control.values[control.defaultIndex]])), seed: 91 };
  const context = {
    rustEngine: {
      legalActions: (current: GameState) => legalActions(current),
      applyAction: (current: GameState, action: import("../app/pathagon.ts").Action) => applyAction(current, action),
    },
  } as unknown as OpponentRuntimeContext;
  const first = opponent.runtime.search(state, context, config);
  const second = opponent.runtime.search(state, context, config);
  const different = opponent.runtime.search(state, context, { ...config, seed: 92 });
  assert.deepEqual(first.ranked, second.ranked);
  assert.notDeepEqual(first.ranked, different.ranked);
  assert.equal(first.interpretation, "random priority/order");
  assert.ok(first.ranked.every((candidate) => candidate.randomPriority !== undefined));
  assert.ok(first.ranked.length <= config.controls.samples);
});

test("learned cards are real playable runtimes after artifact promotion", () => {
  for (const id of [TILE_DRIVER_ID, DOUBLE_DRAGON_ID, YANN_TILESON_ID]) {
    const opponent = getOpponent(id);
    assert.equal(opponent.playable, true);
    assert.equal(opponent.status, "ready");
    assert.equal(opponent.artifact?.startsWith("/models/"), true);
  }
  assert.equal(getOpponent(SEER_ID).playable, true);
});
