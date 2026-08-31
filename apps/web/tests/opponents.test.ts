import assert from "node:assert/strict";
import test from "node:test";
import { COACHING_SEARCH, DEFAULT_WEIGHTS, analyzeAction, analyzeActions, searchBestAction } from "../app/ai.ts";
import {
  LUNATIC_OPPONENT,
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
} from "../app/opponents.ts";
import { applyAction, createGame, legalActions } from "../app/pathagon.ts";
import type { GameState, Player } from "../app/pathagon.ts";
import { PATHFINDER_TACTICAL_FILTER_ID, TRANSITION_PATHFINDER_ID, TRAINED_PATHFINDER_ID } from "../app/agent-ids.ts";

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
