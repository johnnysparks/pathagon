import assert from "node:assert/strict";
import test from "node:test";
import { COACHING_SEARCH, analyzeAction, analyzeActions } from "../app/ai.ts";
import { LUNATIC_OPPONENT, SURVEYOR_OPPONENT } from "../app/opponents.ts";
import { applyAction, createGame, legalActions } from "../app/pathagon.ts";
import type { GameState, Player } from "../app/pathagon.ts";

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
