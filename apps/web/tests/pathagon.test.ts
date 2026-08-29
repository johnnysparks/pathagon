import assert from "node:assert/strict";
import test from "node:test";
import { applyAction, createGame, legalActions } from "../app/pathagon.ts";
import type { Action, GameState, Player } from "../app/pathagon.ts";

function position(
  pieces: Partial<Record<number, Player>>,
  options: Partial<GameState> = {},
): GameState {
  const state = createGame();
  for (const [square, player] of Object.entries(pieces)) state.board[Number(square)] = player!;
  return {
    ...state,
    reserve: { light: 14, dark: 14, ...options.reserve },
    ...options,
  };
}

function play(state: GameState, action: Action) {
  return applyAction(state, action);
}

test("an empty board has 49 legal placements", () => {
  assert.equal(legalActions(createGame()).length, 49);
});

test("exact A-B-A captures automatically", () => {
  const state = position({ 21: "light", 22: "dark" }, { reserve: { light: 13, dark: 13 } });
  const next = play(state, { kind: "place", to: 23 });
  assert.equal(next.board[22], null);
  assert.equal(next.reserve.dark, 14);
  assert.deepEqual(next.forbidden, [22]);
});

test("A-B-B-A captures nothing", () => {
  const state = position({ 21: "light", 22: "dark", 23: "dark" }, { reserve: { light: 13, dark: 12 } });
  const next = play(state, { kind: "place", to: 24 });
  assert.equal(next.board[22], "dark");
  assert.equal(next.board[23], "dark");
  assert.deepEqual(next.forbidden, []);
});

test("one placement resolves every directional capture", () => {
  const state = position(
    { 9: "light", 16: "dark", 21: "light", 22: "dark", 24: "dark", 25: "light", 30: "dark", 37: "light" },
    { reserve: { light: 11, dark: 10 } },
  );
  const next = play(state, { kind: "place", to: 23 });
  assert.deepEqual(new Set(next.lastAction?.captured), new Set([16, 22, 24, 30]));
  assert.equal(next.reserve.dark, 14);
});

test("all capture holes are forbidden for one reply, then reopen", () => {
  const state = position({ 21: "light", 22: "dark" }, { reserve: { light: 13, dark: 13 } });
  const captured = play(state, { kind: "place", to: 23 });
  assert.equal(legalActions(captured).some((action) => action.to === 22), false);
  const replied = play(captured, { kind: "place", to: 0 });
  assert.deepEqual(replied.forbidden, []);
  const afterLight = play(replied, { kind: "place", to: 1 });
  assert.equal(legalActions(afterLight).some((action) => action.to === 22), true);
});

test("a forced return may itself capture", () => {
  const state = position({ 21: "light", 22: "dark", 7: "dark", 8: "light" }, { reserve: { light: 12, dark: 12 } });
  const captured = play(state, { kind: "place", to: 23 });
  const returned = play(captured, { kind: "place", to: 9 });
  assert.equal(returned.board[8], null);
  assert.deepEqual(returned.forbidden, [8]);
});

test("light wins by connecting the near edge to the far edge orthogonally", () => {
  const state = position({ 42: "light", 35: "light", 28: "light", 21: "light", 14: "light", 7: "light" }, { reserve: { light: 8, dark: 14 } });
  const next = play(state, { kind: "place", to: 0 });
  assert.equal(next.winner, "light");
  assert.equal(next.winningPath.length, 7);
  assert.equal(legalActions(next).length, 0);
});

test("dark wins by connecting left to right orthogonally", () => {
  const state = position({ 21: "dark", 22: "dark", 23: "dark", 24: "dark", 25: "dark", 26: "dark" }, { turn: "dark", reserve: { light: 14, dark: 8 } });
  const next = play(state, { kind: "place", to: 27 });
  assert.equal(next.winner, "dark");
  assert.equal(next.winningPath.length, 7);
});

test("diagonal contact does not complete a path", () => {
  const state = position({ 42: "light", 36: "light", 30: "light", 24: "light", 18: "light", 12: "light" }, { reserve: { light: 8, dark: 14 } });
  const next = play(state, { kind: "place", to: 6 });
  assert.equal(next.winner, null);
});

test("movement phase allows relocation but not the same piece twice consecutively", () => {
  const lightSquares = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26];
  const darkSquares = [28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 29, 31, 33];
  const pieces: Partial<Record<number, Player>> = {};
  for (const square of lightSquares) pieces[square] = "light";
  for (const square of darkSquares) pieces[square] = "dark";
  const state = position(pieces, { reserve: { light: 0, dark: 0 } });
  const moved = play(state, { kind: "relocate", from: 0, to: 1 });
  const afterDark = play(moved, { kind: "relocate", from: 28, to: 27 });
  assert.equal(legalActions(afterDark).some((action) => action.kind === "relocate" && action.from === 1), false);
  assert.equal(legalActions(afterDark).some((action) => action.kind === "relocate" && action.from === 2), true);
});

test("relocation can never return a picked-up piece to its source square", () => {
  const pieces: Partial<Record<number, Player>> = {};
  for (const square of [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]) pieces[square] = "light";
  for (const square of [28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 29, 31, 33]) pieces[square] = "dark";
  const state = position(pieces, { reserve: { light: 0, dark: 0 } });

  assert.equal(
    legalActions(state).some((action) => action.kind === "relocate" && action.from === action.to),
    false,
  );
  assert.throws(
    () => play(state, { kind: "relocate", from: 0, to: 0 }),
    /different square/,
  );
  assert.equal(state.board[0], "light");
});
