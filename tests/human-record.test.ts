import assert from "node:assert/strict";
import test from "node:test";
import { compactHumanGame, validateHumanGame } from "../app/game-record.ts";
import type { Action } from "../app/pathagon.ts";

const winningActions: Action[] = [
  { kind: "place", to: 42 }, { kind: "place", to: 48 },
  { kind: "place", to: 35 }, { kind: "place", to: 47 },
  { kind: "place", to: 28 }, { kind: "place", to: 46 },
  { kind: "place", to: 21 }, { kind: "place", to: 45 },
  { kind: "place", to: 14 }, { kind: "place", to: 44 },
  { kind: "place", to: 7 }, { kind: "place", to: 43 },
  { kind: "place", to: 0 },
];

test("completed human games replay before compact encoding", () => {
  const game = validateHumanGame({ opponentId: "surveyor-v0", winner: "light", actions: winningActions });
  const compact = compactHumanGame(game);
  assert.match(compact, /^h1\tsurveyor-v0\tL\t[0-9A-Za-z_-]{26}$/);
});

test("human archive rejects a result that replay does not prove", () => {
  assert.throws(
    () => validateHumanGame({ opponentId: "surveyor-v0", winner: "dark", actions: winningActions }),
    /does not match/,
  );
});

test("human archive rejects illegal and oversized action streams", () => {
  assert.throws(() => validateHumanGame({ opponentId: "surveyor-v0", winner: "light", actions: [{ kind: "place", to: 99 }] }));
  assert.throws(() => validateHumanGame({ opponentId: "surveyor-v0", winner: "light", actions: Array(241).fill({ kind: "place", to: 0 }) }));
});
