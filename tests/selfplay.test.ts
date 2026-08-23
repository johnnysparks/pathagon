import assert from "node:assert/strict";
import test from "node:test";
import { DEFAULT_WEIGHTS, searchBestAction } from "../app/ai.ts";
import { createGame } from "../app/pathagon.ts";
import { createRandomAgent, playGame } from "../selfplay/core.ts";

test("seeded self-play is reproducible", () => {
  const options = { seed: 42, maxPlies: 60, openingRandomPlies: 2 };
  const first = playGame(createRandomAgent("light-random"), createRandomAgent("dark-random"), options);
  const second = playGame(createRandomAgent("light-random"), createRandomAgent("dark-random"), options);
  assert.deepEqual(second, first);
});

test("self-play records every applied move", () => {
  const record = playGame(
    createRandomAgent("light-random"),
    createRandomAgent("dark-random"),
    { seed: 7, maxPlies: 30, openingRandomPlies: 0 },
  );
  assert.equal(record.plies, record.moves.length);
  assert.ok(record.moves.every((move, index) => move.ply === index + 1));
  assert.ok(record.moves.every((move) => Number.isInteger(move.completedDepth) && Number.isInteger(move.tableHits)));
});

test("iterative search returns the last completed depth inside its node budget", () => {
  const result = searchBestAction(createGame(), { depth: 5, maxNodes: 120, beamWidth: 49, weights: DEFAULT_WEIGHTS });
  assert.ok(result.action);
  assert.ok(result.nodes <= 120);
  assert.equal(result.exhausted, true);
  assert.ok(result.completedDepth >= 1 && result.completedDepth < 5);
});
