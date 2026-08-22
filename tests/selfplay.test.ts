import assert from "node:assert/strict";
import test from "node:test";
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
});
