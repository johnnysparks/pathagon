import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { validateContractReplay, validatePosition } from "../app/contract.ts";
import { validateSelfPlayRecord } from "../app/selfplay-record.ts";

test("canonical contract fixture validates in TypeScript", async () => {
  const fixture = JSON.parse(await readFile(new URL("../../../pathagon/contracts/fixtures/replay-v1.json", import.meta.url), "utf8"));
  const record = validateContractReplay(fixture);
  assert.deepEqual(validateSelfPlayRecord(fixture), fixture);
  assert.equal(record.contractVersion, 1);
  assert.equal(record.config.boardSize, 3);
  assert.equal(record.agentSpecifications.light.id, record.agents.light);
  assert.equal(record.agentSpecifications.light.manifest.runtime, "typescript");
  assert.equal(record.agentSpecifications.light.manifest.nodeBudget, 0);
});

test("root-Q targets round-trip and reject partial archives", async () => {
  const fixture = JSON.parse(await readFile(new URL("../../../pathagon/contracts/fixtures/replay-v1.json", import.meta.url), "utf8"));
  fixture.moves[0].actionValues = [-0.25, 0.75];
  fixture.moves[0].actionVisits = [2, 10];
  fixture.moves[0].actionValueSource = "mcts-root-q-v1";
  const record = validateContractReplay(fixture);
  assert.deepEqual(record.moves[0].actionValues, [-0.25, 0.75]);
  assert.deepEqual(record.moves[0].actionVisits, [2, 10]);

  const replayFixture = JSON.parse(JSON.stringify(fixture));
  replayFixture.moves[0].actionValues = Array(9).fill(-0.25);
  replayFixture.moves[0].actionVisits = Array(9).fill(2);
  assert.equal(validateSelfPlayRecord(replayFixture).moves[0].actionValues?.length, 9);

  const partial = { ...fixture, moves: fixture.moves.map((move: Record<string, unknown>) => ({ ...move, actionVisits: [2] })) };
  assert.throws(() => validateContractReplay(partial), /root-Q alignment/);
});

test("contract positions carry the complete rule-relevant state", () => {
  const position = validatePosition({
    contractVersion: 1,
    config: { rulesVersion: "pathagon-rules-v1", boardSize: 3, reservePerPlayer: 6, maxPlies: 36, repetitionLimit: 3 },
    board: ["light", null, null, null, "dark", null, null, null, null],
    reserve: { light: 5, dark: 6 },
    turn: "dark",
    forbidden: [],
    lastRelocatedTo: { light: null, dark: null },
    winner: null,
    ply: 1,
  });
  assert.equal(position.board.length, 9);
  assert.equal(position.turn, "dark");
});
