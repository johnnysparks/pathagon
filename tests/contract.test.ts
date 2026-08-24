import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { validateContractReplay, validatePosition } from "../app/contract.ts";
import { validateSelfPlayRecord } from "../app/selfplay-record.ts";

test("canonical contract fixture validates in TypeScript", async () => {
  const fixture = JSON.parse(await readFile(new URL("../contracts/fixtures/replay-v1.json", import.meta.url), "utf8"));
  const record = validateContractReplay(fixture);
  assert.deepEqual(validateSelfPlayRecord(fixture), fixture);
  assert.equal(record.contractVersion, 1);
  assert.equal(record.config.boardSize, 3);
  assert.equal(record.agentSpecifications.light.id, record.agents.light);
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
