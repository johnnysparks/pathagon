import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { validateSelfPlayRecord } from "../app/selfplay-record.ts";

type Fixture = {
  config: { maxPlies: number };
  moves: Array<Record<string, unknown>>;
};

async function fixture() {
  return JSON.parse(await readFile(new URL("../contracts/fixtures/replay-v1.json", import.meta.url), "utf8")) as Fixture;
}

test("self-play validation rejects target arrays that are not aligned to legal actions", async () => {
  const record = await fixture();
  record.moves[0].policy = [1];
  assert.throws(() => validateSelfPlayRecord(record), /Invalid self-play policy/);

  const withRootQ = await fixture();
  withRootQ.moves[0].actionValues = Array(9).fill(0);
  withRootQ.moves[0].actionVisits = Array(9).fill(1);
  withRootQ.moves[0].actionValueSource = "mcts-root-q-v1";
  assert.equal(validateSelfPlayRecord(withRootQ).moves[0].actionValues?.length, 9);
  withRootQ.moves[0].actionValues = [0];
  assert.throws(() => validateSelfPlayRecord(withRootQ), /root-Q alignment/);
});

test("self-play validation proves max-plies termination", async () => {
  const record = await fixture();
  record.config.maxPlies = 2;
  assert.throws(() => validateSelfPlayRecord(record), /does not reach the configured limit/);
});

test("self-play validation keeps schema-v2 records without nested config readable", async () => {
  const record = await fixture() as Fixture & Record<string, unknown>;
  delete record.contractVersion;
  delete record.config;
  delete record.agentSpecifications;
  record.schemaVersion = 2;
  record.boardSize = 3;
  record.reservePerPlayer = 6;
  record.engine = "rust";
  const normalized = validateSelfPlayRecord(record);
  assert.equal(normalized.config.maxPlies, 1);
  assert.equal(normalized.engine.runtime, "rust");
});

test("self-play validation catches replay tampering", async () => {
  const record = await fixture();
  record.moves[0].captured = [1];
  assert.throws(() => validateSelfPlayRecord(record), /capture data does not match replay/);
});
