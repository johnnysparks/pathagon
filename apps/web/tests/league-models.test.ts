import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { LATEST_RESEARCH, LEAGUE_MODELS, RANKED_LEAGUE_MODELS } from "../app/league-models.ts";

const rankedIds = [
  "pathfinder-action-transition-v4-xent",
  "pathfinder-v0.5.0-trained-evaluator",
  "pathfinder-v0.4.0-tactical-filter",
  "surveyor-v0.2.0",
  "lunatic-v0.1.0",
  "coin-flip-v0.0.1",
];

test("official league contains only Rust-engine opponents", () => {
  assert.deepEqual(RANKED_LEAGUE_MODELS.map((model) => model.id), rankedIds);
  assert.ok(RANKED_LEAGUE_MODELS.every((model) => model.rustEngine));
  assert.ok(LEAGUE_MODELS.some((model) => !model.rustEngine));
  assert.ok(LEAGUE_MODELS.filter((model) => !model.rustEngine).every((model) => !rankedIds.includes(model.id)));
});

test("latest research ledger matches the promoted v4 manifest", async () => {
  const manifestText = await readFile(new URL("../../../data/models/pathfinder-action-transition-v4-xent/manifest.json", import.meta.url), "utf8");
  const manifest = JSON.parse(manifestText) as Record<string, unknown>;
  assert.equal(LATEST_RESEARCH.artifactId, manifest.artifactId);
  assert.equal(LATEST_RESEARCH.modelHash, manifest.sha256);
  assert.equal(LATEST_RESEARCH.heldoutTop1, manifest.heldoutTop1);
  assert.equal(LATEST_RESEARCH.heldoutTop3, manifest.heldoutTop3);
  assert.equal(LATEST_RESEARCH.arenaGames, manifest.arenaGames);
  assert.equal(LATEST_RESEARCH.arenaWins, manifest.arenaWins);
  assert.equal(LATEST_RESEARCH.arenaLosses, manifest.arenaLosses);
  assert.equal(LATEST_RESEARCH.arenaDraws, manifest.arenaDraws);
  assert.equal(manifest.promotionStatus, "supported-user-facing-default");
});
