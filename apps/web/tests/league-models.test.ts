import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { ARCHIVE_LEAGUE_MODELS, LEGACY_LEAGUE_MODELS, LATEST_RESEARCH, LEAGUE_MODELS, RANKED_LEAGUE_MODELS } from "../app/league-models.ts";
import { DOUBLE_DRAGON_ID, PATHMAN_ID, RANDO_RACCON_ID, SEER_ID, TILE_DRIVER_ID, YANN_TILESON_ID } from "../app/agent-ids.ts";

const rankedIds = [
  PATHMAN_ID,
  TILE_DRIVER_ID,
  SEER_ID,
  DOUBLE_DRAGON_ID,
  YANN_TILESON_ID,
  RANDO_RACCON_ID,
];

test("official league contains only Rust-engine opponents", () => {
  assert.deepEqual(RANKED_LEAGUE_MODELS.map((model) => model.id), rankedIds);
  assert.ok(RANKED_LEAGUE_MODELS.every((model) => model.rustEngine));
  assert.equal(LEAGUE_MODELS.length, 6);
  assert.deepEqual(LEAGUE_MODELS.map((model) => model.name), ["Pathman", "Tile Driver", "Seer", "Double Dragon", "Yann Tileson", "Rando Raccon"]);
  assert.equal(LEAGUE_MODELS.find((model) => model.id === PATHMAN_ID)?.status, "default");
  assert.ok(LEAGUE_MODELS.every((model) => model.playable));
  assert.ok(LEGACY_LEAGUE_MODELS.some((model) => model.id === "pathfinder-action-transition-v4-xent"));
  assert.ok(ARCHIVE_LEAGUE_MODELS.length > LEAGUE_MODELS.length);
});

test("canonical league controls preserve the five-step budget contract", () => {
  assert.ok(LEAGUE_MODELS.every((model) => model.budget.length > 0));
  assert.ok(LEAGUE_MODELS.filter((model) => model.playable).every((model) => model.rustEngine));
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
