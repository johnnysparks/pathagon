import assert from "node:assert/strict";
import test from "node:test";
import { PATHFINDER_SEARCH } from "../app/ai.ts";
import { compactPathfinderGameMetadata, type PathfinderMoveTelemetry } from "../app/archive-metadata.ts";
import { compactHumanGame, createGameId, validateGameId, validateHumanGame } from "../app/game-record.ts";
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
  const game = validateHumanGame({
    opponentId: "surveyor-v0",
    winner: "light",
    actions: winningActions,
    metadata: { searchExperiment: "pathfinder-browser-v1", moves: [{ positions: 1200, searchTimeMs: 2800 }] },
  });
  const compact = compactHumanGame(game);
  assert.match(compact, /^h1\tsurveyor-v0\tL\t[0-9A-Za-z_-]{26}$/);
  assert.deepEqual(game.metadata, { searchExperiment: "pathfinder-browser-v1", moves: [{ positions: 1200, searchTimeMs: 2800 }] });
});

test("human archive keeps metadata bounded and object-shaped", () => {
  assert.throws(() => validateHumanGame({ opponentId: "surveyor-v0", winner: "light", actions: winningActions, metadata: [] }), /metadata must be an object/);
  assert.throws(() => validateHumanGame({ opponentId: "surveyor-v0", winner: "light", actions: winningActions, metadata: { trace: "x".repeat(100_001) } }), /metadata is too large/);
});

test("pathfinder archive metadata samples checkpoints instead of persisting the full trace", () => {
  const checkpoints = Array.from({ length: 10_000 }, (_, index) => ({
    action: { kind: "place" as const, to: index % 49 },
    completedDepth: Math.min(100, index),
    nodes: index * 1_000,
    maxNodes: 10_000_000,
    nodeCapReached: index === 9_999,
    elapsedMs: index,
  }));
  const baseSearch: PathfinderMoveTelemetry = {
    ply: 1,
    action: { kind: "place", to: 42 },
    searchTimeMs: 30_000,
    positions: 10_000_000,
    maxNodes: 10_000_000,
    nodeCapReached: true,
    targetDepth: 100,
    completedDepth: 23,
    tableHits: 100_000,
    exhausted: true,
    interrupted: false,
    modelCard: { id: "pathfinder-action-transition-v4-xent", name: "Transition v4", version: "1", engine: "Rust/WASM" },
    config: { ...PATHFINDER_SEARCH, depth: 100, maxNodes: 10_000_000, deadlineMs: 30_000 },
    checkpoints,
  };
  const metadata = compactPathfinderGameMetadata(
    baseSearch.modelCard,
    100,
    10_000_000,
    256,
    30_000,
    Array.from({ length: 120 }, (_, index) => ({ ...baseSearch, ply: index * 2 + 1 })),
  );

  assert.ok(JSON.stringify(metadata).length < 100_000);
  assert.equal(metadata.moves.length, 120);
  assert.equal(metadata.moves[0].checkpointCount, 10_000);
  assert.equal(metadata.moves[0].checkpoints.length, 3);
  assert.equal(metadata.dials.beamWidth, 256);
  assert.equal("modelCard" in metadata.moves[0], false);
  assert.equal("weights" in metadata.moves[0].search, false);
});

test("human archive accepts versioned opponent IDs", () => {
  const game = validateHumanGame({
    opponentId: "pathfinder-v0.5.0-trained-evaluator",
    winner: "light",
    actions: winningActions,
  });
  assert.equal(game.opponentId, "pathfinder-v0.5.0-trained-evaluator");
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

test("game IDs are opaque UUID tokens", () => {
  const first = createGameId();
  const second = createGameId();
  assert.notEqual(first, second);
  validateGameId(first);
  assert.match(first, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/);
  assert.throws(() => validateGameId("surveyor-v0"));
});
