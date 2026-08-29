import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { createGameFromPosition, legalActions, type Player } from "../app/pathagon.ts";

const lines = (await readFile(new URL("../../../data/fixtures/pathfinder-browser-suite-v1.jsonl", import.meta.url), "utf8"))
  .trim()
  .split("\n")
  .map((line) => JSON.parse(line) as Record<string, unknown>);
const header = lines[0];

test("the durable Pathfinder browser suite has stable legal-action counts", () => {
  assert.equal(header.schema, "pathagon-search-browser-suite-v1");
  assert.equal(header.count, lines.length - 1);
  const records = lines.slice(1);
  const ids = records.map((record) => record.id);
  assert.equal(new Set(ids).size, records.length);

  for (const record of records) {
    const state = record.state as {
      light: number[];
      dark: number[];
      reserve: [number, number];
      turn: Player;
      forbidden: number[];
      lastRelocatedTo: [number | null, number | null];
      ply: number;
    };
    const board = Array<Player | null>(49).fill(null);
    for (const square of state.light) board[square] = "light";
    for (const square of state.dark) board[square] = "dark";
    const position = createGameFromPosition({
      contractVersion: 1,
      config: {
        rulesVersion: "pathagon-rules-v1",
        boardSize: 7,
        reservePerPlayer: 14,
        maxPlies: 180,
        repetitionLimit: 3,
      },
      board,
      reserve: { light: state.reserve[0], dark: state.reserve[1] },
      turn: state.turn,
      forbidden: state.forbidden,
      lastRelocatedTo: { light: state.lastRelocatedTo[0], dark: state.lastRelocatedTo[1] },
      lastCapture: 0,
      lastPlayer: null,
      winner: null,
      winningPath: [],
      ply: state.ply,
    });
    assert.equal(legalActions(position).length, record.expectedLegalActions, String(record.id));
  }
});
