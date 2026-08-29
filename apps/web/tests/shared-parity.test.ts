import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { applyAction, createGame } from "../app/pathagon.ts";
import type { Action, Player } from "../app/pathagon.ts";

test("shared rule fixtures match the TypeScript engine", async () => {
  const fixture = await readFile(new URL("../../../data/fixtures/rules-parity.tsv", import.meta.url), "utf8");
  for (const line of fixture.split("\n")) {
    if (!line || line.startsWith("#")) continue;
    const [name, placements, turn, lightReserve, darkReserve, forbidden, lastLight, lastDark, encodedAction, legal, winner, captured] = line.split("\t");
    const state = createGame();
    if (placements !== "-") {
      for (const placement of placements.split(",")) {
        const color = placement.at(-1) === "L" ? "light" : "dark";
        state.board[Number(placement.slice(0, -1))] = color;
      }
    }
    state.turn = turn as Player;
    state.reserve = { light: Number(lightReserve), dark: Number(darkReserve) };
    state.forbidden = squares(forbidden);
    state.lastRelocatedTo = { light: optionalSquare(lastLight), dark: optionalSquare(lastDark) };
    const action = parseAction(encodedAction);
    let next;
    try {
      next = applyAction(state, action);
    } catch {
      next = null;
    }
    assert.equal(Boolean(next), legal === "true", `${name}: legality mismatch`);
    if (!next) continue;
    assert.equal(next.winner, winner === "-" ? null : winner, `${name}: winner mismatch`);
    assert.deepEqual([...(next.lastAction?.captured ?? [])].sort((a, b) => a - b), squares(captured), `${name}: captures mismatch`);
  }
});

function parseAction(value: string): Action {
  if (value.startsWith("P")) return { kind: "place", to: Number(value.slice(1)) };
  const [from, to] = value.slice(1).split(">").map(Number);
  return { kind: "relocate", from, to };
}

function squares(value: string) {
  return value === "-" ? [] : value.split(",").map(Number);
}

function optionalSquare(value: string) {
  return value === "-" ? null : Number(value);
}
