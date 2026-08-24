import { readFile } from "node:fs/promises";
import { applyLegalAction, createGame, legalActions } from "../app/pathagon.ts";
import type { Action, GameState, Player } from "../app/pathagon.ts";

type Fixture = {
  fixtureVersion: number;
  cases: Array<{ name: string; position: RawPosition }>;
};

type RawPosition = {
  config: GameState["config"];
  board: Array<Player | null>;
  reserve: Record<Player, number>;
  turn: Player;
  forbidden: number[];
  lastRelocatedTo: Record<Player, number | null>;
  winner: Player | null;
  ply: number;
};

function actionValue(action: Action) {
  return action.kind === "place" ? { kind: "place", to: action.to } : { kind: "relocate", from: action.from, to: action.to };
}

function stateValue(state: GameState) {
  return {
    board: state.board,
    reserve: { light: state.reserve.light, dark: state.reserve.dark },
    turn: state.turn,
    forbidden: [...state.forbidden].sort((left, right) => left - right),
    lastRelocatedTo: { light: state.lastRelocatedTo.light, dark: state.lastRelocatedTo.dark },
    winner: state.winner,
    ply: state.ply,
  };
}

function makeState(raw: RawPosition): GameState {
  const state = createGame(raw.config);
  state.board = [...raw.board];
  state.reserve = { light: raw.reserve.light, dark: raw.reserve.dark };
  state.turn = raw.turn;
  state.forbidden = [...raw.forbidden];
  state.lastRelocatedTo = { light: raw.lastRelocatedTo.light, dark: raw.lastRelocatedTo.dark };
  state.winner = raw.winner;
  state.ply = raw.ply;
  return state;
}

function runCase(name: string, raw: RawPosition) {
  const state = makeState(raw);
  const actions = legalActions(state);
  return {
    name,
    config: state.config,
    state: stateValue(state),
    legalActions: actions.map(actionValue),
    transitions: actions.map((action) => ({ action: actionValue(action), state: stateValue(applyLegalAction(state, action)) })),
  };
}

const fixturePath = process.argv[2];
if (!fixturePath) throw new Error("usage: parity-runner.ts <fixture.json>");
const fixture = JSON.parse(await readFile(fixturePath, "utf8")) as Fixture;
if (fixture.fixtureVersion !== 1) throw new Error("unsupported parity fixture version");
process.stdout.write(JSON.stringify(fixture.cases.map(({ name, position }) => runCase(name, position))) + "\n");
