import { applyAction, createGame } from "./pathagon.ts";
import type { Action, Player } from "./pathagon.ts";

const ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_";

export type HumanGameSubmission = {
  opponentId: string;
  winner: Player;
  actions: Action[];
};

export function validateHumanGame(value: unknown): HumanGameSubmission {
  if (!value || typeof value !== "object") throw new Error("Game record must be an object");
  const input = value as Partial<HumanGameSubmission>;
  if (typeof input.opponentId !== "string" || !/^[a-z0-9-]{1,64}$/.test(input.opponentId)) {
    throw new Error("Invalid opponent");
  }
  if (input.winner !== "light" && input.winner !== "dark") throw new Error("Invalid winner");
  if (!Array.isArray(input.actions) || input.actions.length < 1 || input.actions.length > 240) {
    throw new Error("Invalid action count");
  }
  let state = createGame();
  const actions = input.actions.map((candidate) => {
    if (!candidate || typeof candidate !== "object") throw new Error("Invalid action");
    const action = candidate as Partial<Action> & { from?: unknown; to?: unknown };
    if (!Number.isInteger(action.to) || Number(action.to) < 0 || Number(action.to) >= 49) {
      throw new Error("Invalid destination");
    }
    const normalized: Action = action.kind === "place"
      ? { kind: "place", to: Number(action.to) }
      : action.kind === "relocate" && Number.isInteger(action.from) && Number(action.from) >= 0 && Number(action.from) < 49
        ? { kind: "relocate", from: Number(action.from), to: Number(action.to) }
        : (() => { throw new Error("Invalid action kind"); })();
    state = applyAction(state, normalized);
    return normalized;
  });
  if (state.winner !== input.winner || state.ply !== actions.length) {
    throw new Error("Recorded result does not match replayed result");
  }
  return { opponentId: input.opponentId, winner: input.winner, actions };
}

export function compactHumanGame(game: HumanGameSubmission) {
  return `h1\t${game.opponentId}\t${game.winner === "light" ? "L" : "D"}\t${game.actions.map(encodeAction).join("")}`;
}

export function encodeAction(action: Action) {
  const code = action.kind === "place" ? action.to : 49 + action.from * 49 + action.to;
  return `${ALPHABET[code >> 6]}${ALPHABET[code & 63]}`;
}
