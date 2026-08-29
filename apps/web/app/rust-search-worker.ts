import { chooseOpponentAction, getOpponent } from "./opponents";
import { loadRustEngine } from "./rust-engine";
import type { GameState } from "./pathagon";

type SearchRequest = {
  type: "search";
  requestId: number;
  state: GameState;
  opponentId: string;
  pathfinderDepth: number;
  deadlineMs: number;
};

type CancelRequest = { type: "cancel"; requestId: number };
type WorkerRequest = SearchRequest | CancelRequest;

const cancelled = new Set<number>();
const enginePromise = loadRustEngine();

self.addEventListener("message", (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  if (request.type === "cancel") {
    cancelled.add(request.requestId);
    return;
  }

  void enginePromise.then((engine) => {
    if (cancelled.delete(request.requestId)) return;
    const opponent = getOpponent(request.opponentId);
    const action = chooseOpponentAction(
      engine,
      opponent,
      request.state,
      undefined,
      request.pathfinderDepth,
      request.deadlineMs,
    );
    if (cancelled.delete(request.requestId)) return;
    self.postMessage({ type: "result", requestId: request.requestId, action });
  }).catch((error: unknown) => {
    if (cancelled.delete(request.requestId)) return;
    self.postMessage({
      type: "error",
      requestId: request.requestId,
      message: error instanceof Error ? error.message : String(error),
    });
  });
});
