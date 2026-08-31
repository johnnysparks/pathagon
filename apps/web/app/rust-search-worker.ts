import { chooseOpponentAction, getOpponent } from "./opponents";
import { TRANSITION_PATHFINDER_ID } from "./agent-ids";
import { loadRustEngine, loadTransitionPolicyEngine } from "./rust-engine";
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
let transitionPolicyPromise: ReturnType<typeof loadTransitionPolicyEngine> | null = null;

function loadDefaultTransitionPolicy() {
  transitionPolicyPromise ??= loadTransitionPolicyEngine();
  return transitionPolicyPromise;
}

self.addEventListener("message", (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  if (request.type === "cancel") {
    cancelled.add(request.requestId);
    return;
  }

  void enginePromise.then(async (engine) => {
    if (cancelled.delete(request.requestId)) return;
    const opponent = getOpponent(request.opponentId);
    const transitionPolicy = request.opponentId === TRANSITION_PATHFINDER_ID
      ? await loadDefaultTransitionPolicy()
      : undefined;
    const action = chooseOpponentAction(
      engine,
      opponent,
      request.state,
      undefined,
      request.pathfinderDepth,
      request.deadlineMs,
      transitionPolicy,
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
