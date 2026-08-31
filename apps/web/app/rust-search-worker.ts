import { TRANSITION_PATHFINDER_ID, TRAINED_PATHFINDER_ID } from "./agent-ids";
import { pathfinderSearchAtDepth, trainedPathfinderSearchAtDepth } from "./opponents";
import { loadRustEngine, loadTransitionPolicyEngine } from "./rust-engine";
import type { GameState } from "./pathagon";
import type { SearchProgress } from "./rust-search-client";

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

function yieldToWorker() {
  return new Promise<void>((resolve) => setTimeout(resolve, 0));
}

async function runPathfinderSearch(request: SearchRequest) {
  const engine = await enginePromise;
  if (cancelled.delete(request.requestId)) return;

  const transitionPolicy = request.opponentId === TRANSITION_PATHFINDER_ID
    ? await loadDefaultTransitionPolicy()
    : undefined;
  if (cancelled.delete(request.requestId)) return;

  const requestedConfig = request.opponentId === TRAINED_PATHFINDER_ID || transitionPolicy
    ? trainedPathfinderSearchAtDepth(request.pathfinderDepth)
    : pathfinderSearchAtDepth(request.pathfinderDepth);
  const targetDepth = requestedConfig.depth;
  const startedAt = performance.now();
  const deadlineAt = startedAt + Math.max(1, request.deadlineMs);
  const maxNodes = requestedConfig.maxNodes;
  let positions = 0;
  let tableHits = 0;
  let completedDepth = 0;
  let bestAction: SearchProgress["action"] = null;
  let bestScore = 0;
  let exhausted = false;

  for (let depth = 1; depth <= targetDepth; depth += 1) {
    if (cancelled.has(request.requestId)) return;
    const remainingMs = Math.floor(deadlineAt - performance.now());
    const remainingNodes = maxNodes - positions;
    if (remainingMs <= 0 || remainingNodes <= 0) {
      exhausted = true;
      break;
    }

    const config = { ...requestedConfig, depth, maxNodes: remainingNodes };
    const passPositions = positions;
    const passTableHits = tableHits;
    const reportPassProgress = (passNodes: number, passHits: number) => {
      const now = performance.now();
      const progress: SearchProgress = {
        action: bestAction,
        score: bestScore,
        nodes: passPositions + passNodes,
        exhausted: false,
        completedDepth,
        tableHits: passTableHits + passHits,
        elapsedMs: Math.round(now - startedAt),
        targetDepth,
      };
      self.postMessage({ type: "progress", requestId: request.requestId, progress });
    };
    const result = transitionPolicy
      ? transitionPolicy.searchBestActionWithProgress(request.state, config, Math.max(1, remainingMs), reportPassProgress)
      : engine.searchBestTacticalActionWithDeadlineProgress(request.state, config, Math.max(1, remainingMs), reportPassProgress);
    positions += result.nodes;
    tableHits += result.tableHits;

    // An incomplete pass must not displace the last fully completed depth,
    // but its legal fallback is still useful before the first checkpoint.
    if (result.completedDepth >= depth && result.action) {
      bestAction = result.action;
      bestScore = result.score;
      completedDepth = depth;
    } else if (!bestAction && result.action) {
      bestAction = result.action;
      bestScore = result.score;
    }

    exhausted = result.exhausted || positions >= maxNodes || performance.now() >= deadlineAt;
    const progress: SearchProgress = {
      action: bestAction,
      score: bestScore,
      nodes: positions,
      exhausted,
      completedDepth,
      tableHits,
      elapsedMs: Math.round(performance.now() - startedAt),
      targetDepth,
    };
    self.postMessage({ type: "progress", requestId: request.requestId, progress });

    if (exhausted || result.completedDepth < depth) break;
    await yieldToWorker();
  }

  if (cancelled.delete(request.requestId)) return;
  const finalProgress: SearchProgress = {
    action: bestAction,
    score: bestScore,
    nodes: positions,
    exhausted: exhausted || completedDepth < targetDepth,
    completedDepth,
    tableHits,
    elapsedMs: Math.round(performance.now() - startedAt),
    targetDepth,
  };
  self.postMessage({ type: "result", requestId: request.requestId, progress: finalProgress });
}

self.addEventListener("message", (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  if (request.type === "cancel") {
    cancelled.add(request.requestId);
    return;
  }

  void runPathfinderSearch(request).catch((error: unknown) => {
    if (cancelled.delete(request.requestId)) return;
    self.postMessage({
      type: "error",
      requestId: request.requestId,
      message: error instanceof Error ? error.message : String(error),
    });
  });
});
