import type { Action, GameState } from "./pathagon";

type WorkerResult = { type: "result"; requestId: number; action: Action | null };
type WorkerError = { type: "error"; requestId: number; message: string };
type WorkerResponse = WorkerResult | WorkerError;

type Pending = {
  resolve: (action: Action | null) => void;
  reject: (error: Error) => void;
};

export type SearchRequestHandle = {
  requestId: number;
  promise: Promise<Action | null>;
};

export class RustSearchClient {
  private readonly worker: Worker;
  private nextRequestId = 1;
  private readonly pending = new Map<number, Pending>();

  constructor() {
    this.worker = new Worker(new URL("./rust-search-worker.ts", import.meta.url), { type: "module" });
    this.worker.addEventListener("message", (event: MessageEvent<WorkerResponse>) => {
      const response = event.data;
      const pending = this.pending.get(response.requestId);
      if (!pending) return;
      this.pending.delete(response.requestId);
      if (response.type === "error") {
        pending.reject(new Error(response.message));
      } else {
        pending.resolve(response.action);
      }
    });
    this.worker.addEventListener("error", (event) => {
      const error = new Error(event.message || "Rust search worker failed");
      for (const pending of this.pending.values()) pending.reject(error);
      this.pending.clear();
    });
  }

  search(
    state: GameState,
    opponentId: string,
    pathfinderDepth: number,
    deadlineMs: number,
  ): SearchRequestHandle {
    const requestId = this.nextRequestId;
    this.nextRequestId += 1;
    const promise = new Promise<Action | null>((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject });
      this.worker.postMessage({
        type: "search",
        requestId,
        state,
        opponentId,
        pathfinderDepth,
        deadlineMs,
      });
    });
    return { requestId, promise };
  }

  cancel(requestId: number) {
    const pending = this.pending.get(requestId);
    if (!pending) return;
    this.pending.delete(requestId);
    pending.resolve(null);
    this.worker.postMessage({ type: "cancel", requestId });
  }

  terminate() {
    for (const pending of this.pending.values()) pending.resolve(null);
    this.pending.clear();
    this.worker.terminate();
  }
}

export function createRustSearchClient() {
  return new RustSearchClient();
}
