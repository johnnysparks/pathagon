import type { Action, GameState } from "./pathagon";

export type SearchProgress = {
  action: Action | null;
  score: number;
  nodes: number;
  exhausted: boolean;
  completedDepth: number;
  tableHits: number;
  elapsedMs: number;
  targetDepth: number;
};

type WorkerProgress = { type: "progress"; requestId: number; progress: SearchProgress };
type WorkerResult = { type: "result"; requestId: number; progress: SearchProgress };
type WorkerError = { type: "error"; requestId: number; message: string };
type WorkerResponse = WorkerProgress | WorkerResult | WorkerError;

type Pending = {
  resolve: (progress: SearchProgress | null) => void;
  reject: (error: Error) => void;
  onProgress?: (progress: SearchProgress) => void;
};

export type SearchRequestHandle = {
  requestId: number;
  promise: Promise<SearchProgress | null>;
};

export class RustSearchClient {
  private worker: Worker;
  private nextRequestId = 1;
  private readonly pending = new Map<number, Pending>();

  constructor() {
    this.worker = this.createWorker();
  }

  private createWorker() {
    const worker = new Worker(new URL("./rust-search-worker.ts", import.meta.url), { type: "module" });
    worker.addEventListener("message", (event: MessageEvent<WorkerResponse>) => {
      const response = event.data;
      const pending = this.pending.get(response.requestId);
      if (!pending) return;
      if (response.type === "progress") {
        pending.onProgress?.(response.progress);
        return;
      }
      this.pending.delete(response.requestId);
      if (response.type === "error") {
        pending.reject(new Error(response.message));
      } else {
        pending.resolve(response.progress);
      }
    });
    worker.addEventListener("error", (event) => {
      const error = new Error(event.message || "Rust search worker failed");
      for (const pending of this.pending.values()) pending.reject(error);
      this.pending.clear();
    });
    return worker;
  }

  search(
    state: GameState,
    opponentId: string,
    pathfinderDepth: number,
    deadlineMs: number,
    onProgress?: (progress: SearchProgress) => void,
  ): SearchRequestHandle {
    const requestId = this.nextRequestId;
    this.nextRequestId += 1;
    const promise = new Promise<SearchProgress | null>((resolve, reject) => {
      this.pending.set(requestId, { resolve, reject, onProgress });
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
    // The WASM entry points are synchronous. A cancel message cannot be
    // handled until the current call returns, so terminate this worker to
    // interrupt a deep search immediately, then make a fresh worker for the
    // next turn.
    for (const current of this.pending.values()) current.resolve(null);
    this.pending.clear();
    this.worker.terminate();
    this.worker = this.createWorker();
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
