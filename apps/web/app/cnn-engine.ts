import type { Action, GameState } from "./pathagon.ts";
import { toRuntimePosition } from "./rust-engine.ts";

export type CnnSearchConfig = {
  simulations: number;
  cpuct: number;
  maxNodes?: number;
  maxTimeMs?: number;
};

export type CnnPolicyEvaluation = {
  actions: Action[];
  policyLogits: number[];
  value: number;
};

export type CnnActionEvaluation = {
  action: Action;
  prior: number;
  visits: number;
  value: number;
};

export type CnnSearchResult = {
  action: Action | null;
  value: number;
  simulations: number;
  nodes: number;
  evaluations: CnnActionEvaluation[];
};

type CnnModelHandle = {
  evaluate(position: string): string;
  selectAction(position: string, simulations: number, cpuct: number, maxNodes: number, maxTimeMs: number): string;
};

type CnnWasmModule = {
  default(input?: string | URL): Promise<unknown>;
  PathagonCnnModel: new (bytes: Uint8Array) => CnnModelHandle;
};

export type CnnEngine = {
  evaluate(state: GameState): CnnPolicyEvaluation;
  selectAction(state: GameState, config: CnnSearchConfig): CnnSearchResult;
};

let cnnEnginePromise: Promise<CnnEngine> | null = null;

export function loadCnnEngine(): Promise<CnnEngine> {
  cnnEnginePromise ??= (async () => {
    const moduleUrl = ["/engine-inference/pathagon_engine", ".js?v=decision-theater-v3"].join("");
    const wasmModule = await import(/* @vite-ignore */ moduleUrl) as unknown as CnnWasmModule;
    await wasmModule.default("/engine-inference/pathagon_engine_bg.wasm?v=decision-theater-v3");
    const response = await fetch("/models/pathagon-cnn.onnx");
    if (!response.ok) throw new Error(`CNN model request failed (${response.status})`);
    const bytes = new Uint8Array(await response.arrayBuffer());
    const model = new wasmModule.PathagonCnnModel(bytes);
    return {
      evaluate(state) {
        return JSON.parse(model.evaluate(JSON.stringify(toRuntimePosition(state)))) as CnnPolicyEvaluation;
      },
      selectAction(state, config) {
        return JSON.parse(model.selectAction(
          JSON.stringify(toRuntimePosition(state)),
          config.simulations,
          config.cpuct,
          config.maxNodes ?? Number.MAX_SAFE_INTEGER,
          config.maxTimeMs ?? 0,
        )) as CnnSearchResult;
      },
    };
  })();
  return cnnEnginePromise;
}
