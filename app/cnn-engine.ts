import type { Action, GameState } from "./pathagon.ts";
import { toRuntimePosition } from "./rust-engine.ts";

export type CnnSearchConfig = {
  simulations: number;
  cpuct: number;
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
  evaluations: CnnActionEvaluation[];
};

type CnnModelHandle = {
  evaluate(position: string): string;
  selectAction(position: string, simulations: number, cpuct: number): string;
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
    const moduleUrl = ["/engine-inference/pathagon_engine", ".js"].join("");
    const wasmModule = await import(/* @vite-ignore */ moduleUrl) as unknown as CnnWasmModule;
    await wasmModule.default();
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
        )) as CnnSearchResult;
      },
    };
  })();
  return cnnEnginePromise;
}
