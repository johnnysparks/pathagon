import type { Action, GameState } from "./pathagon.ts";
import { toRuntimePosition } from "./rust-engine.ts";

type PolicyEvaluation = {
  actions: Action[];
  policyLogits: number[];
  value: number;
};

type QAdvPolicyEvaluation = PolicyEvaluation & { qValues: number[] };

type JepaActionEvaluation = {
  actions: Action[];
  rankLogits: number[];
  actionValues: number[];
};

type ActionEvaluation = {
  action: Action;
  prior: number;
  visits: number;
  value: number;
};

type SearchResult = {
  action: Action | null;
  value: number;
  simulations: number;
  nodes: number;
  evaluations: ActionEvaluation[];
};

type GnnModelHandle = {
  evaluate(position: string): string;
  selectAction(position: string, simulations: number, cpuct: number, maxNodes: number, maxTimeMs: number): string;
};

type QAdvModelHandle = {
  evaluate(position: string): string;
  selectAction(position: string, simulations: number, cpuct: number, qAdvWeight: number, maxNodes: number, maxTimeMs: number): string;
};

type InferenceWasmModule = {
  default(input?: string | URL): Promise<unknown>;
  PathagonGnnModel: new (bytes: Uint8Array) => GnnModelHandle;
  PathagonQAdvModel: new (bytes: Uint8Array) => QAdvModelHandle;
  PathagonJepaModel: new (bytes: Uint8Array) => JepaModelHandle;
};

type JepaModelHandle = {
  evaluate(position: string): string;
};

export type GnnEngine = {
  evaluate(state: GameState): PolicyEvaluation;
  selectAction(state: GameState, config: { simulations: number; cpuct: number; maxNodes?: number; maxTimeMs?: number }): SearchResult;
};

export type QAdvEngine = {
  evaluate(state: GameState): QAdvPolicyEvaluation;
  selectAction(state: GameState, config: { simulations: number; cpuct: number; qAdvWeight?: number; maxNodes?: number; maxTimeMs?: number }): SearchResult;
};

export type JepaEngine = {
  evaluate(state: GameState): JepaActionEvaluation;
};

let modulePromise: Promise<InferenceWasmModule> | null = null;

function loadInferenceModule(): Promise<InferenceWasmModule> {
  modulePromise ??= (async () => {
    const moduleUrl = "/engine-inference/pathagon_engine.js?v=decision-theater-v3";
    const wasmModule = await import(/* @vite-ignore */ moduleUrl) as unknown as InferenceWasmModule;
    await wasmModule.default("/engine-inference/pathagon_engine_bg.wasm?v=decision-theater-v3");
    return wasmModule;
  })();
  return modulePromise;
}

async function modelBytes(modelUrl: string) {
  const response = await fetch(modelUrl);
  if (!response.ok) throw new Error(`learned model request failed (${response.status}): ${modelUrl}`);
  return new Uint8Array(await response.arrayBuffer());
}

export function loadGnnEngine(modelUrl = "/models/pathagon-gnn-policy-value.onnx"): Promise<GnnEngine> {
  return Promise.all([loadInferenceModule(), modelBytes(modelUrl)]).then(([wasmModule, bytes]) => {
    const model = new wasmModule.PathagonGnnModel(bytes);
    return {
      evaluate(state: GameState) {
        return JSON.parse(model.evaluate(JSON.stringify(toRuntimePosition(state)))) as PolicyEvaluation;
      },
      selectAction(state: GameState, config: { simulations: number; cpuct: number; maxNodes?: number; maxTimeMs?: number }) {
        return JSON.parse(model.selectAction(
          JSON.stringify(toRuntimePosition(state)),
          config.simulations,
          config.cpuct,
          config.maxNodes ?? Number.MAX_SAFE_INTEGER,
          config.maxTimeMs ?? 0,
        )) as SearchResult;
      },
    };
  });
}

export function loadQAdvEngine(modelUrl = "/models/pathagon-gnn-qadv.onnx"): Promise<QAdvEngine> {
  return Promise.all([loadInferenceModule(), modelBytes(modelUrl)]).then(([wasmModule, bytes]) => {
    const model = new wasmModule.PathagonQAdvModel(bytes);
    return {
      evaluate(state: GameState) {
        return JSON.parse(model.evaluate(JSON.stringify(toRuntimePosition(state)))) as QAdvPolicyEvaluation;
      },
      selectAction(state: GameState, config: { simulations: number; cpuct: number; qAdvWeight?: number; maxNodes?: number; maxTimeMs?: number }) {
        return JSON.parse(model.selectAction(
          JSON.stringify(toRuntimePosition(state)),
          config.simulations,
          config.cpuct,
          config.qAdvWeight ?? 1,
          config.maxNodes ?? Number.MAX_SAFE_INTEGER,
          config.maxTimeMs ?? 0,
        )) as SearchResult;
      },
    };
  });
}

export function loadJepaEngine(modelUrl = "/models/pathagon-jepa-afterstate.onnx"): Promise<JepaEngine> {
  return Promise.all([loadInferenceModule(), modelBytes(modelUrl)]).then(([wasmModule, bytes]) => {
    const model = new wasmModule.PathagonJepaModel(bytes);
    return {
      evaluate(state: GameState) {
        return JSON.parse(model.evaluate(JSON.stringify(toRuntimePosition(state)))) as JepaActionEvaluation;
      },
    };
  });
}
