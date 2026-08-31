import type { SearchConfig } from "./ai";
import type { Action } from "./pathagon";

export type SearchCheckpoint = {
  action: Action;
  completedDepth: number;
  nodes: number;
  maxNodes: number;
  nodeCapReached: boolean;
  elapsedMs: number;
};

export type PathfinderMoveTelemetry = {
  ply: number;
  action: Action;
  searchTimeMs: number;
  positions: number;
  maxNodes: number;
  nodeCapReached: boolean;
  targetDepth: number;
  completedDepth: number;
  tableHits: number;
  exhausted: boolean;
  interrupted: boolean;
  modelCard: {
    id: string;
    name: string;
    version: string;
    engine: string;
  };
  config: SearchConfig & { deadlineMs: number };
  checkpoints: SearchCheckpoint[];
};

type PathfinderModelCard = Pick<PathfinderMoveTelemetry["modelCard"], "id" | "name" | "version" | "engine">;

const ARCHIVE_CHECKPOINTS_PER_MOVE = 3;

/**
 * Keep useful search evidence without persisting the full high-frequency
 * progress stream. A max-depth search can produce thousands of checkpoints
 * for one move, while the human archive only needs representative samples.
 */
export function compactPathfinderGameMetadata(
  opponent: PathfinderModelCard,
  depth: number,
  maxNodes: number,
  deadlineMs: number,
  searches: PathfinderMoveTelemetry[],
) {
  const firstConfig = searches[0]?.config;
  return {
    searchExperiment: "pathfinder-browser-v1",
    modelCard: { ...opponent },
    dials: { depth, maxNodes, deadlineMs },
    ...(firstConfig ? { evaluatorWeights: { ...firstConfig.weights } } : {}),
    moves: searches.map(compactPathfinderMoveTelemetry),
  };
}

function compactPathfinderMoveTelemetry(search: PathfinderMoveTelemetry) {
  return {
    ply: search.ply,
    action: search.action,
    searchTimeMs: search.searchTimeMs,
    positions: search.positions,
    maxNodes: search.maxNodes,
    nodeCapReached: search.nodeCapReached,
    targetDepth: search.targetDepth,
    completedDepth: search.completedDepth,
    tableHits: search.tableHits,
    exhausted: search.exhausted,
    interrupted: search.interrupted,
    search: {
      depth: search.config.depth,
      maxNodes: search.config.maxNodes,
      beamWidth: search.config.beamWidth,
      deadlineMs: search.config.deadlineMs,
    },
    checkpointCount: search.checkpoints.length,
    checkpoints: representativeCheckpoints(search.checkpoints),
  };
}

function representativeCheckpoints(checkpoints: SearchCheckpoint[]) {
  if (checkpoints.length <= ARCHIVE_CHECKPOINTS_PER_MOVE) {
    return checkpoints.map((checkpoint) => ({ ...checkpoint }));
  }

  const indexes = [0, Math.floor((checkpoints.length - 1) / 2), checkpoints.length - 1];
  return indexes.map((index) => ({ ...checkpoints[index] }));
}
