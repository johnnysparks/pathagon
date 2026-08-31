import { PATHFINDER_TACTICAL_FILTER_ID, TRAINED_PATHFINDER_ID, TRANSITION_PATHFINDER_ID } from "./agent-ids.ts";

/**
 * Shared model metadata for the leaderboard API and the browser lab.
 *
 * `rustEngine` is deliberately narrower than "known to the archive":
 * research-only agents remain valid archive identities for replay/history, but
 * they do not receive a live Elo rank until they clear the promotion boundary.
 */
export type LeagueModelStatus = "default" | "control" | "baseline" | "research";

export type LeagueModelDefinition = {
  id: string;
  name: string;
  nickname?: string;
  family: string;
  role: string;
  mechanics?: string;
  budget: string;
  tone: "green" | "violet" | "gold" | "muted";
  glyph: string;
  status: LeagueModelStatus;
  rustEngine: boolean;
  initialRating: number;
};

export const LEAGUE_MODELS = [
  {
    id: TRANSITION_PATHFINDER_ID,
    name: "The Pathfinder · Transition v4",
    nickname: "Transition v4",
    family: "4-ply iterative · action-transition policy",
    role: "user-facing default",
    mechanics: "A learned transition scorer orders the tactical-safe root; bounded search remains the final authority.",
    budget: "",
    tone: "green",
    glyph: "V4",
    status: "default",
    rustEngine: true,
    initialRating: 1_160,
  },
  {
    id: TRAINED_PATHFINDER_ID,
    name: "The Pathfinder · Trained",
    nickname: "Trained Pathfinder",
    family: "4-ply iterative · trained evaluator",
    role: "rollback control",
    mechanics: "A trained evaluator reweights path, captures, threats, and structure inside tactical-safe search.",
    budget: "",
    tone: "green",
    glyph: "T",
    status: "control",
    rustEngine: true,
    initialRating: 1_160,
  },
  {
    id: PATHFINDER_TACTICAL_FILTER_ID,
    name: "The Pathfinder",
    nickname: "Tactical Pathfinder",
    family: "4-ply iterative · tactical-safe",
    role: "historical control",
    mechanics: "A hand-designed evaluator filters roots that hand the opponent an immediate winning reply.",
    budget: "",
    tone: "green",
    glyph: "P",
    status: "control",
    rustEngine: true,
    initialRating: 1_142,
  },
  {
    id: "surveyor-v0.2.0",
    name: "The Surveyor",
    nickname: "Surveyor",
    family: "2-ply broad-beam search",
    role: "playable baseline",
    mechanics: "A wide 2-ply beam compares more candidate moves, trading depth for broad positional coverage.",
    budget: "",
    tone: "violet",
    glyph: "S",
    status: "baseline",
    rustEngine: true,
    initialRating: 1_085,
  },
  {
    id: "lunatic-v0.1.0",
    name: "Lunatic",
    nickname: "Lunatic",
    family: "1-ply pattern heuristic",
    role: "playable baseline",
    mechanics: "A fast 1-ply pattern heuristic chases path progress, captures, and local structure.",
    budget: "",
    tone: "gold",
    glyph: "L",
    status: "baseline",
    rustEngine: true,
    initialRating: 1_059,
  },
  {
    id: "coin-flip-v0.0.1",
    name: "Coin Flip",
    nickname: "Coin Flip",
    family: "Random legal action",
    role: "random baseline",
    mechanics: "Chooses a legal move at random, giving the league a simple floor for strength comparisons.",
    budget: "",
    tone: "muted",
    glyph: "C",
    status: "baseline",
    rustEngine: true,
    initialRating: 935,
  },
  {
    id: "gnn-warmstart-7x7",
    name: "GNN Learner",
    family: "64 channels · 8 message layers",
    role: "research archive",
    budget: "",
    tone: "green",
    glyph: "G",
    status: "research",
    rustEngine: false,
    initialRating: 957,
  },
  {
    id: "qadv-arbiter-7x7-v0.1.0",
    name: "The Q-Arbiter",
    family: "Transition-aware Q / Advantage action ranking",
    role: "research archive",
    budget: "",
    tone: "violet",
    glyph: "Q",
    status: "research",
    rustEngine: false,
    initialRating: 1_000,
  },
  {
    id: "qadv-arbiter-guided-7x7-v0.2.0",
    name: "The Q-Arbiter · Guided Search",
    family: "QAdv top-k ranking · shallow reply verification",
    role: "research archive",
    budget: "",
    tone: "green",
    glyph: "Q",
    status: "research",
    rustEngine: false,
    initialRating: 1_000,
  },
  {
    id: "gnn-reval30k-7x7",
    name: "Re-evaluated GNN 30k",
    family: "Re-evaluated GNN · 4 PUCT simulations",
    role: "research archive",
    budget: "",
    tone: "green",
    glyph: "G",
    status: "research",
    rustEngine: false,
    initialRating: 1_000,
  },
  {
    id: "cnn-baseline-7x7",
    name: "CNN baseline",
    family: "7×7 residual CNN · 87.4k params",
    role: "research archive",
    budget: "",
    tone: "gold",
    glyph: "C",
    status: "research",
    rustEngine: false,
    initialRating: 950,
  },
  {
    id: "cnn-reval30k-7x7",
    name: "Re-evaluated CNN 30k",
    family: "Re-evaluated CNN · 4 PUCT simulations",
    role: "research archive",
    budget: "",
    tone: "gold",
    glyph: "C",
    status: "research",
    rustEngine: false,
    initialRating: 1_000,
  },
  {
    id: "gnn-scout-7x7",
    name: "GNN Scout",
    family: "Compact message passing · 17.5k params",
    role: "research archive",
    budget: "",
    tone: "violet",
    glyph: "S",
    status: "research",
    rustEngine: false,
    initialRating: 940,
  },
  {
    id: "gnn-scout-puct32-7x7",
    name: "Scout + PUCT",
    family: "GNN Scout policy/value · neural tree search",
    role: "research archive",
    budget: "32 PUCT simulations / move",
    tone: "violet",
    glyph: "P",
    status: "research",
    rustEngine: false,
    initialRating: 940,
  },
  {
    id: "gnn-scout-beam-7x7",
    name: "Scout + Neural Beam",
    family: "GNN Scout policy · 8-move iterative beam",
    role: "research archive",
    budget: "1,000 node budget / move",
    tone: "green",
    glyph: "B",
    status: "research",
    rustEngine: false,
    initialRating: 940,
  },
  {
    id: "gnn-scout-hybrid-beam-7x7",
    name: "Scout + Hybrid Beam",
    family: "Scout policy + Pathfinder value · 8-move beam",
    role: "research archive",
    budget: "1,000 node budget / move",
    tone: "gold",
    glyph: "H",
    status: "research",
    rustEngine: false,
    initialRating: 940,
  },
  {
    id: "pathfinder-deep-10k-7x7",
    name: "Pathfinder + Deep Search",
    family: "Pathfinder heuristic · 4-ply iterative search",
    role: "research archive",
    budget: "10,000 node budget / move",
    tone: "green",
    glyph: "P",
    status: "research",
    rustEngine: false,
    initialRating: 940,
  },
  {
    id: "gnn-scout-beam10k-7x7",
    name: "Scout + 10k Beam",
    family: "GNN Scout policy · 5-ply neural beam",
    role: "research archive",
    budget: "10,000 node budget / move",
    tone: "violet",
    glyph: "B",
    status: "research",
    rustEngine: false,
    initialRating: 940,
  },
] as const satisfies readonly LeagueModelDefinition[];

export const RANKED_LEAGUE_MODELS = LEAGUE_MODELS.filter((model) => model.rustEngine);

export const LATEST_RESEARCH = {
  title: "Transition v4",
  status: "Promoted default",
  researchPath: "20260830-nextgen-scaled",
  artifactId: "pathfinder-action-transition-v4-xent",
  modelHash: "f11d7ddee101ccab35ee162e53c95ced076b1fb10242443ad562dbd51c1085d4",
  trainingViewRoots: 14_000,
  trainingRoots: 11_145,
  heldoutRoots: 2_855,
  heldoutTop1: 895,
  heldoutTop3: 1_369,
  arenaGames: 1_000,
  arenaWins: 565,
  arenaLosses: 401,
  arenaDraws: 34,
  arenaPointRate: 0.582,
  lightPointRate: 0.575,
  darkPointRate: 0.589,
  replayAudit: "1,000 games · 46,604 plies · 10,859 captures · 0 mismatches",
  retainedControls: "v3 packaged prior · v0.5 rollback control · v0.4 tactical-safe control",
} as const;

export const RESEARCH_LANES = [
  {
    label: "v4 · promoted",
    tone: "green" as const,
    detail: "Explicit transition scorer; 1,000-game arena cleared both colors and the replay audit.",
  },
  {
    label: "v3 · prior package",
    tone: "violet" as const,
    detail: "Useful 800-game signal and retained as the previous version, not the active default.",
  },
  {
    label: "v0.5 · rollback control",
    tone: "gold" as const,
    detail: "Supported trained evaluator retained for comparison and rollback coverage.",
  },
  {
    label: "Other research",
    tone: "muted" as const,
    detail: "Board-only, QAdv, sorter, curriculum, and proof-guided lanes remain unpromoted.",
  },
] as const;

export function isRankedLeagueModel(id: string) {
  return LEAGUE_MODELS.some((model) => model.id === id && model.rustEngine);
}

export function leagueModel(id: string) {
  return LEAGUE_MODELS.find((model) => model.id === id);
}
