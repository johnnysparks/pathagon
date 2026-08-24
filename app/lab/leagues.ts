export type LeagueManifest = {
  id: string;
  label: string;
  boardSize: number;
  reservePerPlayer: number;
  gamesPerMatch: number;
  simulations: number;
  archive: string;
  note: string;
};

export type LeagueSummary = {
  games: number;
  wins: number;
  losses: number;
  draws: number;
  points: number;
};

export type LeagueStanding = LeagueSummary & {
  id: string;
  label: string;
  kind: "gnn" | "heuristic" | "random";
  rating: number;
};

export type LeagueMatch = {
  left: string;
  right: string;
  games: number;
  leftSummary: LeagueSummary;
  rightSummary: LeagueSummary;
};

export type LeagueArchive = {
  boardSize: number;
  reservePerPlayer: number;
  gamesPerMatch: number;
  simulations: number;
  standings: LeagueStanding[];
  headToHead: LeagueMatch[];
};

export const LEAGUES: LeagueManifest[] = [
  {
    id: "league-4x4-r6-transfer",
    label: "4×4 · 6-piece transfer",
    boardSize: 4,
    reservePerPlayer: 6,
    gamesPerMatch: 2,
    simulations: 2,
    archive: "league-4x4-r6-transfer.json",
    note: "A compact solvability diagnostic: 29 of 30 transfer-league games reached a path, with no repetition draws.",
  },
  {
    id: "league-5x5-r8-generation-10",
    label: "5×5 · Gen 10 evaluation",
    boardSize: 5,
    reservePerPlayer: 8,
    gamesPerMatch: 4,
    simulations: 4,
    archive: "league-5x5-r8-generation-10.json",
    note: "Generation 10 candidate evaluation: expanded to seven agents, with four color-balanced games per matchup.",
  },
  {
    id: "league-5x5-r8",
    label: "5×5 · 8-piece reserve",
    boardSize: 5,
    reservePerPlayer: 8,
    gamesPerMatch: 4,
    simulations: 4,
    archive: "league-5x5-r8.json",
    note: "The compact-board promotion pool: every agent met every other agent four times, twice in each color.",
  },
  {
    id: "league-6x6-r10-transfer",
    label: "6×6 · 10-piece transfer",
    boardSize: 6,
    reservePerPlayer: 10,
    gamesPerMatch: 2,
    simulations: 2,
    archive: "league-6x6-r10-transfer.json",
    note: "A medium-board transfer diagnostic: 22 of 30 games reached a path; the rest hit the bounded move cap.",
  },
  {
    id: "league-7x7-r14",
    label: "7×7 · standard reserve",
    boardSize: 7,
    reservePerPlayer: 14,
    gamesPerMatch: 2,
    simulations: 1,
    archive: "league-7x7-r14.json",
    note: "A lower-budget larger-board pool, with one Light and one Dark game per matchup.",
  },
];
