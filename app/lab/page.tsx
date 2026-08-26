"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { applyAction, createGame, type GameState } from "../pathagon";
import type { ContractMove, ContractReplayRecord } from "../contract";

const ALL_CROSS_PLAY_RUN_ID = "all-cross-play";
const GAME_THUMBNAIL_RESOLUTION = 256;

const MODELS = [
  {
    id: "pathfinder-v0.3.0",
    name: "The Pathfinder",
    family: "4-ply iterative search",
    role: "playable opponent",
    budget: "",
    tone: "green",
    glyph: "P",
    planned: false,
    disabled: false,
  },
  {
    id: "surveyor-v0.2.0",
    name: "The Surveyor",
    family: "2-ply broad-beam search",
    role: "playable opponent",
    budget: "",
    tone: "violet",
    glyph: "S",
    planned: false,
    disabled: false,
  },
  {
    id: "lunatic-v0.1.0",
    name: "Lunatic",
    family: "1-ply pattern heuristic",
    role: "playable opponent",
    budget: "",
    tone: "gold",
    glyph: "L",
    planned: false,
    disabled: false,
  },
  {
    id: "coin-flip-v0.0.1",
    name: "Coin Flip",
    family: "Random legal action",
    role: "playable opponent",
    budget: "",
    tone: "muted",
    glyph: "C",
    planned: false,
    disabled: false,
  },
  {
    id: "gnn-warmstart-7x7",
    name: "GNN Learner",
    family: "64 channels · 8 message layers",
    role: "neural candidate",
    budget: "",
    tone: "green",
    glyph: "G",
    planned: false,
    disabled: false,
  },
  {
    id: "qadv-arbiter-7x7-v0.1.0",
    name: "The Q-Arbiter",
    family: "Transition-aware Q / Advantage action ranking",
    role: "learned action selector",
    budget: "",
    tone: "violet",
    glyph: "Q",
    planned: false,
    disabled: false,
  },
  {
    id: "qadv-arbiter-guided-7x7-v0.2.0",
    name: "The Q-Arbiter · Guided Search",
    family: "QAdv top-k ranking · shallow reply verification",
    role: "learned search candidate",
    budget: "",
    tone: "green",
    glyph: "Q",
    planned: false,
    disabled: false,
  },
  {
    id: "gnn-reval30k-7x7",
    name: "Re-evaluated GNN 30k",
    family: "Re-evaluated GNN · 4 PUCT simulations",
    role: "neural reference",
    budget: "",
    tone: "green",
    glyph: "G",
    planned: false,
    disabled: false,
  },
  {
    id: "cnn-baseline-7x7",
    name: "CNN baseline",
    family: "7×7 residual CNN · 87.4k params",
    role: "neural candidate",
    budget: "",
    tone: "gold",
    glyph: "C",
    planned: false,
    disabled: false,
  },
  {
    id: "cnn-reval30k-7x7",
    name: "Re-evaluated CNN 30k",
    family: "Re-evaluated CNN · 4 PUCT simulations",
    role: "neural reference",
    budget: "",
    tone: "gold",
    glyph: "C",
    planned: false,
    disabled: false,
  },
  {
    id: "gnn-scout-7x7",
    name: "GNN Scout",
    family: "Compact message passing · 17.5k params",
    role: "neural data generator",
    budget: "",
    tone: "violet",
    glyph: "S",
    planned: false,
    disabled: false,
  },
  {
    id: "gnn-scout-puct32-7x7",
    name: "Scout + PUCT",
    family: "GNN Scout policy/value · neural tree search",
    role: "search candidate",
    budget: "32 PUCT simulations / move",
    tone: "violet",
    glyph: "P",
    planned: true,
    disabled: false,
  },
  {
    id: "gnn-scout-beam-7x7",
    name: "Scout + Neural Beam",
    family: "GNN Scout policy · 8-move iterative beam",
    role: "search candidate",
    budget: "1,000 node budget / move",
    tone: "green",
    glyph: "B",
    planned: true,
    disabled: false,
  },
  {
    id: "gnn-scout-hybrid-beam-7x7",
    name: "Scout + Hybrid Beam",
    family: "Scout policy + Pathfinder value · 8-move beam",
    role: "hybrid candidate",
    budget: "1,000 node budget / move",
    tone: "gold",
    glyph: "H",
    planned: true,
    disabled: false,
  },
  {
    id: "pathfinder-deep-10k-7x7",
    name: "Pathfinder + Deep Search",
    family: "Pathfinder heuristic · 4-ply iterative search",
    role: "high-budget heuristic",
    budget: "10,000 node budget / move",
    tone: "green",
    glyph: "P",
    planned: true,
    disabled: false,
  },
  {
    id: "gnn-scout-beam10k-7x7",
    name: "Scout + 10k Beam",
    family: "GNN Scout policy · 5-ply neural beam",
    role: "high-budget learned",
    budget: "10,000 node budget / move",
    tone: "violet",
    glyph: "B",
    planned: true,
    disabled: false,
  },
] as const;

type LiveStanding = {
  id: string;
  label: string;
  rating: number;
  games: number;
  wins: number;
  losses: number;
  draws: number;
  points: number;
};

type LiveGame = {
  id: string;
  seed: number;
  light: string;
  dark: string;
  winner: string | null;
  result: "win" | "draw";
  reason: string;
  plies: number;
  finalBoard: GameState["board"];
  winningPath: number[];
};

type ArchivedReplayGame = {
  id: string;
  recordedAt: string;
  engine: string;
  mode: string;
  runId: string | null;
  record: ContractReplayRecord;
};

type HeadToHead = {
  leftId: string;
  rightId: string;
  leftLabel: string;
  rightLabel: string;
  games: number;
  leftWins: number;
  rightWins: number;
  draws: number;
  leftPoints: number;
  rightPoints: number;
  leftLightGames: number;
  rightLightGames: number;
};

type CrossPlayState = {
  runId: string;
  targetGames: number;
  games: number;
  status: "ready" | "running" | "complete";
  standings: LiveStanding[];
  headToHead: HeadToHead[];
  latest: LiveGame[];
};

type StandingFilter = "all" | "rated" | "candidate" | "waiting" | "disabled";
type StandingSort = "rank" | "elo-desc" | "games-desc" | "name-asc";
type HeadToHeadFilter = "all" | "active" | "inactive";
type HeadToHeadSort = "games-desc" | "games-asc" | "win-rate-desc" | "loss-rate-asc" | "draw-rate-desc" | "score-rate-desc" | "pairing-asc";

type HeadToHeadView = {
  focusLabel: string;
  opponentId: string;
  opponentLabel: string;
  games: number;
  wins: number;
  losses: number;
  draws: number;
  points: number;
  focusLightGames: number;
  opponentLightGames: number;
  winRate: number;
  lossRate: number;
  drawRate: number;
  scoreRate: number;
};

export default function LearningLab() {
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [crossPlay, setCrossPlay] = useState<CrossPlayState | null>(null);
  const [crossPlayError, setCrossPlayError] = useState<string | null>(null);
  const [replaySummary, setReplaySummary] = useState<LiveGame | null>(null);
  const [replayGame, setReplayGame] = useState<ArchivedReplayGame | null>(null);
  const [replayPly, setReplayPly] = useState(0);
  const [replayPlaying, setReplayPlaying] = useState(false);
  const [replayLoadingId, setReplayLoadingId] = useState<string | null>(null);
  const [replayError, setReplayError] = useState<string | null>(null);
  const [replayModalOpen, setReplayModalOpen] = useState(false);
  const [standingSearch, setStandingSearch] = useState("");
  const [standingFilter, setStandingFilter] = useState<StandingFilter>("all");
  const [standingSort, setStandingSort] = useState<StandingSort>("rank");
  const [headToHeadSearch, setHeadToHeadSearch] = useState("");
  const [headToHeadFocus, setHeadToHeadFocus] = useState("all");
  const [headToHeadFilter, setHeadToHeadFilter] = useState<HeadToHeadFilter>("all");
  const [headToHeadSort, setHeadToHeadSort] = useState<HeadToHeadSort>("games-desc");
  const replayRequest = useRef(0);

  const liveStandingById = useMemo(
    () => new Map((crossPlay?.standings ?? []).map((standing) => [standing.id, standing])),
    [crossPlay],
  );
  const liveRankById = useMemo(
    () => new Map((crossPlay?.standings ?? []).map((standing, index) => [standing.id, String(index + 1).padStart(2, "0")])),
    [crossPlay],
  );
  const visibleModels = useMemo(() => {
    const query = standingSearch.trim().toLowerCase();
    const filtered = MODELS.filter((model) => {
      const live = liveStandingById.get(model.id);
      const status = modelStandingStatus(model, live);
      const searchable = `${model.name} ${model.family} ${model.role} ${model.id}`.toLowerCase();
      return (!query || searchable.includes(query)) && (standingFilter === "all" || status === standingFilter);
    });

    return [...filtered].sort((left, right) => {
      const leftLive = liveStandingById.get(left.id);
      const rightLive = liveStandingById.get(right.id);
      if (standingSort === "elo-desc") return compareNullableNumbers(leftLive?.rating, rightLive?.rating, left, right);
      if (standingSort === "games-desc") return compareNullableNumbers(leftLive?.games, rightLive?.games, left, right);
      if (standingSort === "name-asc") return left.name.localeCompare(right.name) || MODELS.indexOf(left) - MODELS.indexOf(right);

      const leftRank = liveRankById.get(left.id);
      const rightRank = liveRankById.get(right.id);
      if (leftRank && rightRank) return Number(leftRank) - Number(rightRank);
      if (leftRank) return -1;
      if (rightRank) return 1;
      return MODELS.indexOf(left) - MODELS.indexOf(right);
    });
  }, [liveRankById, liveStandingById, standingFilter, standingSearch, standingSort]);

  const scopedHeadToHead = useMemo(() => {
    if (!crossPlay) return [];
    if (headToHeadFocus === "all") return crossPlay.headToHead;
    return crossPlay.headToHead.filter((pairing) => pairing.leftId === headToHeadFocus || pairing.rightId === headToHeadFocus);
  }, [crossPlay, headToHeadFocus]);

  const visibleHeadToHead = useMemo(() => {
    const query = headToHeadSearch.trim().toLowerCase();
    const filtered = scopedHeadToHead
      .filter((pairing) => {
        const active = pairing.games > 0;
        const searchable = `${pairing.leftLabel} ${pairing.rightLabel} ${pairing.leftId} ${pairing.rightId}`.toLowerCase();
        return (!query || searchable.includes(query))
          && (headToHeadFilter === "all" || (headToHeadFilter === "active" ? active : !active));
      })
      .map((pairing) => orientHeadToHead(pairing, headToHeadFocus));

    return [...filtered].sort((left, right) => {
      if (headToHeadSort === "games-asc") return left.games - right.games || left.opponentLabel.localeCompare(right.opponentLabel);
      if (headToHeadSort === "win-rate-desc") return compareHeadToHeadMetric(left.winRate, right.winRate, left.games, right.games, left.opponentLabel, right.opponentLabel, "desc");
      if (headToHeadSort === "loss-rate-asc") return compareHeadToHeadMetric(left.lossRate, right.lossRate, left.games, right.games, left.opponentLabel, right.opponentLabel, "asc");
      if (headToHeadSort === "draw-rate-desc") return compareHeadToHeadMetric(left.drawRate, right.drawRate, left.games, right.games, left.opponentLabel, right.opponentLabel, "desc");
      if (headToHeadSort === "score-rate-desc") return compareHeadToHeadMetric(left.scoreRate, right.scoreRate, left.games, right.games, left.opponentLabel, right.opponentLabel, "desc");
      if (headToHeadSort === "pairing-asc") return `${left.focusLabel} ${left.opponentLabel}`.localeCompare(`${right.focusLabel} ${right.opponentLabel}`);
      return right.games - left.games || left.opponentLabel.localeCompare(right.opponentLabel);
    });
  }, [headToHeadFilter, headToHeadFocus, headToHeadSearch, headToHeadSort, scopedHeadToHead]);

  const headToHeadFocusLabel = MODELS.find((model) => model.id === headToHeadFocus)?.name;

  const strengthLeaderLive = crossPlay?.standings[0];
  const strengthLeader = MODELS.find((model) => model.id === strengthLeaderLive?.id) ?? MODELS[0];
  const ratedAgentCount = crossPlay?.standings.filter((standing) => standing.games > 0).length ?? MODELS.filter((model) => !model.planned).length;
  const queuedCandidateCount = MODELS.filter((model) => model.planned && !liveStandingById.get(model.id)?.games).length;

  useEffect(() => {
    const savedTheme = window.localStorage.getItem("pathagon-lab-theme");
    const preferredTheme = savedTheme === "dark" || savedTheme === "light"
      ? savedTheme
      : window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    const timer = window.setTimeout(() => setTheme(preferredTheme), 0);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (!replayModalOpen || !replayPlaying || !replayGame) return;
    if (replayPly >= replayGame.record.moves.length) {
      const finishTimer = window.setTimeout(() => setReplayPlaying(false), 0);
      return () => window.clearTimeout(finishTimer);
    }
    const timer = window.setTimeout(() => setReplayPly((current) => current + 1), 650);
    return () => window.clearTimeout(timer);
  }, [replayGame, replayModalOpen, replayPlaying, replayPly]);

  useEffect(() => {
    if (!replayModalOpen) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        replayRequest.current += 1;
        setReplayPlaying(false);
        setReplayModalOpen(false);
        return;
      }
      if (!replayGame) return;
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        setReplayPlaying(false);
        setReplayPly((current) => Math.max(0, current - 1));
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        setReplayPly((current) => Math.min(replayGame.record.moves.length, current + 1));
      } else if (event.key === " ") {
        event.preventDefault();
        setReplayPlaying((current) => !current);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [replayGame, replayModalOpen]);

  useEffect(() => {
    if (!replayModalOpen) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = previousOverflow; };
  }, [replayModalOpen]);

  useEffect(() => {
    document.body.dataset.labTheme = theme;
    return () => { delete document.body.dataset.labTheme; };
  }, [theme]);

  useEffect(() => {
    let active = true;
    const refresh = () => {
      void readLatestCrossPlay()
        .then((snapshot) => {
          if (!active) return;
          setCrossPlay(snapshot);
          setCrossPlayError(null);
        })
        .catch((error: unknown) => {
          if (active) setCrossPlayError(error instanceof Error ? error.message : "Imported archive unavailable");
        });
    };
    refresh();
    const timer = window.setInterval(refresh, 900);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  function toggleTheme() {
    const nextTheme = theme === "dark" ? "light" : "dark";
    setTheme(nextTheme);
    window.localStorage.setItem("pathagon-lab-theme", nextTheme);
  }

  async function openReplay(game: LiveGame) {
    const requestId = replayRequest.current + 1;
    replayRequest.current = requestId;
    setReplaySummary(game);
    setReplayGame(null);
    setReplayPly(0);
    setReplayPlaying(false);
    setReplayLoadingId(game.id);
    setReplayError(null);
    setReplayModalOpen(true);
    try {
      const response = await fetch(`/api/selfplay/${encodeURIComponent(game.id)}`, { cache: "no-store" });
      const payload = await response.json() as { found?: boolean; game?: ArchivedReplayGame; error?: string };
      if (requestId !== replayRequest.current) return;
      if (!response.ok || !payload.game) throw new Error(payload.error ?? "Replay unavailable");
      setReplayGame(payload.game);
    } catch (error: unknown) {
      if (requestId === replayRequest.current) setReplayError(error instanceof Error ? error.message : "Replay unavailable");
    } finally {
      if (requestId === replayRequest.current) setReplayLoadingId(null);
    }
  }

  function closeReplay() {
    replayRequest.current += 1;
    setReplayPlaying(false);
    setReplayModalOpen(false);
  }

  function changeReplayPly(nextPly: number) {
    if (!replayGame) return;
    setReplayPlaying(false);
    setReplayPly(Math.max(0, Math.min(replayGame.record.moves.length, nextPly)));
  }

  return (
    <main className={`portal-app leaderboard-app ${theme === "dark" ? "dark" : ""}`}>
      <nav className="portal-nav" aria-label="Leaderboard navigation">
        <Link className="portal-breadcrumb" href="/">
          <span className="portal-mark">P</span>
          <span>Pathagon</span>
          <span className="portal-slash">/</span>
          <span>Leaderboard</span>
        </Link>
        <div className="portal-nav-right">
          <span className="portal-live"><span /> 7×7 league</span>
          <button className="portal-theme-toggle" type="button" onClick={toggleTheme} aria-pressed={theme === "dark"} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}>
            <span aria-hidden="true">{theme === "dark" ? "☼" : "☾"}</span>
            {theme === "dark" ? "Light" : "Dark"}
          </button>
          <Link className="portal-game-link" href="/">Back to game <span>↗</span></Link>
        </div>
      </nav>

      <header className="leaderboard-hero">
        <div className="leaderboard-hero-copy">
          <span className="portal-kicker">7×7 model league</span>
          <h1>Leaderboard</h1>
          <p>Model rankings backed by every imported and offline cross-play result.</p>
        </div>

        <div className="leaderboard-leader-card" aria-label="Current strength leader">
          <div className="leaderboard-card-topline"><span>Current strength leader</span><span className="leaderboard-provisional">{strengthLeaderLive ? "Live Elo" : "Waiting for poll"}</span></div>
          <div className="leaderboard-leader-main">
            <ModelGlyph tone={strengthLeader.tone} glyph={strengthLeader.glyph} />
            <div className="leaderboard-leader-name"><strong>{strengthLeaderLive?.label ?? strengthLeader.name}</strong><span>{strengthLeader.family}</span></div>
            <div className="leaderboard-leader-score"><strong>{strengthLeaderLive?.rating.toLocaleString() ?? "—"}</strong><small>{strengthLeaderLive ? "cumulative Elo" : "live data pending"}</small></div>
          </div>
          <div className="leaderboard-signal-row"><span>{strengthLeaderLive ? `${formatRecord(strengthLeaderLive)} cumulative` : "Live standings are loading"}</span><span>higher is better</span></div>
          <div className="leaderboard-card-footer"><span><i className="live-dot" /> {crossPlay ? `${crossPlay.games} cumulative games` : "Polling cumulative archive"}</span><small>refreshes every 0.9s</small></div>
        </div>
      </header>

      <section className="leaderboard-stat-grid" aria-label="Model league summary">
        <LeaderboardStat label="Agents tracked" value={String(MODELS.length)} detail={`${ratedAgentCount} rated · ${queuedCandidateCount} queued candidates`} accent="green" />
        <LeaderboardStat label="7×7 benchmark" value="3,251" detail="2,037 unique · 416 held out" accent="gold" />
        <LeaderboardStat label="Imported cross-play" value={crossPlay ? String(crossPlay.games) : "—"} detail={crossPlay ? "cumulative archive records" : "waiting for first poll"} accent="gold" />
        <LeaderboardStat label="Held-out policy NLL" value="2.112" detail="GNN · 416 held-out records" accent="ink" />
      </section>

      <section className="leaderboard-panel cross-play-live-panel" aria-labelledby="live-run-title">
        <div className="cross-play-live-heading">
          <div><span className="portal-kicker">Imported archive · read-only</span><h2 id="live-run-title">Cumulative cross-play archive</h2><p>The browser polls the database every 0.9 seconds. Imported offline results from chat or the terminal appear here automatically.</p></div>
          <span className="leaderboard-status polling-status"><span /> {crossPlay ? "Polling" : "Connecting"}</span>
        </div>
        <div className="live-run-summary">
          <div><strong>{crossPlay?.games ?? "—"}<small>{crossPlay ? ` / ${crossPlay.targetGames}` : ""}</small></strong><span>games counted</span></div>
          <div><strong>{crossPlay ? "Archive polling" : "Connecting"}</strong><span>browser status</span></div>
          <div><strong>{crossPlay?.latest[0]?.winner ?? (crossPlay?.latest[0] ? "draw" : "—")}</strong><span>latest stored result</span></div>
        </div>
        {crossPlayError ? <p className="live-run-error" role="status">{crossPlayError} · retrying automatically</p> : null}
        {crossPlay?.latest.length ? <div className="live-game-list" aria-label="Latest cross-play games">{crossPlay.latest.map((game) => <button className="live-game-row" type="button" key={game.id} onClick={() => void openReplay(game)} aria-label={`Replay ${game.light} versus ${game.dark}`}><GameThumbnail board={game.finalBoard} winningPath={game.winningPath} /><span className="live-game-number">{shortGameId(game.id)}</span><strong className="live-game-light">{game.light}</strong><span className="live-game-versus">vs</span><strong className="live-game-dark">{game.dark}</strong><span className="live-game-result">{game.winner ? `${game.winner} · ${game.plies} plies` : `draw · ${game.plies} plies`} <em>Replay ↗</em></span></button>)}</div> : <p className="live-run-empty">Waiting for an imported offline cross-play result.</p>}
      </section>

      <section className="leaderboard-panel" id="standings" aria-labelledby="standings-title">
        <div className="leaderboard-panel-heading">
          <div><span className="portal-kicker">Standings · imported archive</span><h2 id="standings-title">Every model in the ladder.</h2></div>
          <span className="leaderboard-status"><span /> {crossPlay ? `${ratedAgentCount} rated · archive polling` : "Polling archive"}</span>
        </div>
        <p className="leaderboard-intro">Rankings and records are cumulative across all imported and offline 7×7 cross-play games. Search candidates are shown with their planned budgets until they have games; benchmark metrics remain separate from ladder evidence.</p>

        <div className="table-toolbar" aria-label="Standings table controls">
          <label className="table-control table-search" htmlFor="standings-search"><span>Search models</span><input id="standings-search" type="search" value={standingSearch} onChange={(event) => setStandingSearch(event.target.value)} placeholder="Name, family, or role" /></label>
          <label className="table-control" htmlFor="standings-filter"><span>Status</span><select id="standings-filter" value={standingFilter} onChange={(event) => setStandingFilter(event.target.value as StandingFilter)}><option value="all">All models</option><option value="rated">Rated</option><option value="candidate">Candidates</option><option value="waiting">Waiting</option><option value="disabled">Disabled</option></select></label>
          <label className="table-control" htmlFor="standings-sort"><span>Sort by</span><select id="standings-sort" value={standingSort} onChange={(event) => setStandingSort(event.target.value as StandingSort)}><option value="rank">Leaderboard rank</option><option value="elo-desc">Elo · high to low</option><option value="games-desc">Games · high to low</option><option value="name-asc">Name · A to Z</option></select></label>
          {standingSearch || standingFilter !== "all" ? <button className="table-clear-button" type="button" onClick={() => { setStandingSearch(""); setStandingFilter("all"); }}>Clear</button> : null}
          <span className="table-result-count" aria-live="polite">Showing {visibleModels.length} of {MODELS.length} models</span>
        </div>

        <div className="leaderboard-table" role="table" aria-label="Current model standings">
          <div className="leaderboard-table-row leaderboard-table-header" role="row">
            <span>#</span><span>Agent</span><span>Role</span><span>Elo</span><span>Record</span><span>Signal</span>
          </div>
          {visibleModels.length ? visibleModels.map((model) => <ModelStanding key={model.id} model={model} live={liveStandingById.get(model.id)} liveRank={liveRankById.get(model.id)} snapshotLoaded={Boolean(crossPlay)} />) : <div className="table-empty-state" role="row"><strong>No models match those filters.</strong><span>Try a different search or clear the status filter.</span></div>}
        </div>
      </section>

      <section className="leaderboard-panel head-to-head-panel" id="head-to-head" aria-labelledby="head-to-head-title">
        <div className="leaderboard-panel-heading">
          <div><span className="portal-kicker">Head-to-head · imported archive</span><h2 id="head-to-head-title">Pairwise results.</h2></div>
          <span className="leaderboard-status"><span /> {crossPlay ? `${crossPlay.headToHead.filter((pairing) => pairing.games > 0).length} active pairings` : "Waiting for poll"}</span>
        </div>
        <p className="leaderboard-intro">{headToHeadFocusLabel ? `${headToHeadFocusLabel} is the focus model in every row. Its W–L–D, rates, and score are ranked against each opponent.` : "Choose a focus model to see every matchup from that model's perspective. With no focus selected, each row uses the left model's perspective."} Human games stay in their separate archive.</p>

        <div className="table-toolbar" aria-label="Head-to-head table controls">
          <label className="table-control table-search" htmlFor="head-to-head-search"><span>Search models</span><input id="head-to-head-search" type="search" value={headToHeadSearch} onChange={(event) => setHeadToHeadSearch(event.target.value)} placeholder="Name or model ID" /></label>
          <label className="table-control" htmlFor="head-to-head-focus"><span>Focus model</span><select id="head-to-head-focus" value={headToHeadFocus} onChange={(event) => setHeadToHeadFocus(event.target.value)}><option value="all">All models</option>{MODELS.map((model) => <option value={model.id} key={model.id}>{model.name}</option>)}</select></label>
          <label className="table-control" htmlFor="head-to-head-filter"><span>Games</span><select id="head-to-head-filter" value={headToHeadFilter} onChange={(event) => setHeadToHeadFilter(event.target.value as HeadToHeadFilter)}><option value="all">All matchups</option><option value="active">Played only</option><option value="inactive">No games</option></select></label>
          <label className="table-control" htmlFor="head-to-head-sort"><span>Rank by</span><select id="head-to-head-sort" value={headToHeadSort} onChange={(event) => setHeadToHeadSort(event.target.value as HeadToHeadSort)}><option value="games-desc">Play count · high to low</option><option value="games-asc">Play count · low to high</option><option value="win-rate-desc">Win rate · high to low</option><option value="loss-rate-asc">Loss rate · low to high</option><option value="draw-rate-desc">Draw rate · high to low</option><option value="score-rate-desc">Score rate · high to low</option><option value="pairing-asc">Opponent · A to Z</option></select></label>
          {headToHeadSearch || headToHeadFocus !== "all" || headToHeadFilter !== "all" ? <button className="table-clear-button" type="button" onClick={() => { setHeadToHeadSearch(""); setHeadToHeadFocus("all"); setHeadToHeadFilter("all"); }}>Clear</button> : null}
          <span className="table-result-count" aria-live="polite">Showing {visibleHeadToHead.length} of {scopedHeadToHead.length} matchups</span>
        </div>

        {crossPlay?.headToHead.length ? visibleHeadToHead.length ? <div className="head-to-head-table" role="table" aria-label="Head-to-head model results">
          <div className="head-to-head-row head-to-head-header" role="row"><span>Matchup</span><span>Games</span><span>W–L–D</span><span>Rates · W / L / D</span><span>Score rate</span><span>Light starts</span></div>
          {visibleHeadToHead.map((pairing) => <HeadToHeadRow key={`${pairing.focusLabel}-${pairing.opponentId}`} pairing={pairing} />)}
        </div> : <p className="table-empty-state" role="status"><strong>No pairings match those filters.</strong><span>Try a different model name or clear the status filter.</span></p> : <p className="live-run-empty">Waiting for imported pairwise results.</p>}
      </section>

      <footer className="portal-footer"><span>7×7 model leaderboard</span><span>Read-only view · polling the imported archive</span></footer>
      {replayModalOpen ? <ReplayModal summary={replaySummary} game={replayGame} loading={Boolean(replayLoadingId)} error={replayError} ply={replayPly} playing={replayPlaying} onClose={closeReplay} onPlayPause={() => setReplayPlaying((current) => !current)} onPlyChange={changeReplayPly} /> : null}
    </main>
  );
}

function ModelGlyph({ tone, glyph }: { tone: string; glyph: string }) {
  return <span className={`model-glyph ${tone}`} aria-hidden="true">{glyph}</span>;
}

function LeaderboardStat({ label, value, detail, accent }: { label: string; value: string; detail: string; accent: string }) {
  return <div className={`leaderboard-stat ${accent}`}><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}

function HeadToHeadRow({ pairing }: { pairing: HeadToHeadView }) {
  const active = pairing.games > 0;
  return <div className={`head-to-head-row ${active ? "" : "disabled"}`} role="row"><div className="head-to-head-match"><strong>{pairing.focusLabel}</strong><span>vs</span><strong>{pairing.opponentLabel}</strong></div><span className="head-to-head-games">{active ? pairing.games : "—"}</span><span className="head-to-head-record">{active ? `${pairing.wins}–${pairing.losses}–${pairing.draws}` : "no games"}</span><div className="head-to-head-rates" aria-label={active ? `Win ${formatRate(pairing.winRate)}, loss ${formatRate(pairing.lossRate)}, draw ${formatRate(pairing.drawRate)}` : "No games"}><span><b>W</b>{active ? formatRate(pairing.winRate) : "—"}</span><span><b>L</b>{active ? formatRate(pairing.lossRate) : "—"}</span><span><b>D</b>{active ? formatRate(pairing.drawRate) : "—"}</span></div><span className="head-to-head-score">{active ? formatRate(pairing.scoreRate) : "—"}</span><span className="head-to-head-colors">{active ? `${pairing.focusLightGames}–${pairing.opponentLightGames}` : "disabled"}</span></div>;
}

function orientHeadToHead(pairing: HeadToHead, focusId: string): HeadToHeadView {
  const focusIsRight = focusId !== "all" && pairing.rightId === focusId;
  const games = pairing.games;
  const wins = focusIsRight ? pairing.rightWins : pairing.leftWins;
  const losses = focusIsRight ? pairing.leftWins : pairing.rightWins;
  const draws = pairing.draws;
  const points = focusIsRight ? pairing.rightPoints : pairing.leftPoints;
  return {
    focusLabel: focusIsRight ? pairing.rightLabel : pairing.leftLabel,
    opponentId: focusIsRight ? pairing.leftId : pairing.rightId,
    opponentLabel: focusIsRight ? pairing.leftLabel : pairing.rightLabel,
    games,
    wins,
    losses,
    draws,
    points,
    focusLightGames: focusIsRight ? pairing.rightLightGames : pairing.leftLightGames,
    opponentLightGames: focusIsRight ? pairing.leftLightGames : pairing.rightLightGames,
    winRate: games ? wins / games : 0,
    lossRate: games ? losses / games : 0,
    drawRate: games ? draws / games : 0,
    scoreRate: games ? points / games : 0,
  };
}

function compareHeadToHeadMetric(left: number, right: number, leftGames: number, rightGames: number, leftLabel: string, rightLabel: string, direction: "asc" | "desc") {
  return (direction === "asc" ? left - right : right - left) || rightGames - leftGames || leftLabel.localeCompare(rightLabel);
}

function formatRate(rate: number) {
  return `${(rate * 100).toFixed(1)}%`;
}

function modelStandingStatus(model: (typeof MODELS)[number], live?: LiveStanding): Exclude<StandingFilter, "all"> {
  if (live?.games) return "rated";
  if (model.disabled) return "disabled";
  if (model.planned) return "candidate";
  return "waiting";
}

function compareNullableNumbers<T extends (typeof MODELS)[number]>(left: number | undefined, right: number | undefined, leftModel: T, rightModel: T) {
  if (left === undefined && right === undefined) return MODELS.indexOf(leftModel) - MODELS.indexOf(rightModel);
  if (left === undefined) return 1;
  if (right === undefined) return -1;
  return right - left || MODELS.indexOf(leftModel) - MODELS.indexOf(rightModel);
}

function ModelStanding({ model, live, liveRank, snapshotLoaded }: { model: (typeof MODELS)[number]; live?: LiveStanding; liveRank?: string; snapshotLoaded: boolean }) {
  const liveActive = Boolean(live?.games);
  const planned = Boolean(model.planned);
  const disabled = Boolean(model.disabled);
  const waiting = !liveActive && !disabled && !planned;
  const rank = liveActive ? liveRank ?? "—" : "—";
  const record = liveActive ? formatRecord(live!) : disabled || planned ? "not rated" : "—";
  const signal = liveActive ? `${live!.games} games` : disabled ? "offline only" : planned ? model.budget ?? "candidate spec" : snapshotLoaded ? "no live games" : "waiting for poll";
  const signalDetail = liveActive ? `${live!.points.toFixed(1)} points · cumulative` : disabled ? "not rated" : planned ? "planned · awaiting evaluation" : "no ladder evidence";
  const stateClass = disabled ? "disabled" : liveActive ? "" : planned ? "candidate" : waiting ? "disabled" : "";
  return <div className={`leaderboard-table-row model-standing ${rank === "01" ? "leader" : ""} ${stateClass}`} role="row"><span className="model-rank">{rank}</span><div className="standing-model"><ModelGlyph tone={model.tone} glyph={model.glyph} /><div><strong>{model.name}</strong><span>{model.family}</span></div></div><div className="standing-role"><strong>{disabled ? "Disabled" : liveActive ? "Live ladder" : planned ? "Candidate" : "Waiting"}</strong><span>{model.role}</span></div><span className="standing-elo">{liveActive ? live!.rating.toLocaleString() : "—"}</span><div className="standing-record"><strong>{record}</strong><span>{liveActive ? "cumulative" : disabled || planned ? "not rated" : "no games"}</span></div><div className="standing-signal"><strong>{signal}</strong><span>{signalDetail}</span></div></div>;
}

function formatRecord(standing: Pick<LiveStanding, "wins" | "losses" | "draws">) {
  return `${standing.wins}–${standing.losses}–${standing.draws}`;
}

function shortGameId(id: string) {
  const suffix = id.split("-").pop();
  return suffix && /^\d+$/.test(suffix) ? `#${suffix}` : `…${id.slice(-8)}`;
}

function ReplayModal({ summary, game, loading, error, ply, playing, onClose, onPlayPause, onPlyChange }: {
  summary: LiveGame | null;
  game: ArchivedReplayGame | null;
  loading: boolean;
  error: string | null;
  ply: number;
  playing: boolean;
  onClose: () => void;
  onPlayPause: () => void;
  onPlyChange: (ply: number) => void;
}) {
  return <div className="replay-modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <section className="replay-modal" role="dialog" aria-modal="true" aria-labelledby="replay-modal-title">
      <header className="replay-modal-header">
        <div><span className="portal-kicker">Imported archive · read-only</span><h2 id="replay-modal-title">Game playback</h2><p>{summary ? `${summary.light} vs ${summary.dark}` : "Loading archived game"}</p></div>
        <button className="replay-modal-close" type="button" onClick={onClose} aria-label="Close game playback">×</button>
      </header>
      {loading ? <div className="replay-modal-state"><strong>Loading the archived moves…</strong><span>Fetching the validated replay from the lab archive.</span></div> : error ? <div className="replay-modal-state error" role="alert"><strong>Couldn&apos;t load this replay.</strong><span>{error}</span></div> : game ? <ReplayViewer game={game} summary={summary} ply={ply} playing={playing} onPlayPause={onPlayPause} onPlyChange={onPlyChange} /> : <div className="replay-modal-state"><strong>Replay not selected.</strong></div>}
    </section>
  </div>;
}

function ReplayViewer({ game, summary, ply, playing, onPlayPause, onPlyChange }: {
  game: ArchivedReplayGame;
  summary: LiveGame | null;
  ply: number;
  playing: boolean;
  onPlayPause: () => void;
  onPlyChange: (ply: number) => void;
}) {
  const moves = game.record.moves;
  const state = useMemo(() => {
    let current = createGame(game.record.config);
    for (const move of moves.slice(0, ply)) current = applyAction(current, move.action);
    return current;
  }, [game.record.config, moves, ply]);
  const move = ply > 0 ? moves[ply - 1] : null;
  const winnerLabel = summary?.winner ?? (game.record.winner ? playerName(game.record.winner) : null);
  const status = state.winner ? `${winnerLabel ?? playerName(state.winner)} wins · ${game.record.reason}` : ply === 0 ? "Starting position" : `${playerName(state.turn)} to move`;

  return <div className="replay-viewer">
    <div className="replay-viewer-topline"><span>{game.record.engine.runtime} engine · seed {game.record.seed}</span><strong>{ply} / {moves.length} plies</strong></div>
    <div className="replay-board-frame">
      <div className="replay-board" role="grid" aria-label={`Pathagon board at ply ${ply}`}>
        {state.board.map((piece, index) => {
          const forbidden = state.forbidden.includes(index);
          const winning = state.winningPath.includes(index);
          const last = state.lastAction?.to === index;
          return <div className={`replay-cell ${last ? "last" : ""} ${winning ? "winning" : ""}`} key={index} role="gridcell" aria-label={replayCellLabel(index, piece, forbidden, winning, state.config.boardSize)}><span className="replay-socket" />{piece ? <span className={`replay-piece ${piece}`} /> : null}{forbidden ? <span className="replay-forbidden" aria-hidden="true">×</span> : null}</div>;
        })}
      </div>
      {ply === moves.length ? <FinalStateBadge state={state} winnerLabel={winnerLabel} /> : null}
    </div>
    <div className="replay-status-row"><span className={`replay-turn-dot ${state.turn}`} /><div><strong>{move ? describeReplayMove(move, state.config.boardSize) : status}</strong><small>{move ? status : `${game.record.config.boardSize}×${game.record.config.boardSize} board · ${game.record.agents.light} vs ${game.record.agents.dark}`}</small></div></div>
    <div className="replay-controls" aria-label="Replay controls">
      <button className="replay-control" type="button" onClick={() => onPlyChange(0)} disabled={ply === 0} aria-label="Go to start">|◀</button>
      <button className="replay-control" type="button" onClick={() => onPlyChange(ply - 1)} disabled={ply === 0} aria-label="Previous move">←</button>
      <button className="replay-control replay-play" type="button" onClick={onPlayPause} aria-label={playing ? "Pause replay" : "Play replay"}>{playing ? "Pause" : "Play"}</button>
      <button className="replay-control" type="button" onClick={() => onPlyChange(ply + 1)} disabled={ply === moves.length} aria-label="Next move">→</button>
      <button className="replay-control" type="button" onClick={() => onPlyChange(moves.length)} disabled={ply === moves.length} aria-label="Go to final position">▶|</button>
    </div>
    <input className="replay-scrubber" type="range" min="0" max={moves.length} value={ply} onChange={(event) => onPlyChange(Number(event.target.value))} aria-label="Replay position" />
    <div className="replay-scrubber-labels"><span>Start</span><span>{moves.length ? `Final · ${winnerLabel ?? "draw"}` : "No moves"}</span></div>
    <p className="replay-key-hint">Space to play/pause · ← → to step · Esc to close</p>
  </div>;
}

function FinalStateBadge({ state, winnerLabel }: { state: GameState; winnerLabel: string | null }) {
  return <div className="final-state-badge" aria-label={`Final board state${winnerLabel ? ` · ${winnerLabel}` : " · draw"}`}>
    <span className="final-state-badge-label">Final</span>
    <FinalStatePixels board={state.board} winningPath={state.winningPath} />
    <span className="final-state-badge-result">{winnerLabel ?? "Draw"}</span>
  </div>;
}

function GameThumbnail({ board, winningPath }: { board: GameState["board"]; winningPath: number[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;

    drawGameThumbnail(context, board, winningPath);
  }, [board, winningPath]);

  return <span className="live-game-thumbnail" aria-hidden="true"><canvas ref={canvasRef} width={GAME_THUMBNAIL_RESOLUTION} height={GAME_THUMBNAIL_RESOLUTION} /></span>;
}

function drawGameThumbnail(context: CanvasRenderingContext2D, board: GameState["board"], winningPath: number[]) {
  const boardSize = Math.sqrt(board.length);
  if (!Number.isInteger(boardSize)) return;

  const resolution = GAME_THUMBNAIL_RESOLUTION;
  const padding = 8;
  const gap = 4;
  const cellSize = (resolution - padding * 2 - gap * (boardSize - 1)) / boardSize;
  const winning = new Set(winningPath);

  context.clearRect(0, 0, resolution, resolution);
  context.imageSmoothingEnabled = false;
  context.fillStyle = "rgba(147,183,143,.12)";
  context.fillRect(0, 0, resolution, resolution);

  board.forEach((piece, index) => {
    const column = index % boardSize;
    const row = Math.floor(index / boardSize);
    const x = padding + column * (cellSize + gap);
    const y = padding + row * (cellSize + gap);
    const isWinning = winning.has(index);

    context.fillStyle = isWinning ? "#d9c66f" : piece === "light" ? "#e9dfc4" : piece === "dark" ? "#49392f" : "rgba(211,230,204,.08)";
    context.fillRect(x, y, cellSize, cellSize);
    context.strokeStyle = isWinning ? "#fff1a4" : piece === "light" ? "rgba(245,237,209,.76)" : piece === "dark" ? "rgba(25,20,17,.72)" : "rgba(211,230,204,.17)";
    context.lineWidth = 2;
    context.strokeRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
  });
}

function FinalStatePixels({ board, winningPath }: { board: GameState["board"]; winningPath: number[] }) {
  return <span className="final-state-pixels">{board.map((piece, index) => <span className={`final-state-pixel ${piece ?? "empty"} ${winningPath.includes(index) ? "winning" : ""}`} key={index} />)}</span>;
}

function describeReplayMove(move: ContractMove, boardSize: number) {
  const action = move.action.kind === "place"
    ? `placed at ${coordinate(move.action.to, boardSize)}`
    : `moved ${coordinate(move.action.from, boardSize)} → ${coordinate(move.action.to, boardSize)}`;
  const captures = move.captured.length ? ` · captured ${move.captured.length} ${move.captured.length === 1 ? "piece" : "pieces"}` : "";
  return `${playerName(move.player)} ${action}${captures}`;
}

function replayCellLabel(index: number, piece: "light" | "dark" | null, forbidden: boolean, winning: boolean, boardSize: number) {
  const location = coordinate(index, boardSize);
  if (winning) return `${location}, ${piece ?? "empty"}, winning path`;
  if (forbidden) return `${location}, captured and temporarily unavailable`;
  return `${location}, ${piece ?? "empty"}`;
}

function coordinate(index: number, boardSize: number) {
  return `${String.fromCharCode(65 + (index % boardSize))}${Math.floor(index / boardSize) + 1}`;
}

function playerName(player: "light" | "dark") {
  return player === "light" ? "Light" : "Dark";
}

async function readLatestCrossPlay(): Promise<CrossPlayState> {
  const response = await fetch(`/api/cross-play?runId=${ALL_CROSS_PLAY_RUN_ID}`, { cache: "no-store" });
  const payload = await response.json() as CrossPlayState & { error?: string };
  if (!response.ok) throw new Error(payload.error ?? "No imported cross-play archive available");
  return payload;
}
