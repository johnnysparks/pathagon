"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import { applyAction, createGame, type GameState } from "../pathagon";
import type { ContractMove, ContractReplayRecord } from "../contract";
import {
  LATEST_RESEARCH,
  LEAGUE_MODELS,
  RANKED_LEAGUE_MODELS,
  RESEARCH_LANES,
} from "../league-models";

const ALL_CROSS_PLAY_RUN_ID = "all-cross-play";
const GAME_THUMBNAIL_RESOLUTION = 256;
const CROSS_PLAY_POLL_MS = 15_000;

const MODELS = LEAGUE_MODELS;
const RANKED_MODEL_IDS = new Set(RANKED_LEAGUE_MODELS.map((model) => model.id));

type LiveStanding = {
  id: string;
  label: string;
  rating: number;
  games: number;
  wins: number;
  losses: number;
  draws: number;
  points: number;
  rustEngine?: boolean;
};

type LiveGame = {
  id: string;
  recordedAt: string;
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
  rankedGames?: number;
  status: "ready" | "running" | "complete";
  standings: LiveStanding[];
  headToHead: HeadToHead[];
  latest: LiveGame[];
};

type HistoryPage = {
  runId: string;
  games: LiveGame[];
  total: number;
  limit: number;
  offset: number;
  hasMore: boolean;
};

type HeadToHeadSort = "games-desc" | "games-asc" | "win-rate-desc" | "loss-rate-asc" | "draw-rate-desc" | "score-rate-desc" | "pairing-asc";

type HeadToHeadView = {
  leftId: string;
  rightId: string;
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

type GameSet = "all" | "pairwise";
type MobileView = "sets" | "games" | "game";
type PairSelection = {
  leftId: string;
  rightId: string;
  leftLabel: string;
  rightLabel: string;
  games: number;
};

export default function LearningLab() {
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [crossPlay, setCrossPlay] = useState<CrossPlayState | null>(null);
  const [crossPlayError, setCrossPlayError] = useState<string | null>(null);
  const [activeSet, setActiveSet] = useState<GameSet>("all");
  const [selectedPair, setSelectedPair] = useState<PairSelection | null>(null);
  const [mobileView, setMobileView] = useState<MobileView>("sets");
  const [historyGames, setHistoryGames] = useState<LiveGame[]>([]);
  const [historyTotal, setHistoryTotal] = useState<number | null>(null);
  const [historyHasMore, setHistoryHasMore] = useState(true);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [replaySummary, setReplaySummary] = useState<LiveGame | null>(null);
  const [replayGame, setReplayGame] = useState<ArchivedReplayGame | null>(null);
  const [replayPly, setReplayPly] = useState(0);
  const [replayPlaying, setReplayPlaying] = useState(false);
  const [replayLoadingId, setReplayLoadingId] = useState<string | null>(null);
  const [replayError, setReplayError] = useState<string | null>(null);
  const [replayModalOpen, setReplayModalOpen] = useState(false);
  const [headToHeadSearch, setHeadToHeadSearch] = useState("");
  const [headToHeadFocus, setHeadToHeadFocus] = useState("all");
  const [headToHeadSort, setHeadToHeadSort] = useState<HeadToHeadSort>("games-desc");
  const replayRequest = useRef(0);
  const historySentinel = useRef<HTMLDivElement | null>(null);
  const historyOffset = useRef(0);
  const historyHasMoreRef = useRef(true);
  const historyLoadingRef = useRef(false);
  const historyRequest = useRef(0);

  const liveStandingById = useMemo(
    () => new Map((crossPlay?.standings ?? []).map((standing) => [standing.id, standing])),
    [crossPlay],
  );
  const liveRankById = useMemo(
    () => new Map(
      (crossPlay?.standings ?? [])
        .filter((standing) => standing.rustEngine ?? RANKED_MODEL_IDS.has(standing.id))
        .map((standing, index) => [standing.id, String(index + 1).padStart(2, "0")]),
    ),
    [crossPlay],
  );
  const profileModels = useMemo(() => [...RANKED_LEAGUE_MODELS].sort((left, right) => {
    const leftRank = liveRankById.get(left.id);
    const rightRank = liveRankById.get(right.id);
    if (leftRank && rightRank) return Number(leftRank) - Number(rightRank);
    if (leftRank) return -1;
    if (rightRank) return 1;
    return RANKED_LEAGUE_MODELS.indexOf(left) - RANKED_LEAGUE_MODELS.indexOf(right);
  }), [liveRankById]);

  const scopedHeadToHead = useMemo(() => {
    if (!crossPlay) return [];
    const rankedPairings = crossPlay.headToHead.filter((pairing) => RANKED_MODEL_IDS.has(pairing.leftId) && RANKED_MODEL_IDS.has(pairing.rightId));
    if (headToHeadFocus === "all") return rankedPairings;
    return rankedPairings.filter((pairing) => pairing.leftId === headToHeadFocus || pairing.rightId === headToHeadFocus);
  }, [crossPlay, headToHeadFocus]);

  const visibleHeadToHead = useMemo(() => {
    const query = headToHeadSearch.trim().toLowerCase();
    const filtered = scopedHeadToHead
      .filter((pairing) => {
        const active = pairing.games > 0;
        const searchable = `${pairing.leftLabel} ${pairing.rightLabel} ${pairing.leftId} ${pairing.rightId}`.toLowerCase();
        return (!query || searchable.includes(query))
          && active;
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
  }, [headToHeadFocus, headToHeadSearch, headToHeadSort, scopedHeadToHead]);

  const strengthLeaderLive = crossPlay?.standings.find((standing) => (standing.rustEngine ?? RANKED_MODEL_IDS.has(standing.id)) && standing.games > 0);
  const ratedAgentCount = crossPlay?.standings.filter((standing) => (standing.rustEngine ?? RANKED_MODEL_IDS.has(standing.id)) && standing.games > 0).length ?? RANKED_LEAGUE_MODELS.length;

  useEffect(() => {
    const savedTheme = window.localStorage.getItem("pathagon-lab-theme");
    const preferredTheme = savedTheme === "dark" || savedTheme === "light"
      ? savedTheme
      : window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    const timer = window.setTimeout(() => setTheme(preferredTheme), 0);
    return () => window.clearTimeout(timer);
  }, []);

  const loadHistoryPage = useCallback(async (reset = false) => {
    if (!reset && (historyLoadingRef.current || !historyHasMoreRef.current)) return;
    const requestId = historyRequest.current + 1;
    historyRequest.current = requestId;
    const offset = reset ? 0 : historyOffset.current;
    if (reset) {
      historyOffset.current = 0;
      historyHasMoreRef.current = true;
      setHistoryGames([]);
      setHistoryTotal(null);
    }
    if (activeSet === "pairwise" && !selectedPair) {
      historyHasMoreRef.current = false;
      setHistoryHasMore(false);
      setHistoryTotal(0);
      setHistoryError(null);
      historyLoadingRef.current = false;
      setHistoryLoading(false);
      return;
    }
    historyLoadingRef.current = true;
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const params = new URLSearchParams({ runId: ALL_CROSS_PLAY_RUN_ID, history: "1", limit: "24", offset: String(offset) });
      if (selectedPair) {
        params.set("pairLeft", selectedPair.leftId);
        params.set("pairRight", selectedPair.rightId);
      }
      const response = await fetch(`/api/cross-play?${params.toString()}`, { cache: "no-store" });
      const payload = await response.json() as HistoryPage & { error?: string };
      if (requestId !== historyRequest.current) return;
      if (!response.ok) throw new Error(payload.error ?? "History unavailable");
      setHistoryGames((current) => reset ? payload.games : appendUniqueGames(current, payload.games));
      historyOffset.current = offset + payload.games.length;
      historyHasMoreRef.current = payload.hasMore;
      setHistoryHasMore(payload.hasMore);
      setHistoryTotal(payload.total);
    } catch (error: unknown) {
      if (requestId === historyRequest.current) setHistoryError(error instanceof Error ? error.message : "History unavailable");
    } finally {
      if (requestId === historyRequest.current) {
        historyLoadingRef.current = false;
        setHistoryLoading(false);
      }
    }
  }, [activeSet, selectedPair]);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadHistoryPage(true), 0);
    return () => window.clearTimeout(timer);
  }, [loadHistoryPage]);

  useEffect(() => {
    const sentinel = historySentinel.current;
    if (!sentinel) return;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) void loadHistoryPage();
    }, { rootMargin: "360px 0px" });
    observer.observe(sentinel);
    return () => observer.disconnect();
  }, [historyGames.length, historyHasMore, loadHistoryPage]);

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
        setMobileView("games");
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
    const timer = window.setInterval(refresh, CROSS_PLAY_POLL_MS);
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
    setMobileView("games");
  }

  function changeReplayPly(nextPly: number) {
    if (!replayGame) return;
    setReplayPlaying(false);
    setReplayPly(Math.max(0, Math.min(replayGame.record.moves.length, nextPly)));
  }

  function selectPairing(pairing: HeadToHeadView) {
    const left = MODELS.find((model) => model.id === pairing.leftId);
    const right = MODELS.find((model) => model.id === pairing.rightId);
    if (!left || !right || pairing.games === 0) return;
    setSelectedPair({ leftId: left.id, rightId: right.id, leftLabel: left.name, rightLabel: right.name, games: pairing.games });
    setMobileView("games");
  }

  return (
    <main className={`portal-app leaderboard-app ${theme === "dark" ? "dark" : ""}`}>
      <nav className="portal-nav" aria-label="Model league navigation">
        <Link className="portal-breadcrumb" href="/">
          <span className="portal-mark">P</span>
          <span>Pathagon</span>
          <span className="portal-slash">/</span>
          <span>Model league</span>
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

      <header className="leaderboard-compact-header">
        <div className="leaderboard-compact-copy">
          <span className="portal-kicker">7×7 model league</span>
          <h1>Model league.</h1>
          <p>Browse the supported ladder, compare pairings, replay archived games, and see why Transition v4 is the current default.</p>
        </div>
        <div className="leaderboard-compact-meta" aria-label="Archive summary">
          <span className="leaderboard-status polling-status"><span /> {crossPlay ? "Polling" : "Connecting"}</span>
          <div><strong>{crossPlay?.games ?? "—"}</strong><span>archive games</span></div>
          <div><strong>{crossPlay?.rankedGames ?? "—"}</strong><span>ranked games</span></div>
          <div><strong>{RANKED_LEAGUE_MODELS.length}</strong><span>ranked agents</span></div>
        </div>
      </header>

      {crossPlayError ? <p className="live-run-error compact-error" role="status">{crossPlayError} · retrying automatically</p> : null}

      <section className="player-profiles" aria-labelledby="player-profiles-title">
        <div className="player-profiles-heading">
          <div><span className="portal-kicker">The ranked roster</span><h2 id="player-profiles-title">Players in the league.</h2></div>
          <p>{RANKED_LEAGUE_MODELS.length} Rust-engine opponents · live Elo updates as imported games land.</p>
        </div>
        <div className="player-profile-grid">
          {profileModels.map((model) => <PlayerProfileCard key={model.id} model={model} live={liveStandingById.get(model.id)} liveRank={liveRankById.get(model.id)} snapshotLoaded={Boolean(crossPlay)} />)}
        </div>
      </section>

      <div className={`game-browser mobile-view-${mobileView}`}>
      <section className="leaderboard-panel game-sets-panel" aria-labelledby="game-sets-title">
        <div className="game-section-heading">
          <div><span className="portal-kicker">Section one</span><h2 id="game-sets-title">Game sets</h2><p>Choose a lens for the archive. Research-only identities stay out of rank.</p></div>
          <span className="leaderboard-status"><span /> {crossPlay ? `${ratedAgentCount} rated` : "Waiting for poll"}</span>
        </div>
        <div className="game-set-tabs" role="tablist" aria-label="Game sets">
          <button className={`game-set-tab ${activeSet === "all" ? "active" : ""}`} type="button" role="tab" aria-selected={activeSet === "all"} onClick={() => { setActiveSet("all"); setSelectedPair(null); }}>
            <strong>All games</strong><span>Archive · {crossPlay?.games ?? "—"} results</span>
          </button>
          <button className={`game-set-tab ${activeSet === "pairwise" ? "active" : ""}`} type="button" role="tab" aria-selected={activeSet === "pairwise"} onClick={() => setActiveSet("pairwise")}>
            <strong>Pairwise results</strong><span>{crossPlay ? `${crossPlay.headToHead.filter((pairing) => pairing.games > 0).length} active matchups` : "Compare matchups"}</span>
          </button>
        </div>

        {activeSet === "all" ? <div className="game-set-content" role="tabpanel">
          <div className="game-set-summary">
            <div><span className="game-set-label">Archive browser</span><strong>Every imported game</strong><p>Newest results appear in Games. Open a replay to inspect the move-by-move evidence.</p></div>
            <div className="game-set-highlight"><span>Ranked roster</span><strong>{RANKED_LEAGUE_MODELS.length} Rust profiles</strong><small>{strengthLeaderLive ? `${strengthLeaderLive.label} leads · ${strengthLeaderLive.rating.toLocaleString()} Elo` : "Waiting for live standings"}</small></div>
          </div>
          <button className="mobile-section-cta" type="button" onClick={() => setMobileView("games")}>View {crossPlay?.games ?? "all"} games <span>→</span></button>
        </div> : <div className="game-set-content" role="tabpanel">
          <p className="game-set-intro">Select a played matchup to make it the active game list. Only rankable agents appear here; research-only runs remain available in the archive below.</p>
          <div className="table-toolbar compact-table-toolbar pairwise-toolbar" aria-label="Pairwise table controls">
            <label className="table-control table-search" htmlFor="head-to-head-search"><span>Search</span><input id="head-to-head-search" type="search" value={headToHeadSearch} onChange={(event) => setHeadToHeadSearch(event.target.value)} placeholder="Model or pairing" /></label>
            <label className="table-control" htmlFor="head-to-head-focus"><span>Focus</span><select id="head-to-head-focus" value={headToHeadFocus} onChange={(event) => setHeadToHeadFocus(event.target.value)}><option value="all">All models</option>{RANKED_LEAGUE_MODELS.map((model) => <option value={model.id} key={model.id}>{model.name}</option>)}</select></label>
            <label className="table-control" htmlFor="head-to-head-sort"><span>Sort</span><select id="head-to-head-sort" value={headToHeadSort} onChange={(event) => setHeadToHeadSort(event.target.value as HeadToHeadSort)}><option value="games-desc">Play count</option><option value="win-rate-desc">Win rate</option><option value="score-rate-desc">Score rate</option><option value="pairing-asc">Opponent · A to Z</option></select></label>
            <span className="table-result-count" aria-live="polite">{visibleHeadToHead.length} matchups</span>
          </div>
          {crossPlay?.headToHead.length ? visibleHeadToHead.length ? <div className="head-to-head-table game-set-pairings" role="table" aria-label="Head-to-head model results">
            <div className="head-to-head-row head-to-head-header" role="row"><span>Matchup</span><span>Games</span><span>W–L–D</span><span>Score</span></div>
            {visibleHeadToHead.map((pairing) => <HeadToHeadRow key={`${pairing.leftId}-${pairing.rightId}`} pairing={pairing} selected={selectedPair?.leftId === pairing.leftId && selectedPair.rightId === pairing.rightId} onSelect={() => selectPairing(pairing)} />)}
          </div> : <p className="table-empty-state" role="status"><strong>No pairings match.</strong><span>Try a different search.</span></p> : <p className="live-run-empty">Waiting for imported pairwise results.</p>}
          {selectedPair ? <button className="mobile-section-cta" type="button" onClick={() => setMobileView("games")}>View {selectedPair.games} games <span>→</span></button> : null}
        </div>}
      </section>

      <section className="leaderboard-panel game-list-panel" aria-labelledby="game-list-title">
        <div className="game-section-heading game-list-heading">
          <button className="mobile-back-button" type="button" onClick={() => setMobileView("sets")}><span>←</span> Game sets</button>
          <div><span className="portal-kicker">Section two</span><h2 id="game-list-title">Games</h2><p>{activeSet === "pairwise" ? selectedPair ? `${selectedPair.leftLabel} vs ${selectedPair.rightLabel} · newest first` : "Choose a pairing from Game sets" : "All cross-play results · newest first"}</p></div>
          <div className="game-list-heading-meta"><span className="archive-history-count">{historyTotal === null ? "Loading" : `${historyGames.length} of ${historyTotal}`}</span><span>games loaded</span></div>
        </div>
        {activeSet === "pairwise" && !selectedPair ? <div className="game-list-empty"><strong>Choose a pairing</strong><span>Open Pairwise results in Game sets and select any played matchup.</span><button type="button" onClick={() => setMobileView("sets")}>Browse pairings <span>→</span></button></div> : historyGames.length ? <>
          <div className="game-list-toolbar"><span>Scroll to load older games</span><span>{historyHasMore ? "More available" : "End of history"}</span></div>
          <div className="game-card-grid" aria-label="Games">
            {historyGames.map((game) => <button className="game-card" type="button" key={game.id} onClick={() => { setMobileView("game"); void openReplay(game); }} aria-label={`Replay ${game.light} versus ${game.dark}`}>
              <GameThumbnail board={game.finalBoard} winningPath={game.winningPath} />
              <span className="game-card-body"><span className="game-card-topline"><span>{shortGameId(game.id)}</span><time dateTime={game.recordedAt}>{formatArchiveDate(game.recordedAt)}</time></span><span className="game-card-match"><strong>{game.light}</strong><span>vs</span><strong>{game.dark}</strong></span><span className="game-card-footer"><span>{game.winner ? `${game.winner} won` : "Draw"} · {game.plies} plies</span><em>Replay ↗</em></span></span>
            </button>)}
          </div>
        </> : historyLoading ? <div className="game-list-empty"><strong>Loading games…</strong><span>Fetching the newest results from the archive.</span></div> : historyError ? <div className="game-list-empty error" role="alert"><strong>Games couldn&apos;t load.</strong><span>{historyError}</span><button type="button" onClick={() => void loadHistoryPage(true)}>Try again</button></div> : <div className="game-list-empty"><strong>No games yet</strong><span>This set does not have any stored results.</span></div>}
        <div className="history-load-sentinel" ref={historySentinel} aria-live="polite">
          {historyLoading && historyGames.length ? "Loading older games…" : historyError && historyGames.length ? <button type="button" onClick={() => void loadHistoryPage()}>Try again</button> : !historyHasMore && historyGames.length ? "End of history" : historyGames.length ? "Keep scrolling for more" : null}
        </div>
      </section>
      </div>

      <section className="leaderboard-panel research-panel" aria-labelledby="research-title">
        <div className="research-panel-heading">
          <div>
            <span className="portal-kicker">Research ledger · {LATEST_RESEARCH.researchPath}</span>
            <h2 id="research-title">The latest signal is shipped.</h2>
            <p>Transition v4 is the promoted Pathfinder default. The ladder stays limited to rankable identities; historical candidates and controls remain documented here without being presented as supported strength.</p>
          </div>
          <span className="research-promotion-badge"><span /> {LATEST_RESEARCH.status}</span>
        </div>
        <div className="research-evidence-layout">
          <article className="research-feature-card">
            <div className="research-feature-topline"><span>Current default</span><strong>{LATEST_RESEARCH.title}</strong></div>
            <p>Explicit placement/relocation transition scoring with tactical-safe root ordering, trained from a {LATEST_RESEARCH.trainingViewRoots.toLocaleString()}-root corpus ({LATEST_RESEARCH.trainingRoots.toLocaleString()} train · {LATEST_RESEARCH.heldoutRoots.toLocaleString()} held out), and evaluated with the same bounded search envelope as the incumbent.</p>
            <dl className="research-metric-grid">
              <div><dt>Arena</dt><dd>{LATEST_RESEARCH.arenaWins}–{LATEST_RESEARCH.arenaLosses}–{LATEST_RESEARCH.arenaDraws}</dd><small>{LATEST_RESEARCH.arenaGames} paired games</small></div>
              <div><dt>Game points</dt><dd>{formatPercent(LATEST_RESEARCH.arenaPointRate)}</dd><small>{formatPercent(LATEST_RESEARCH.lightPointRate)} Light · {formatPercent(LATEST_RESEARCH.darkPointRate)} Dark</small></div>
              <div><dt>Held-out top 1</dt><dd>{formatPercent(LATEST_RESEARCH.heldoutTop1 / LATEST_RESEARCH.heldoutRoots)}</dd><small>{LATEST_RESEARCH.heldoutTop1.toLocaleString()} / {LATEST_RESEARCH.heldoutRoots.toLocaleString()} roots · top 3 {formatPercent(LATEST_RESEARCH.heldoutTop3 / LATEST_RESEARCH.heldoutRoots)}</small></div>
              <div><dt>Replay audit</dt><dd>0</dd><small>legality or capture mismatches</small></div>
            </dl>
            <p className="research-audit-note">{LATEST_RESEARCH.replayAudit}</p>
          </article>
          <div className="research-ledger-column">
            <div className="research-lineage-card">
              <div className="research-card-heading"><span className="portal-kicker">Promotion trail</span><span>v4 → v3 → v0.5</span></div>
              <ul className="research-lane-list">
                {RESEARCH_LANES.map((lane) => <li key={lane.label} className={`research-lane ${lane.tone}`}><span className="research-lane-dot" /><div><strong>{lane.label}</strong><p>{lane.detail}</p></div></li>)}
              </ul>
            </div>
            <div className="research-retention-card"><span className="portal-kicker">Retained controls</span><strong>Rollback and history stay visible.</strong><p>{LATEST_RESEARCH.retainedControls}. No discarded research artifact is treated as a current opponent.</p></div>
          </div>
        </div>
      </section>

      <footer className="portal-footer"><span>7×7 Rust-engine model leaderboard</span><span>Read-only archive · research lanes are not ranked</span></footer>
      {replayModalOpen ? <ReplayModal summary={replaySummary} game={replayGame} loading={Boolean(replayLoadingId)} error={replayError} ply={replayPly} playing={replayPlaying} onClose={closeReplay} onPlayPause={() => setReplayPlaying((current) => !current)} onPlyChange={changeReplayPly} /> : null}
    </main>
  );
}

function ModelGlyph({ tone, glyph }: { tone: string; glyph: string }) {
  return <span className={`model-glyph ${tone}`} aria-hidden="true">{glyph}</span>;
}

function HeadToHeadRow({ pairing, selected = false, onSelect }: { pairing: HeadToHeadView; selected?: boolean; onSelect?: () => void }) {
  const active = pairing.games > 0;
  return <button className={`head-to-head-row ${active ? "" : "disabled"} ${selected ? "selected" : ""}`} type="button" role="row" disabled={!onSelect || !active} aria-selected={selected} onClick={onSelect}><div className="head-to-head-match"><strong>{pairing.focusLabel}</strong><span>vs</span><strong>{pairing.opponentLabel}</strong></div><span className="head-to-head-games">{active ? pairing.games : "—"}</span><span className="head-to-head-record">{active ? `${pairing.wins}–${pairing.losses}–${pairing.draws}` : "no games"}</span><div className="head-to-head-rates" aria-label={active ? `Win ${formatRate(pairing.winRate)}, loss ${formatRate(pairing.lossRate)}, draw ${formatRate(pairing.drawRate)}` : "No games"}><span><b>W</b>{active ? formatRate(pairing.winRate) : "—"}</span><span><b>L</b>{active ? formatRate(pairing.lossRate) : "—"}</span><span><b>D</b>{active ? formatRate(pairing.drawRate) : "—"}</span></div><span className="head-to-head-score">{active ? formatRate(pairing.scoreRate) : "—"}</span><span className="head-to-head-colors">{active ? `${pairing.focusLightGames}–${pairing.opponentLightGames}` : "disabled"}</span></button>;
}

function orientHeadToHead(pairing: HeadToHead, focusId: string): HeadToHeadView {
  const focusIsRight = focusId !== "all" && pairing.rightId === focusId;
  const games = pairing.games;
  const wins = focusIsRight ? pairing.rightWins : pairing.leftWins;
  const losses = focusIsRight ? pairing.leftWins : pairing.rightWins;
  const draws = pairing.draws;
  const points = focusIsRight ? pairing.rightPoints : pairing.leftPoints;
  return {
    leftId: pairing.leftId,
    rightId: pairing.rightId,
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

function formatPercent(rate: number) {
  return `${(rate * 100).toFixed(1)}%`;
}

function PlayerProfileCard({ model, live, liveRank, snapshotLoaded }: { model: (typeof RANKED_LEAGUE_MODELS)[number]; live?: LiveStanding; liveRank?: string; snapshotLoaded: boolean }) {
  const liveActive = Boolean(live?.games);
  const rank = liveRank ?? "—";
  const status = model.status === "default" ? "Current default" : model.status === "control" ? "Control" : "Baseline";
  return <article className={`player-profile-card ${model.tone} ${rank === "01" ? "leader" : ""}`}>
    <div className="player-profile-topline"><span>#{rank}</span><small>{status}</small></div>
    <div className="player-profile-identity"><ModelGlyph tone={model.tone} glyph={model.glyph} /><div><h3>{model.nickname ?? model.name}</h3><span>{model.name}</span></div></div>
    <div className="player-profile-rating"><strong>{liveActive ? live!.rating.toLocaleString() : "—"}</strong><span>Elo rating</span></div>
    <p className="player-profile-detail">{model.mechanics ?? model.family}</p>
    <div className="player-profile-footer"><span>{model.family}</span><strong>{liveActive ? `${formatRecord(live!)} record` : snapshotLoaded ? "Awaiting matches" : "Connecting"}</strong></div>
  </article>;
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

function appendUniqueGames(current: LiveGame[], incoming: LiveGame[]) {
  const known = new Set(current.map((game) => game.id));
  return [...current, ...incoming.filter((game) => !known.has(game.id))];
}

function formatArchiveDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}
