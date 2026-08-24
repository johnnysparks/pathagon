"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

const ALL_CROSS_PLAY_RUN_ID = "all-cross-play";

const MODELS = [
  {
    id: "pathfinder-v0.3.0",
    name: "The Pathfinder",
    family: "4-ply iterative search",
    role: "playable opponent",
    tone: "green",
    glyph: "P",
    disabled: false,
  },
  {
    id: "surveyor-v0.2.0",
    name: "The Surveyor",
    family: "2-ply broad-beam search",
    role: "playable opponent",
    tone: "violet",
    glyph: "S",
    disabled: false,
  },
  {
    id: "lunatic-v0.1.0",
    name: "Lunatic",
    family: "1-ply pattern heuristic",
    role: "playable opponent",
    tone: "gold",
    glyph: "L",
    disabled: false,
  },
  {
    id: "coin-flip-v0.0.1",
    name: "Coin Flip",
    family: "Random legal action",
    role: "playable opponent",
    tone: "muted",
    glyph: "C",
    disabled: false,
  },
  {
    id: "gnn-warmstart-7x7",
    name: "GNN Learner",
    family: "64 channels · 8 message layers",
    role: "neural candidate",
    tone: "green",
    glyph: "G",
    disabled: false,
  },
  {
    id: "cnn-baseline-7x7",
    name: "CNN baseline",
    family: "7×7 residual CNN · 87.4k params",
    role: "neural candidate",
    tone: "gold",
    glyph: "C",
    disabled: false,
  },
  {
    id: "gnn-scout-7x7",
    name: "GNN Scout",
    family: "Compact message passing · 17.5k params",
    role: "neural data generator",
    tone: "violet",
    glyph: "S",
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

export default function LearningLab() {
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [crossPlay, setCrossPlay] = useState<CrossPlayState | null>(null);
  const [crossPlayError, setCrossPlayError] = useState<string | null>(null);

  const liveStandingById = useMemo(
    () => new Map((crossPlay?.standings ?? []).map((standing) => [standing.id, standing])),
    [crossPlay],
  );
  const liveRankById = useMemo(
    () => new Map((crossPlay?.standings ?? []).map((standing, index) => [standing.id, String(index + 1).padStart(2, "0")])),
    [crossPlay],
  );
  const visibleModels = useMemo(() => {
    if (!crossPlay) return MODELS;
    return [...MODELS].sort((left, right) => {
      const leftRank = liveRankById.get(left.id);
      const rightRank = liveRankById.get(right.id);
      if (leftRank && rightRank) return Number(leftRank) - Number(rightRank);
      if (leftRank) return -1;
      if (rightRank) return 1;
      return Number(Boolean(left.disabled)) - Number(Boolean(right.disabled));
    });
  }, [crossPlay, liveRankById]);

  const strengthLeaderLive = crossPlay?.standings[0];
  const strengthLeader = MODELS.find((model) => model.id === strengthLeaderLive?.id) ?? MODELS[0];

  useEffect(() => {
    const savedTheme = window.localStorage.getItem("pathagon-lab-theme");
    const preferredTheme = savedTheme === "dark" || savedTheme === "light"
      ? savedTheme
      : window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    const timer = window.setTimeout(() => setTheme(preferredTheme), 0);
    return () => window.clearTimeout(timer);
  }, []);

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
          if (active) setCrossPlayError(error instanceof Error ? error.message : "Live archive unavailable");
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
          <p>Live model rankings backed by every archived cross-play result.</p>
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
        <LeaderboardStat label="Live agents" value={crossPlay ? String(crossPlay.standings.length) : "—"} detail="4 playable · 3 neural" accent="green" />
        <LeaderboardStat label="7×7 benchmark" value="3,251" detail="2,037 unique · 416 held out" accent="gold" />
        <LeaderboardStat label="Live cross-play" value={crossPlay ? String(crossPlay.games) : "—"} detail={crossPlay ? "cumulative database records" : "waiting for first poll"} accent="gold" />
        <LeaderboardStat label="Held-out policy NLL" value="2.112" detail="GNN · 416 held-out records" accent="ink" />
      </section>

      <section className="leaderboard-panel cross-play-live-panel" aria-labelledby="live-run-title">
        <div className="cross-play-live-heading">
          <div><span className="portal-kicker">Live archive · read-only</span><h2 id="live-run-title">Cumulative cross-play</h2><p>The browser polls the database every 0.9 seconds. Add games from chat or the terminal and this view will pick up the new result automatically.</p></div>
          <span className="leaderboard-status polling-status"><span /> {crossPlay ? "Polling" : "Connecting"}</span>
        </div>
        <div className="live-run-summary">
          <div><strong>{crossPlay?.games ?? "—"}<small>{crossPlay ? ` / ${crossPlay.targetGames}` : ""}</small></strong><span>games counted</span></div>
          <div><strong>{crossPlay ? "Polling" : "Connecting"}</strong><span>browser status</span></div>
          <div><strong>{crossPlay?.latest[0]?.winner ?? (crossPlay?.latest[0] ? "draw" : "—")}</strong><span>latest stored result</span></div>
        </div>
        {crossPlayError ? <p className="live-run-error" role="status">{crossPlayError} · retrying automatically</p> : null}
        {crossPlay?.latest.length ? <div className="live-game-list" aria-label="Latest cross-play games">{crossPlay.latest.map((game) => <div className="live-game-row" key={game.id}><span className="live-game-number">{shortGameId(game.id)}</span><strong>{game.light}</strong><span>vs</span><strong>{game.dark}</strong><span className="live-game-result">{game.winner ? `${game.winner} · ${game.plies} plies` : `draw · ${game.plies} plies`}</span></div>)}</div> : <p className="live-run-empty">Waiting for an archived cross-play result.</p>}
      </section>

      <section className="leaderboard-panel" id="standings" aria-labelledby="standings-title">
        <div className="leaderboard-panel-heading">
          <div><span className="portal-kicker">Standings · live database</span><h2 id="standings-title">Every model in the ladder.</h2></div>
          <span className="leaderboard-status"><span /> {crossPlay ? `${crossPlay.standings.length} agents · polling` : "Polling database"}</span>
        </div>
        <p className="leaderboard-intro">Rankings and records are cumulative across all archived 7×7 cross-play games. Offline benchmark metrics remain separate from live ladder evidence.</p>

        <div className="leaderboard-table" role="table" aria-label="Current model standings">
          <div className="leaderboard-table-row leaderboard-table-header" role="row">
            <span>#</span><span>Agent</span><span>Role</span><span>Elo</span><span>Record</span><span>Signal</span>
          </div>
          {visibleModels.map((model) => <ModelStanding key={model.id} model={model} live={liveStandingById.get(model.id)} liveRank={liveRankById.get(model.id)} snapshotLoaded={Boolean(crossPlay)} />)}
        </div>
      </section>

      <section className="leaderboard-panel head-to-head-panel" id="head-to-head" aria-labelledby="head-to-head-title">
        <div className="leaderboard-panel-heading">
          <div><span className="portal-kicker">Head-to-head · cross-play only</span><h2 id="head-to-head-title">Pairwise results.</h2></div>
          <span className="leaderboard-status"><span /> {crossPlay ? `${crossPlay.headToHead.filter((pairing) => pairing.games > 0).length} active pairings` : "Waiting for poll"}</span>
        </div>
        <p className="leaderboard-intro">Each row is recomputed from the same cumulative archive as the Elo ladder. W–L–D and points are shown from the left model&apos;s perspective; Light starts shows color coverage. Human games stay in their separate archive.</p>

        {crossPlay?.headToHead.length ? <div className="head-to-head-table" role="table" aria-label="Head-to-head model results">
          <div className="head-to-head-row head-to-head-header" role="row"><span>Pairing</span><span>Games</span><span>W–L–D</span><span>Points</span><span>Light starts</span></div>
          {crossPlay.headToHead.map((pairing) => <HeadToHeadRow key={`${pairing.leftId}-${pairing.rightId}`} pairing={pairing} />)}
        </div> : <p className="live-run-empty">Waiting for live pairwise results.</p>}
      </section>

      <footer className="portal-footer"><span>7×7 model leaderboard</span><span>Read-only view · polling the live archive</span></footer>
    </main>
  );
}

function ModelGlyph({ tone, glyph }: { tone: string; glyph: string }) {
  return <span className={`model-glyph ${tone}`} aria-hidden="true">{glyph}</span>;
}

function LeaderboardStat({ label, value, detail, accent }: { label: string; value: string; detail: string; accent: string }) {
  return <div className={`leaderboard-stat ${accent}`}><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}

function HeadToHeadRow({ pairing }: { pairing: HeadToHead }) {
  const active = pairing.games > 0;
  return <div className={`head-to-head-row ${active ? "" : "disabled"}`} role="row"><div className="head-to-head-match"><strong>{pairing.leftLabel}</strong><span>vs</span><strong>{pairing.rightLabel}</strong></div><span className="head-to-head-games">{active ? pairing.games : "—"}</span><span className="head-to-head-record">{active ? `${pairing.leftWins}–${pairing.rightWins}–${pairing.draws}` : "no games"}</span><span className="head-to-head-points">{active ? `${pairing.leftPoints.toFixed(1)}–${pairing.rightPoints.toFixed(1)}` : "—"}</span><span className="head-to-head-colors">{active ? `${pairing.leftLightGames}–${pairing.rightLightGames}` : "disabled"}</span></div>;
}

function ModelStanding({ model, live, liveRank, snapshotLoaded }: { model: (typeof MODELS)[number]; live?: LiveStanding; liveRank?: string; snapshotLoaded: boolean }) {
  const liveActive = Boolean(live?.games);
  const disabled = Boolean(model.disabled);
  const waiting = !liveActive && !disabled;
  const rank = liveActive ? liveRank ?? "—" : "—";
  const record = liveActive ? formatRecord(live!) : disabled ? "disabled" : "—";
  const signal = liveActive ? `${live!.games} games` : disabled ? "offline only" : snapshotLoaded ? "no live games" : "waiting for poll";
  const signalDetail = liveActive ? `${live!.points.toFixed(1)} points · cumulative` : disabled ? "not rated" : "no ladder evidence";
  return <div className={`leaderboard-table-row model-standing ${rank === "01" ? "leader" : ""} ${disabled || waiting ? "disabled" : ""}`} role="row"><span className="model-rank">{rank}</span><div className="standing-model"><ModelGlyph tone={model.tone} glyph={model.glyph} /><div><strong>{model.name}</strong><span>{model.family}</span></div></div><div className="standing-role"><strong>{disabled ? "Disabled" : liveActive ? "Live ladder" : "Waiting"}</strong><span>{model.role}</span></div><span className="standing-elo">{liveActive ? live!.rating.toLocaleString() : "—"}</span><div className="standing-record"><strong>{record}</strong><span>{liveActive ? "cumulative" : disabled ? "not rated" : "no games"}</span></div><div className="standing-signal"><strong>{signal}</strong><span>{signalDetail}</span></div></div>;
}

function formatRecord(standing: Pick<LiveStanding, "wins" | "losses" | "draws">) {
  return `${standing.wins}–${standing.losses}–${standing.draws}`;
}

function shortGameId(id: string) {
  const suffix = id.split("-").pop();
  return suffix && /^\d+$/.test(suffix) ? `#${suffix}` : `…${id.slice(-8)}`;
}

async function readLatestCrossPlay(): Promise<CrossPlayState> {
  const response = await fetch(`/api/cross-play?runId=${ALL_CROSS_PLAY_RUN_ID}`, { cache: "no-store" });
  const payload = await response.json() as CrossPlayState & { error?: string };
  if (!response.ok) throw new Error(payload.error ?? "No live archive available");
  return payload;
}
