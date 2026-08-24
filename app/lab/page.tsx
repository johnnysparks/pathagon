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
        {crossPlay?.latest.length ? <div className="live-game-list" aria-label="Latest cross-play games">{crossPlay.latest.map((game) => <div className="live-game-row" key={game.id}><span className="live-game-number">{shortGameId(game.id)}</span><strong>{game.light}</strong><span>vs</span><strong>{game.dark}</strong><span className="live-game-result">{game.winner ? `${game.winner} · ${game.plies} plies` : `draw · ${game.plies} plies`}</span></div>)}</div> : <p className="live-run-empty">Waiting for an imported offline cross-play result.</p>}
      </section>

      <section className="leaderboard-panel" id="standings" aria-labelledby="standings-title">
        <div className="leaderboard-panel-heading">
          <div><span className="portal-kicker">Standings · imported archive</span><h2 id="standings-title">Every model in the ladder.</h2></div>
          <span className="leaderboard-status"><span /> {crossPlay ? `${ratedAgentCount} rated · archive polling` : "Polling archive"}</span>
        </div>
        <p className="leaderboard-intro">Rankings and records are cumulative across all imported and offline 7×7 cross-play games. Search candidates are shown with their planned budgets until they have games; benchmark metrics remain separate from ladder evidence.</p>

        <div className="leaderboard-table" role="table" aria-label="Current model standings">
          <div className="leaderboard-table-row leaderboard-table-header" role="row">
            <span>#</span><span>Agent</span><span>Role</span><span>Elo</span><span>Record</span><span>Signal</span>
          </div>
          {visibleModels.map((model) => <ModelStanding key={model.id} model={model} live={liveStandingById.get(model.id)} liveRank={liveRankById.get(model.id)} snapshotLoaded={Boolean(crossPlay)} />)}
        </div>
      </section>

      <section className="leaderboard-panel head-to-head-panel" id="head-to-head" aria-labelledby="head-to-head-title">
        <div className="leaderboard-panel-heading">
          <div><span className="portal-kicker">Head-to-head · imported archive</span><h2 id="head-to-head-title">Pairwise results.</h2></div>
          <span className="leaderboard-status"><span /> {crossPlay ? `${crossPlay.headToHead.filter((pairing) => pairing.games > 0).length} active pairings` : "Waiting for poll"}</span>
        </div>
        <p className="leaderboard-intro">Each row is recomputed from the same cumulative archive as the Elo ladder. W–L–D and points are shown from the left model&apos;s perspective; Light starts shows color coverage. Human games stay in their separate archive.</p>

        {crossPlay?.headToHead.length ? <div className="head-to-head-table" role="table" aria-label="Head-to-head model results">
          <div className="head-to-head-row head-to-head-header" role="row"><span>Pairing</span><span>Games</span><span>W–L–D</span><span>Points</span><span>Light starts</span></div>
          {crossPlay.headToHead.map((pairing) => <HeadToHeadRow key={`${pairing.leftId}-${pairing.rightId}`} pairing={pairing} />)}
        </div> : <p className="live-run-empty">Waiting for imported pairwise results.</p>}
      </section>

      <footer className="portal-footer"><span>7×7 model leaderboard</span><span>Read-only view · polling the imported archive</span></footer>
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

async function readLatestCrossPlay(): Promise<CrossPlayState> {
  const response = await fetch(`/api/cross-play?runId=${ALL_CROSS_PLAY_RUN_ID}`, { cache: "no-store" });
  const payload = await response.json() as CrossPlayState & { error?: string };
  if (!response.ok) throw new Error(payload.error ?? "No imported cross-play archive available");
  return payload;
}
