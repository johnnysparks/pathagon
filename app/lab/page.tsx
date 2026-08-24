"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";

const BATCH_COMMAND = `./.venv-pathagon-gnn/bin/python scripts/generate-7x7-selfplay.py \\
  --games-per-player 1000 --players scout,learner,cnn \\
  --workers 8 --simulations 4 --temperature-moves 32 \\
  --max-plies 196 \\
  --output-dir training/gnn/benchmark-7x7/generated/<batch-id>`;

const TARGET_CROSS_PLAY_GAMES = 10;

const MODELS = [
  {
    id: "pathfinder-v0.3.0",
    rank: "01",
    name: "The Pathfinder",
    family: "4-ply iterative search",
    role: "playable anchor",
    elo: "1,142",
    record: "14–0–0",
    signal: "archive leader",
    signalDetail: "14 balanced games",
    tone: "green",
    glyph: "P",
    status: "Strength leader",
  },
  {
    id: "surveyor-v0.2.0",
    rank: "02",
    name: "The Surveyor",
    family: "2-ply broad-beam search",
    role: "playable anchor",
    elo: "1,085",
    record: "11–2–1",
    signal: "second anchor",
    signalDetail: "14 balanced games",
    tone: "violet",
    glyph: "S",
    status: "Provisional #2",
  },
  {
    id: "lunatic-v0.1.0",
    rank: "03",
    name: "Lunatic",
    family: "1-ply pattern heuristic",
    role: "playable breadth",
    elo: "1,059",
    record: "10–3–1",
    signal: "third anchor",
    signalDetail: "14 balanced games",
    tone: "gold",
    glyph: "L",
    status: "Playable",
  },
  {
    id: "coin-flip-v0.0.1",
    rank: "04",
    name: "Coin Flip",
    family: "Random legal action",
    role: "playable floor",
    elo: "935",
    record: "2–8–4",
    signal: "control floor",
    signalDetail: "14 balanced games",
    tone: "muted",
    glyph: "C",
    status: "Playable",
  },
  {
    id: "gnn-warmstart-7x7",
    rank: "05",
    name: "GNN Learner",
    family: "64 channels · 8 message layers",
    role: "promotion candidate",
    elo: "929",
    record: "0–4–0",
    signal: "fresh result",
    signalDetail: "Surveyor sweep · 4 games",
    tone: "green",
    glyph: "G",
    status: "Fresh match",
  },
  {
    id: "gnn-scout-7x7",
    rank: "—",
    name: "GNN Scout",
    family: "Compact message passing · 17.5k params",
    role: "bulk data generator",
    elo: "—",
    record: "not played",
    signal: "2.215 NLL",
    signalDetail: "fastest search",
    tone: "violet",
    glyph: "S",
    status: "Cross-play queued",
  },
  {
    id: "cnn-baseline-7x7",
    rank: "—",
    name: "CNN baseline",
    family: "7×7 residual CNN · 87.4k params",
    role: "parallel reference",
    elo: "—",
    record: "not played",
    signal: "2.291 NLL",
    signalDetail: "architecture reference",
    tone: "gold",
    glyph: "C",
    status: "Cross-play queued",
  },
];

const MATCHUPS = [
  { left: "The Pathfinder", right: "GNN Learner", detail: "priority lane · alternating colors" },
  { left: "The Surveyor", right: "GNN Learner", detail: "priority lane · alternating colors" },
  { left: "Lunatic", right: "GNN Learner", detail: "anchor check · alternating colors" },
  { left: "Coin Flip", right: "GNN Learner", detail: "floor check · alternating colors" },
  { left: "GNN Learner", right: "GNN Scout", detail: "candidate check · alternating colors" },
  { left: "GNN Learner", right: "CNN baseline", detail: "reference check · alternating colors" },
];

const FLOW = [
  { number: "01", label: "Train", detail: "A checkpoint clears its parent on held-out play.", state: "complete" },
  { number: "02", label: "Generate", detail: "The promoted model makes the next 7×7 batch.", state: "active" },
  { number: "03", label: "Cross-play", detail: "Every model meets every other model for Elo evidence.", state: "waiting" },
  { number: "04", label: "Promote", detail: "A small, repeatable improvement becomes the new baseline.", state: "waiting" },
];

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

type CrossPlayState = {
  runId: string;
  targetGames: number;
  games: number;
  status: "ready" | "running" | "complete";
  standings: LiveStanding[];
  latest: LiveGame[];
};

export default function LearningLab() {
  const [copied, setCopied] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [crossPlay, setCrossPlay] = useState<CrossPlayState | null>(null);
  const [crossPlayRunId, setCrossPlayRunId] = useState<string | null>(null);
  const [crossPlayBusy, setCrossPlayBusy] = useState(false);
  const [crossPlayError, setCrossPlayError] = useState<string | null>(null);

  const liveStandingById = useMemo(() => new Map((crossPlay?.standings ?? []).map((standing) => [standing.id, standing])), [crossPlay]);
  const liveRankById = useMemo(() => new Map((crossPlay?.standings ?? []).map((standing, index) => [standing.id, String(index + 1).padStart(2, "0")])), [crossPlay]);
  const strengthLeader = MODELS.find((model) => model.id === (crossPlay?.standings[0]?.id ?? "pathfinder-v0.3.0")) ?? MODELS[0];
  const strengthLeaderLive = liveStandingById.get(strengthLeader.id)?.games ? liveStandingById.get(strengthLeader.id) : undefined;
  const learnerLive = liveStandingById.get("gnn-warmstart-7x7");

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
    const savedRun = window.localStorage.getItem("pathagon-cross-play-run");
    const timer = window.setTimeout(() => {
      void readLatestCrossPlay()
        .then((snapshot) => {
          setCrossPlayRunId(snapshot.runId);
          setCrossPlay(snapshot);
        })
        .catch(() => {
          if (savedRun) setCrossPlayRunId(savedRun);
        });
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (!crossPlayRunId) return;
    let active = true;
    const refresh = () => {
      void readCrossPlay(crossPlayRunId)
        .then((snapshot) => { if (active) setCrossPlay(snapshot); })
        .catch((error: unknown) => { if (active) setCrossPlayError(error instanceof Error ? error.message : "Live run unavailable"); });
    };
    refresh();
    const timer = window.setInterval(refresh, 900);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [crossPlayRunId]);

  function toggleTheme() {
    const nextTheme = theme === "dark" ? "light" : "dark";
    setTheme(nextTheme);
    window.localStorage.setItem("pathagon-lab-theme", nextTheme);
  }

  async function copyBatchCommand() {
    try {
      await navigator.clipboard.writeText(BATCH_COMMAND);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2200);
    } catch {
      setCopied(false);
    }
  }

  async function startCrossPlay() {
    if (crossPlayBusy) return;
    const runId = crypto.randomUUID();
    const baseSeed = Math.floor(Math.random() * 2_000_000_000);
    window.localStorage.setItem("pathagon-cross-play-run", runId);
    setCrossPlayRunId(runId);
    setCrossPlay({ runId, targetGames: 10, games: 0, status: "ready", standings: [], latest: [] });
    setCrossPlayError(null);
    setCrossPlayBusy(true);
    try {
      for (let sequence = 0; sequence < TARGET_CROSS_PLAY_GAMES; sequence += 1) {
        const response = await fetch("/api/cross-play", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ runId, sequence, seed: baseSeed + sequence }),
        });
        const payload = await response.json() as { accepted?: boolean; error?: string };
        if (!response.ok || !payload.accepted) throw new Error(payload.error ?? "Cross-play game rejected");
        setCrossPlay(await readCrossPlay(runId));
      }
    } catch (error) {
      setCrossPlayError(error instanceof Error ? error.message : "Cross-play run stopped");
    } finally {
      setCrossPlayBusy(false);
    }
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
          <p>Compare model strength, live cross-play results, and promotion candidates.</p>
          <div className="portal-actions">
            <button className="portal-primary" type="button" onClick={copyBatchCommand}>
              {copied ? "Batch command copied" : "Copy next batch recipe"}<span>{copied ? "✓" : "↗"}</span>
            </button>
            <a className="portal-quiet-action" href="#standings">See current standings <span>↓</span></a>
          </div>
        </div>

        <div className="leaderboard-leader-card" aria-label="Current strength leader">
          <div className="leaderboard-card-topline"><span>Strength leader</span><span className="leaderboard-provisional">Archived Elo</span></div>
          <div className="leaderboard-leader-main">
            <ModelGlyph tone={strengthLeader.tone} glyph={strengthLeader.glyph} />
            <div className="leaderboard-leader-name"><strong>{strengthLeader.name}</strong><span>{strengthLeader.family} · playable opponent</span></div>
            <div className="leaderboard-leader-score"><strong>{strengthLeaderLive?.rating.toLocaleString() ?? strengthLeader.elo}</strong><small>{strengthLeaderLive ? "live Elo" : "provisional Elo"}</small></div>
          </div>
          <div className="leaderboard-signal-row"><span>{strengthLeaderLive ? `${formatRecord(strengthLeaderLive)} in this run` : `${strengthLeader.record} in the 7×7 archive`}</span><span>higher is better</span></div>
          <div className="leaderboard-signal-bar"><span /></div>
          <div className="leaderboard-card-footer"><span><i className="live-dot" /> {crossPlay?.games ? `${crossPlay.games} live games counted` : "playable strength anchor"}</span><small>candidates queue below</small></div>
        </div>
      </header>

      <section className="leaderboard-stat-grid" aria-label="Model league summary">
        <LeaderboardStat label="Agents tracked" value="7" detail="4 playable · 3 candidates" accent="green" />
        <LeaderboardStat label="Archived league" value="56" detail="7×7 · color-balanced games" accent="gold" />
        <LeaderboardStat label="Live cross-play" value={String(crossPlay?.games ?? 0)} detail={crossPlay ? `${crossPlay.games} of ${crossPlay.targetGames} games` : "10-game starter run"} accent="gold" />
        <LeaderboardStat label="Best held-out NLL" value="2.112" detail="GNN Learner · policy signal" accent="ink" />
      </section>

      <section className="leaderboard-panel cross-play-live-panel" aria-labelledby="live-run-title">
        <div className="cross-play-live-heading">
          <div><span className="portal-kicker">Live arena · random cross-play</span><h2 id="live-run-title">Make the ladder move.</h2><p>{isGnnBridgeRun(crossPlay?.runId) ? "The Python bridge streams each GNN Learner vs Surveyor result into the same live ladder." : "Ten seeded games are drawn from the four playable opponents. Each result is archived immediately and the standings below refresh while the run is playing."}</p></div>
          <button className="portal-primary live-run-button" type="button" onClick={startCrossPlay} disabled={crossPlayBusy}>{crossPlayBusy ? `Playing ${Math.min((crossPlay?.games ?? 0) + 1, TARGET_CROSS_PLAY_GAMES)} / ${TARGET_CROSS_PLAY_GAMES}…` : crossPlay?.status === "complete" ? "Play another 10" : "Play 10 random games"}<span>{crossPlayBusy ? "◌" : "↗"}</span></button>
        </div>
        <div className="live-run-summary">
          <div><strong>{crossPlay?.games ?? 0}<small> / {TARGET_CROSS_PLAY_GAMES}</small></strong><span>games complete</span></div>
          <div><strong>{crossPlay?.status === "complete" ? "Ready" : crossPlayBusy ? "Playing" : "Idle"}</strong><span>run status</span></div>
          <div><strong>{crossPlay?.latest[0]?.winner ?? "—"}</strong><span>latest winner</span></div>
        </div>
        {crossPlayError ? <p className="live-run-error" role="status">{crossPlayError}</p> : null}
        {crossPlay?.latest.length ? <div className="live-game-list" aria-label="Latest cross-play games">{crossPlay.latest.map((game) => <div className="live-game-row" key={game.id}><span className="live-game-number">#{game.seed % 1000}</span><strong>{game.light}</strong><span>vs</span><strong>{game.dark}</strong><span className="live-game-result">{game.winner ? `${game.winner} · ${game.plies} plies` : `draw · ${game.plies} plies`}</span></div>)}</div> : <p className="live-run-empty">No live games yet. Start the run and watch each pairing land here.</p>}
      </section>

      <section className="leaderboard-panel" id="standings" aria-labelledby="standings-title">
        <div className="leaderboard-panel-heading">
          <div><span className="portal-kicker">Standings · strength anchors + candidates</span><h2 id="standings-title">Every playable opponent is here.</h2></div>
          <span className="leaderboard-status"><span /> {learnerLive?.games ? "4 anchors · GNN live" : "4 anchors · 1 fresh · 2 queued"}</span>
        </div>
        <p className="leaderboard-intro">The four opponents available in the game are the strength anchors: Pathfinder, Surveyor, Lunatic, and Coin Flip. The new GNN Learner result is a fresh two-color head-to-head against Surveyor; the remaining neural candidates stay queued. NLL remains a separate training signal.</p>

        <div className="leaderboard-grid">
          <div className="leaderboard-table" role="table" aria-label="Current model standings">
            <div className="leaderboard-table-row leaderboard-table-header" role="row">
              <span>#</span><span>Agent</span><span>Role</span><span>Elo</span><span>Record</span><span>Signal</span>
            </div>
            {MODELS.map((model) => <ModelStanding key={model.name} model={model} live={liveStandingById.get(model.id)} liveRank={liveRankById.get(model.id)} />)}
          </div>

          <aside className="elo-card" aria-labelledby="elo-title">
            <div className="elo-card-heading"><div><span className="portal-kicker">Strength evidence</span><h3 id="elo-title">Elo ladder</h3></div><span className="portal-chip gold">provisional archive</span></div>
            <div className="elo-score"><strong>1,142</strong><span>Pathfinder · current high</span></div>
            <div className="elo-track"><span style={{ width: "100%" }} /></div>
            <div className="elo-meta"><strong>4 / 4</strong><span>playable opponents rated</span></div>
            <p>Self-play makes data. Head-to-head matches make a ranking. The archive confirms the selectable opponents’ current order; the neural candidates join the ladder only after balanced cross-play.</p>
            <a className="elo-link" href="#cross-play">View the cross-play gate <span>↓</span></a>
          </aside>
        </div>
      </section>

      <section className="leaderboard-panel leaderboard-lineage-panel" aria-labelledby="lineage-title">
        <div className="leaderboard-panel-heading"><div><span className="portal-kicker">Model lineage</span><h2 id="lineage-title">A small step is still a step.</h2></div><span className="portal-muted">promote slowly</span></div>
        <div className="leaderboard-lineage">
          <LineageNode label="Warm start" detail="replay baseline" status="archived" tone="muted" />
          <LineageConnector />
          <LineageNode label="GNN Scout" detail="17.5k params · generator" status="makes data" tone="violet" />
          <LineageConnector />
          <LineageNode label="GNN Learner" detail="100.2k params · candidate" status="current leader" tone="green" />
          <LineageConnector branch />
          <LineageNode label="CNN baseline" detail="87.4k params · reference" status="parallel check" tone="gold" />
        </div>
      </section>

      <section className="leaderboard-flow-panel" aria-labelledby="flow-title">
        <div className="leaderboard-flow-copy"><span className="portal-kicker">Promotion loop</span><h2 id="flow-title">Each generation earns the right to make the next one.</h2><p>We want a stable upward drift, not a dramatic leap followed by a collapse. The league is the guardrail around that loop.</p></div>
        <div className="leaderboard-flow-steps">{FLOW.map((step) => <FlowStep key={step.number} {...step} />)}</div>
      </section>

      <section className="leaderboard-evidence-grid" aria-label="Current model evidence">
        <div className="leaderboard-panel signal-panel">
          <div className="leaderboard-panel-heading"><div><span className="portal-kicker">Best current policy signal</span><h2>GNN Learner</h2></div><span className="portal-chip green">2.112 NLL</span></div>
          <p className="leaderboard-intro">The larger GNN is the current learner to test, not yet the league leader. Its first fresh head-to-head is now recorded below; placement remains its strongest policy signal, while relocation is the next gate.</p>
          <div className="signal-metrics"><SignalMetric label="Placement NLL" value="2.024" note="best of the three" /><SignalMetric label="Relocation NLL" value="2.248" note="phase to improve" /><SignalMetric label="Value MSE" value="0.609" note="effectively tied" /></div>
          <div className="fresh-match-callout"><span>Fresh offline match · 7×7 · 4 games</span><strong>Surveyor swept GNN Learner 4–0</strong><small>2 games per color · 12k-node Surveyor budget · 4 PUCT simulations</small></div>
        </div>

        <div className="leaderboard-panel batch-panel">
          <div className="leaderboard-panel-heading"><div><span className="portal-kicker">Latest generation</span><h2>3,000 games in the bank.</h2></div><span className="portal-chip green">seed clean</span></div>
          <div className="batch-summary"><div><strong>420,470</strong><span>positions</span></div><div><strong>0</strong><span>duplicate replays</span></div><div><strong>3,000</strong><span>unique seeds</span></div></div>
          <div className="batch-bar"><span className="batch-bar-scout" style={{ width: "33.33%" }} /><span className="batch-bar-learner" style={{ width: "33.33%" }} /><span className="batch-bar-cnn" style={{ width: "33.34%" }} /></div>
          <div className="batch-legend"><span><i className="legend-dot violet" /> Scout · 1k</span><span><i className="legend-dot green" /> Learner · 1k</span><span><i className="legend-dot gold" /> CNN · 1k</span></div>
        </div>
      </section>

      <section className="leaderboard-panel cross-play-panel" id="cross-play" aria-labelledby="cross-play-title">
        <div className="cross-play-heading"><div><span className="portal-kicker">Next gate · cross-play</span><h2 id="cross-play-title">Let the full field settle the order.</h2><p>Self-play archives stay separate. The next league run keeps all four playable opponents in the pool, swaps colors, and lets every candidate face every anchor before promotion.</p></div><span className="portal-chip gold">7 agents · 21 pairings</span></div>
        <div className="matchup-list">{MATCHUPS.map((matchup) => <div className="matchup-card" key={`${matchup.left}-${matchup.right}`}><div className="matchup-model"><ModelGlyph tone={glyphTone(matchup.left)} glyph={glyphFor(matchup.left)} /><strong>{matchup.left}</strong></div><span className="matchup-versus">vs</span><div className="matchup-model"><ModelGlyph tone={glyphTone(matchup.right)} glyph={glyphFor(matchup.right)} /><strong>{matchup.right}</strong></div><small>{matchup.detail}</small><span className="matchup-queued">queued</span></div>)}</div>
      </section>

      <footer className="portal-footer"><span>7×7 model leaderboard</span><span>Train a little. Play a lot. Promote slowly.</span></footer>
    </main>
  );
}

function ModelGlyph({ tone, glyph }: { tone: string; glyph: string }) {
  return <span className={`model-glyph ${tone}`} aria-hidden="true">{glyph}</span>;
}

function LeaderboardStat({ label, value, detail, accent }: { label: string; value: string; detail: string; accent: string }) {
  return <div className={`leaderboard-stat ${accent}`}><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}

function ModelStanding({ model, live, liveRank }: { model: (typeof MODELS)[number]; live?: LiveStanding; liveRank?: string }) {
  const liveActive = Boolean(live?.games);
  const rank = liveActive ? liveRank ?? model.rank : model.rank;
  const record = liveActive ? formatRecord(live!) : model.record;
  const signal = liveActive ? `${live!.games} live games` : model.signal;
  const signalDetail = liveActive ? `${live!.points.toFixed(1)} points · updates live` : model.signalDetail;
  return <div className={`leaderboard-table-row model-standing ${rank === "01" ? "leader" : ""}`} role="row"><span className="model-rank">{rank}</span><div className="standing-model"><ModelGlyph tone={model.tone} glyph={model.glyph} /><div><strong>{model.name}</strong><span>{model.family}</span></div></div><div className="standing-role"><strong>{liveActive ? "Live ladder" : model.status}</strong><span>{model.role}</span></div><span className="standing-elo">{liveActive ? live!.rating.toLocaleString() : model.elo}</span><div className="standing-record"><strong>{record}</strong><span>{liveActive ? "this run" : model.rank === "—" ? "awaiting games" : "archive record"}</span></div><div className="standing-signal"><strong>{signal}</strong><span>{signalDetail}</span></div></div>;
}

function glyphTone(name: string) {
  if (name.includes("Pathfinder") || name.includes("Learner")) return "green";
  if (name.includes("Surveyor") || name.includes("Scout")) return "violet";
  if (name.includes("Coin")) return "muted";
  return "gold";
}

function glyphFor(name: string) {
  if (name.includes("Pathfinder")) return "P";
  if (name.includes("Surveyor")) return "S";
  if (name.includes("Learner")) return "G";
  if (name.includes("Scout")) return "S";
  if (name.includes("CNN")) return "C";
  if (name.includes("Coin")) return "C";
  return "L";
}

function formatRecord(standing: Pick<LiveStanding, "wins" | "losses" | "draws">) {
  return `${standing.wins}–${standing.losses}–${standing.draws}`;
}

async function readCrossPlay(runId: string): Promise<CrossPlayState> {
  const response = await fetch(`/api/cross-play?runId=${encodeURIComponent(runId)}`, { cache: "no-store" });
  const payload = await response.json() as CrossPlayState & { error?: string };
  if (!response.ok) throw new Error(payload.error ?? "Live run unavailable");
  return payload;
}

async function readLatestCrossPlay(): Promise<CrossPlayState> {
  const response = await fetch("/api/cross-play?latest=1", { cache: "no-store" });
  const payload = await response.json() as CrossPlayState & { error?: string };
  if (!response.ok) throw new Error(payload.error ?? "No live run available");
  return payload;
}

function isGnnBridgeRun(runId?: string | null) {
  return Boolean(runId?.startsWith("gnn-surveyor-"));
}

function LineageNode({ label, detail, status, tone }: { label: string; detail: string; status: string; tone: string }) {
  return <div className={`lineage-node ${tone}`}><div className="lineage-node-top"><span className="lineage-pip" /><small>{status}</small></div><strong>{label}</strong><span>{detail}</span></div>;
}

function LineageConnector({ branch = false }: { branch?: boolean }) {
  return <span className={`lineage-connector ${branch ? "branch" : ""}`} aria-hidden="true"><i /></span>;
}

function FlowStep({ number, label, detail, state }: { number: string; label: string; detail: string; state: string }) {
  return <div className={`leaderboard-flow-step ${state}`}><span>{state === "complete" ? "✓" : number}</span><strong>{label}</strong><p>{detail}</p></div>;
}

function SignalMetric({ label, value, note }: { label: string; value: string; note: string }) {
  return <div className="signal-metric"><span>{label}</span><strong>{value}</strong><small>{note}</small></div>;
}
