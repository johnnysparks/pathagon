"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

const BATCH_COMMAND = `./.venv-pathagon-gnn/bin/python scripts/generate-7x7-selfplay.py \\
  --games-per-player 1000 --players scout,learner,cnn \\
  --workers 8 --simulations 4 --temperature-moves 32 \\
  --max-plies 196 \\
  --output-dir training/gnn/benchmark-7x7/generated/<batch-id>`;

const MODELS = [
  {
    rank: "01",
    name: "GNN Learner",
    family: "Residual message passing",
    role: "current candidate",
    params: "100.2k",
    nll: "2.112",
    delta: "−0.103 vs Scout",
    games: "1,000",
    positions: "129,790",
    tone: "green",
    glyph: "G",
    status: "Promote next",
  },
  {
    rank: "02",
    name: "GNN Scout",
    family: "Compact message passing",
    role: "bulk data generator",
    params: "17.5k",
    nll: "2.215",
    delta: "fastest search",
    games: "1,000",
    positions: "142,760",
    tone: "violet",
    glyph: "S",
    status: "Generating",
  },
  {
    rank: "03",
    name: "CNN baseline",
    family: "7×7 residual CNN",
    role: "parallel reference",
    params: "87.4k",
    nll: "2.291",
    delta: "−0.079 vs Scout",
    games: "1,000",
    positions: "147,920",
    tone: "gold",
    glyph: "C",
    status: "Reference",
  },
];

const MATCHUPS = [
  { left: "GNN Learner", right: "GNN Scout", detail: "100 games · alternating colors" },
  { left: "GNN Learner", right: "CNN baseline", detail: "100 games · alternating colors" },
  { left: "GNN Scout", right: "CNN baseline", detail: "100 games · alternating colors" },
];

const FLOW = [
  { number: "01", label: "Train", detail: "A checkpoint clears its parent on held-out play.", state: "complete" },
  { number: "02", label: "Generate", detail: "The promoted model makes the next 7×7 batch.", state: "active" },
  { number: "03", label: "Cross-play", detail: "Every model meets every other model for Elo evidence.", state: "waiting" },
  { number: "04", label: "Promote", detail: "A small, repeatable improvement becomes the new baseline.", state: "waiting" },
];

export default function LearningLab() {
  const [copied, setCopied] = useState(false);
  const [theme, setTheme] = useState<"light" | "dark">("light");

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
          <span className="portal-kicker">Model progression · 7×7 league</span>
          <h1>A quiet ratchet<br /><em>for stronger play.</em></h1>
          <p>Every new checkpoint should be a little better than the one before it. The winner makes the next games; the games teach the next model; cross-play keeps the ranking honest.</p>
          <div className="portal-actions">
            <button className="portal-primary" type="button" onClick={copyBatchCommand}>
              {copied ? "Batch command copied" : "Copy next batch recipe"}<span>{copied ? "✓" : "↗"}</span>
            </button>
            <a className="portal-quiet-action" href="#standings">See current standings <span>↓</span></a>
          </div>
        </div>

        <div className="leaderboard-leader-card" aria-label="Current model leader">
          <div className="leaderboard-card-topline"><span>Current leader</span><span className="leaderboard-provisional">Policy signal</span></div>
          <div className="leaderboard-leader-main">
            <ModelGlyph tone="green" glyph="G" />
            <div className="leaderboard-leader-name"><strong>GNN Learner</strong><span>64 channels · 8 message layers</span></div>
            <div className="leaderboard-leader-score"><strong>2.112</strong><small>held-out NLL</small></div>
          </div>
          <div className="leaderboard-signal-row"><span>−0.103 vs compact Scout</span><span>lower is better</span></div>
          <div className="leaderboard-signal-bar"><span /></div>
          <div className="leaderboard-card-footer"><span><i className="live-dot" /> candidate for promotion</span><small>Elo pending cross-play</small></div>
        </div>
      </header>

      <section className="leaderboard-stat-grid" aria-label="Model league summary">
        <LeaderboardStat label="Models in league" value="3" detail="two GNNs · one CNN" accent="green" />
        <LeaderboardStat label="Fresh games" value="3,000" detail="1,000 from each player" accent="gold" />
        <LeaderboardStat label="Positions generated" value="420,470" detail="seed-clean 7×7 replay" accent="violet" />
        <LeaderboardStat label="Best held-out NLL" value="2.112" detail="GNN Learner · current signal" accent="ink" />
      </section>

      <section className="leaderboard-panel" id="standings" aria-labelledby="standings-title">
        <div className="leaderboard-panel-heading">
          <div><span className="portal-kicker">Standings · signal quality</span><h2 id="standings-title">Models first. Elo next.</h2></div>
          <span className="leaderboard-status"><span /> provisional ranking</span>
        </div>
        <p className="leaderboard-intro">The current order is based on held-out policy quality, not a claimed strength rating. Once the league has balanced cross-play, this table can promote a model because it is both better on paper and better across the board.</p>

        <div className="leaderboard-grid">
          <div className="leaderboard-table" role="table" aria-label="Current model standings">
            <div className="leaderboard-table-row leaderboard-table-header" role="row">
              <span>#</span><span>Model</span><span>Role</span><span>Params</span><span>NLL</span><span>New games</span>
            </div>
            {MODELS.map((model) => <ModelStanding key={model.name} model={model} />)}
          </div>

          <aside className="elo-card" aria-labelledby="elo-title">
            <div className="elo-card-heading"><div><span className="portal-kicker">Strength evidence</span><h3 id="elo-title">Elo ladder</h3></div><span className="portal-chip gold">not locked</span></div>
            <div className="elo-score"><strong>—</strong><span>provisional until cross-play</span></div>
            <div className="elo-track"><span style={{ width: "0%" }} /></div>
            <div className="elo-meta"><strong>0 / 3</strong><span>pairings complete</span></div>
            <p>Self-play makes data. Head-to-head matches make a ranking. Run every pairing with alternating colors, then update ratings after each accepted generation.</p>
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
          <div className="leaderboard-panel-heading"><div><span className="portal-kicker">Best current signal</span><h2>GNN Learner</h2></div><span className="portal-chip green">2.112 NLL</span></div>
          <p className="leaderboard-intro">The larger GNN is the current learner to beat. Its advantage is clearest in placement, while relocation remains a useful second gate.</p>
          <div className="signal-metrics"><SignalMetric label="Placement NLL" value="2.024" note="best of the three" /><SignalMetric label="Relocation NLL" value="2.248" note="phase to improve" /><SignalMetric label="Value MSE" value="0.609" note="effectively tied" /></div>
        </div>

        <div className="leaderboard-panel batch-panel">
          <div className="leaderboard-panel-heading"><div><span className="portal-kicker">Latest generation</span><h2>3,000 games in the bank.</h2></div><span className="portal-chip green">seed clean</span></div>
          <div className="batch-summary"><div><strong>420,470</strong><span>positions</span></div><div><strong>0</strong><span>duplicate replays</span></div><div><strong>3,000</strong><span>unique seeds</span></div></div>
          <div className="batch-bar"><span className="batch-bar-scout" style={{ width: "33.33%" }} /><span className="batch-bar-learner" style={{ width: "33.33%" }} /><span className="batch-bar-cnn" style={{ width: "33.34%" }} /></div>
          <div className="batch-legend"><span><i className="legend-dot violet" /> Scout · 1k</span><span><i className="legend-dot green" /> Learner · 1k</span><span><i className="legend-dot gold" /> CNN · 1k</span></div>
        </div>
      </section>

      <section className="leaderboard-panel cross-play-panel" id="cross-play" aria-labelledby="cross-play-title">
        <div className="cross-play-heading"><div><span className="portal-kicker">Next gate · cross-play</span><h2 id="cross-play-title">Let the models settle the order.</h2><p>Self-play archives stay separate. The next league run lets every model face every other model, swaps colors, and turns those results into a real Elo baseline.</p></div><span className="portal-chip gold">6 color-balanced legs</span></div>
        <div className="matchup-list">{MATCHUPS.map((matchup) => <div className="matchup-card" key={`${matchup.left}-${matchup.right}`}><div className="matchup-model"><ModelGlyph tone={matchup.left.includes("Learner") ? "green" : matchup.left.includes("Scout") ? "violet" : "gold"} glyph={matchup.left.charAt(4)} /><strong>{matchup.left}</strong></div><span className="matchup-versus">vs</span><div className="matchup-model"><ModelGlyph tone={matchup.right.includes("Learner") ? "green" : matchup.right.includes("Scout") ? "violet" : "gold"} glyph={matchup.right.charAt(0)} /><strong>{matchup.right}</strong></div><small>{matchup.detail}</small><span className="matchup-queued">queued</span></div>)}</div>
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

function ModelStanding({ model }: { model: (typeof MODELS)[number] }) {
  return <div className={`leaderboard-table-row model-standing ${model.rank === "01" ? "leader" : ""}`} role="row"><span className="model-rank">{model.rank}</span><div className="standing-model"><ModelGlyph tone={model.tone} glyph={model.glyph} /><div><strong>{model.name}</strong><span>{model.family}</span></div></div><div className="standing-role"><strong>{model.status}</strong><span>{model.role}</span></div><span className="standing-value">{model.params}</span><div className="standing-nll"><strong>{model.nll}</strong><span>{model.delta}</span></div><div className="standing-games"><strong>{model.games}</strong><span>{model.positions} pos.</span></div></div>;
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
