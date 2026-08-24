"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

const GENERATION_COMMAND = `./.venv-pathagon-gnn/bin/python -m learning.gnn.train alphazero \\
  --resume training/gnn/benchmark-7x7/small-gnn-warmstart.pt \\
  --games 1000 --workers 8 --simulations 4 \\
  --temperature-moves 32 --updates 1`;

const BOARD_PIECES: Record<number, "light" | "dark"> = {
  3: "dark",
  9: "light",
  14: "dark",
  19: "light",
  24: "dark",
  28: "light",
  34: "dark",
  39: "light",
  44: "dark",
};

const ACTIVITY = [
  { label: "Scout checkpoint trained", detail: "17.5k params · 2.215 held-out policy NLL", tone: "green" },
  { label: "Held-out gate passed", detail: "416 games · 25,751 positions · 0 seed overlap", tone: "gold" },
  { label: "Canonical corpus locked", detail: "2,037 unique 7×7 games · 123,691 positions", tone: "ink" },
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

  async function copyGenerationCommand() {
    try {
      await navigator.clipboard.writeText(GENERATION_COMMAND);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2200);
    } catch {
      setCopied(false);
    }
  }

  return (
    <main className={`portal-app ${theme === "dark" ? "dark" : ""}`}>
      <nav className="portal-nav" aria-label="Lab navigation">
        <Link className="portal-breadcrumb" href="/">
          <span className="portal-mark">P</span>
          <span>Pathagon</span>
          <span className="portal-slash">/</span>
          <span>Learning lab</span>
        </Link>
        <div className="portal-nav-right">
          <span className="portal-live"><span /> 7×7 target locked</span>
          <button className="portal-theme-toggle" type="button" onClick={toggleTheme} aria-pressed={theme === "dark"} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}>
            <span aria-hidden="true">{theme === "dark" ? "☼" : "☾"}</span>
            {theme === "dark" ? "Light" : "Dark"}
          </button>
          <Link className="portal-game-link" href="/">Back to game <span>↗</span></Link>
        </div>
      </nav>

      <header className="portal-hero">
        <div className="portal-hero-copy">
          <span className="portal-kicker">Diverse game generation · control room</span>
          <h1>Make the dataset<br /><em>harder to fool.</em></h1>
          <p>One board size. Many ways to play it. This is the progress portal for building a deep, diverse 7×7 corpus that can keep improving the learner.</p>
          <div className="portal-actions">
            <button className="portal-primary" type="button" onClick={copyGenerationCommand}>
              {copied ? "Command copied" : "Copy next run command"}<span>{copied ? "✓" : "↗"}</span>
            </button>
            <a className="portal-quiet-action" href="#proof">Inspect the proof archive <span>↓</span></a>
          </div>
        </div>
        <div className="portal-board-card" aria-label="Illustration of the 7 by 7 generation field">
          <div className="portal-board-topline"><span>SCOUT / FIELD 01</span><span>7 × 7</span></div>
          <PortalBoard />
          <div className="portal-board-caption"><span>Current exploration field</span><strong>root noise + τ 32</strong></div>
        </div>
      </header>

      <section className="portal-stat-grid" aria-label="7x7 corpus summary">
        <PortalStat label="Canonical games" value="2,037" detail="unique records in the 7×7 set" accent="green" />
        <PortalStat label="Position bank" value="123,691" detail="states available for training" accent="ink" />
        <PortalStat label="Held-out gate" value="416" detail="games kept outside the train split" accent="gold" />
        <PortalStat label="Scout footprint" value="17.5k" detail="parameters in the fast generator" accent="violet" />
      </section>

      <section className="portal-panel portal-progress-panel" aria-labelledby="progress-title">
        <div className="portal-panel-heading">
          <div><span className="portal-kicker">Corpus mission</span><h2 id="progress-title">Fill the 10k game target</h2></div>
          <span className="portal-status ready"><span /> Ready to launch</span>
        </div>
        <p className="portal-panel-intro">The archive is clean enough to grow now. The next batch is designed to add movement-phase variety instead of simply repeating the opening distribution.</p>
        <div className="portal-progress-track"><span style={{ width: "20.37%" }} /></div>
        <div className="portal-progress-meta"><strong>2,037 <small>/ 10,000 games</small></strong><span>20.4% of target corpus</span></div>
        <div className="portal-pipeline" aria-label="Generation pipeline">
          <PipelineStep number="01" label="Archive" detail="2,037 games" state="complete" />
          <PipelineStep number="02" label="Deduplicate" detail="0 seed overlap" state="complete" />
          <PipelineStep number="03" label="Scout batch" detail="1,000 games queued" state="active" />
          <PipelineStep number="04" label="Retrain learner" detail="After quality gate" state="waiting" />
        </div>
      </section>

      <section className="portal-two-column">
        <div className="portal-panel portal-recipe-panel">
          <div className="portal-panel-heading"><div><span className="portal-kicker">Next run</span><h2>The Scout recipe</h2></div><span className="portal-chip green">Fast + diverse</span></div>
          <p className="portal-panel-intro">A compact GNN plays the field cheaply, while search noise and a long temperature window keep the corpus from collapsing into one style.</p>
          <div className="portal-recipe-grid">
            <RecipeItem label="Player" value="Compact GNN" detail="32 channels · 4 layers" />
            <RecipeItem label="Search" value="4 simulations" detail="per move" />
            <RecipeItem label="Exploration" value="τ 32" detail="temperature window" />
            <RecipeItem label="Workers" value="8 CPU lanes" detail="parallel games" />
          </div>
          <div className="portal-command"><code>{GENERATION_COMMAND}</code><button type="button" onClick={copyGenerationCommand} aria-label="Copy generation command">{copied ? "✓" : "⧉"}</button></div>
        </div>

        <div className="portal-panel portal-health-panel">
          <div className="portal-panel-heading"><div><span className="portal-kicker">Quality gates</span><h2>Dataset health</h2></div><span className="portal-chip gold">Held-out v1</span></div>
          <HealthRow label="Split integrity" value="0 overlap" detail="seed groups stay disjoint" meter={100} tone="green" />
          <HealthRow label="Phase coverage" value="61 / 39" detail="placement / relocation in held out" meter={61} tone="gold" />
          <HealthRow label="Draw pressure" value="High" detail="more variety needed beyond the cap" meter={74} tone="violet" />
          <HealthRow label="Board commitment" value="7×7 locked" detail="14 pieces per player" meter={100} tone="ink" />
        </div>
      </section>

      <section className="portal-two-column portal-lower-grid">
        <div className="portal-panel">
          <div className="portal-panel-heading"><div><span className="portal-kicker">Model bench</span><h2>Who is playing?</h2></div><span className="portal-muted">same held-out set</span></div>
          <div className="portal-model-list">
            <ModelRow name="Scout" tag="generator" detail="Compact GNN · 17.5k params" score="2.215" outcome="7–5–28" active />
            <ModelRow name="Learner" tag="target" detail="GNN · 100.2k params" score="2.112" outcome="5–6–29" />
            <ModelRow name="CNN" tag="baseline" detail="7×7 CNN · 87.4k params" score="2.291" outcome="8–5–27" />
          </div>
          <div className="portal-model-legend"><span><i className="legend-dot green" /> lower NLL is better</span><span><i className="legend-dot gold" /> arena W–L–D smoke check</span></div>
        </div>

        <div className="portal-panel portal-activity-panel">
          <div className="portal-panel-heading"><div><span className="portal-kicker">Run log</span><h2>What changed</h2></div><span className="portal-muted">latest work</span></div>
          <div className="portal-activity-list">{ACTIVITY.map((item) => <div className="portal-activity-item" key={item.label}><span className={`activity-dot ${item.tone}`} /><div><strong>{item.label}</strong><span>{item.detail}</span></div><small>done</small></div>)}</div>
        </div>
      </section>

      <section className="portal-proof" id="proof" aria-labelledby="proof-title">
        <div className="portal-proof-copy"><span className="portal-kicker">Proof of life</span><h2 id="proof-title">The archive is playable.</h2><p>Before the next flood of games, inspect the existing 7×7 material that anchors the experiment. It is the baseline the Scout needs to broaden—not erase.</p><div className="portal-proof-actions"><a className="portal-primary small" href="/lab/replays/selfplay-generation-8-7x7.jsonl" download>Download replay JSONL <span>↓</span></a><span className="portal-proof-note">10 games · 1,883 positions</span></div></div>
        <div className="portal-proof-board"><div className="portal-proof-board-label"><span>ARCHIVE SAMPLE</span><strong>generation 8</strong></div><PortalBoard compact /></div>
      </section>

      <footer className="portal-footer"><span>7×7 diverse game generation</span><span>Scout first. Measure everything. Promote slowly.</span></footer>
    </main>
  );
}

function PortalBoard({ compact = false }: { compact?: boolean }) {
  return <div className={`portal-board ${compact ? "compact" : ""}`}>{Array.from({ length: 49 }, (_, index) => <span className={`portal-board-cell ${index % 8 === 0 ? "route" : ""}`} key={index}>{BOARD_PIECES[index] && <i className={`portal-board-piece ${BOARD_PIECES[index]}`} />}</span>)}</div>;
}

function PortalStat({ label, value, detail, accent }: { label: string; value: string; detail: string; accent: string }) {
  return <div className={`portal-stat ${accent}`}><span className="portal-stat-label">{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}

function PipelineStep({ number, label, detail, state }: { number: string; label: string; detail: string; state: "complete" | "active" | "waiting" }) {
  return <div className={`portal-pipeline-step ${state}`}><span className="pipeline-number">{state === "complete" ? "✓" : number}</span><div><strong>{label}</strong><small>{detail}</small></div></div>;
}

function RecipeItem({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <div className="portal-recipe-item"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}

function HealthRow({ label, value, detail, meter, tone }: { label: string; value: string; detail: string; meter: number; tone: string }) {
  return <div className="portal-health-row"><div className="portal-health-copy"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div><div className="portal-health-meter"><span className={tone} style={{ width: `${meter}%` }} /></div></div>;
}

function ModelRow({ name, tag, detail, score, outcome, active = false }: { name: string; tag: string; detail: string; score: string; outcome: string; active?: boolean }) {
  return <div className={`portal-model-row ${active ? "active" : ""}`}><span className="model-status" /><div className="portal-model-name"><strong>{name}</strong><span>{tag}</span><small>{detail}</small></div><div className="portal-model-metric"><small>NLL</small><strong>{score}</strong></div><div className="portal-model-metric arena"><small>arena</small><strong>{outcome}</strong></div></div>;
}
