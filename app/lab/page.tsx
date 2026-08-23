"use client";

import { useMemo, useState, type ReactNode } from "react";
import { RUNS, type RunRecord } from "./runs";

export default function LearningLab() {
  const [selectedId, setSelectedId] = useState(RUNS[0].id);
  const selected = useMemo(() => RUNS.find((run) => run.id === selectedId) ?? RUNS[0], [selectedId]);
  const checkpointCount = RUNS.length;
  const replayCount = RUNS.filter((run) => run.replay).length;
  const boardSizes = [...new Set(RUNS.map((run) => `${run.boardSize}×${run.boardSize}`))].join(" · ");
  const latestArena = RUNS.find((run) => run.arena);

  return (
    <main className="lab-app">
      <header className="lab-header">
        <div>
          <a className="lab-breadcrumb" href="/">Pathagon / Learning lab</a>
          <h1>Checkpoint shelf</h1>
          <p>Browse the model lineage, replay archives, and the small signals we have collected so far.</p>
        </div>
        <a className="lab-back" href="/">Back to game</a>
      </header>

      <section className="lab-summary" aria-label="Learning lab summary">
        <SummaryCard label="Checkpoints" value={checkpointCount} detail="saved model files" />
        <SummaryCard label="Replay archives" value={replayCount} detail="JSONL runs with positions" />
        <SummaryCard label="Board curriculum" value={boardSizes} detail="same dynamic network" />
        <SummaryCard
          label="Latest arena"
          value={latestArena ? `${latestArena.arena?.wins}–${latestArena.arena?.draws}–${latestArena.arena?.losses}` : "—"}
          detail="wins · draws · losses"
        />
      </section>

      <section className="lab-browser" aria-label="Checkpoint browser">
        <div className="run-list-panel">
          <div className="lab-panel-heading">
            <div>
              <span className="lab-kicker">Run browser</span>
              <h2>Lineage</h2>
            </div>
            <span className="run-count">{RUNS.length} runs</span>
          </div>
          <div className="run-list">
            {RUNS.map((run) => <RunListItem key={run.id} run={run} selected={run.id === selected.id} onSelect={() => setSelectedId(run.id)} />)}
          </div>
        </div>

        <RunDetail run={selected} />
      </section>

      <footer className="lab-footer">
        <span>v0 learning lab</span>
        <span>Loss curves are directional; arena results are the stronger signal.</span>
      </footer>
    </main>
  );
}

function SummaryCard({ label, value, detail }: { label: string; value: number | string; detail: string }) {
  return <div className="lab-summary-card"><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}

function RunListItem({ run, selected, onSelect }: { run: RunRecord; selected: boolean; onSelect: () => void }) {
  const outcome = run.outcomes;
  return (
    <button className={`run-list-item ${selected ? "selected" : ""}`} onClick={onSelect} type="button" aria-pressed={selected}>
      <span className="run-item-main">
        <span className="run-item-title">{run.label}</span>
        <span className="run-item-stage">{run.stage}</span>
      </span>
      <span className="run-item-side">
        <span className="board-chip">{run.boardSize}×{run.boardSize}</span>
        <span className="run-item-score">{outcome ? `${outcome.wins}W · ${outcome.draws}D` : `${run.metrics.examples.toLocaleString()} pos`}</span>
      </span>
    </button>
  );
}

function RunDetail({ run }: { run: RunRecord }) {
  const outcome = run.outcomes;
  const arena = run.arena;
  const totalOutcomes = outcome ? outcome.wins + outcome.draws + outcome.losses : 0;
  const maxOutcome = Math.max(outcome?.wins ?? 0, outcome?.draws ?? 0, outcome?.losses ?? 0, 1);

  return (
    <div className="run-detail-panel">
      <div className="detail-heading">
        <div>
          <div className="detail-kicker"><span className={`kind-dot ${run.kind}`} /> {run.kind === "warmstart" ? "Initialization" : "AlphaZero-style generation"}</div>
          <h2>{run.label}</h2>
          <p>{run.summary}</p>
        </div>
        <span className="detail-board">{run.boardSize}×{run.boardSize} board</span>
      </div>

      <div className="lineage-strip">
        <span>Parent</span><strong>{run.parent ?? "—"}</strong><span className="lineage-arrow">→</span><strong>{run.label}</strong>
      </div>

      <div className="file-stack">
        <FileRow label="Checkpoint" name={run.checkpoint.name} bytes={run.checkpoint.bytes} detail="PyTorch weights + metadata" />
        {run.replay ? <FileRow label="Replay archive" name={run.replay.name} bytes={run.replay.bytes} detail={`${run.replay.games} games · ${run.replay.positions.toLocaleString()} positions`} /> : <div className="file-row missing"><div><span className="file-label">Replay archive</span><strong>No separate JSONL saved</strong><small>Training examples are recorded in checkpoint metadata.</small></div><span className="file-state">legacy</span></div>}
      </div>

      <div className="detail-grid">
        <MetricGroup title="Training signal">
          <Metric label="Positions" value={run.metrics.examples.toLocaleString()} />
          <Metric label="Average plies" value={run.metrics.averagePlies === null ? "—" : run.metrics.averagePlies.toFixed(1)} />
          <Metric label="Policy loss" value={formatMetric(run.metrics.policyLoss)} />
          <Metric label="Value loss" value={formatMetric(run.metrics.valueLoss)} />
        </MetricGroup>
        <MetricGroup title="Self-play outcome">
          {outcome ? <OutcomeBars outcome={outcome} max={maxOutcome} total={totalOutcomes} /> : <p className="empty-state">No replay result breakdown is attached to this run yet.</p>}
        </MetricGroup>
      </div>

      <section className="arena-card">
        <div className="arena-heading"><div><span className="lab-kicker">Independent check</span><h3>Random-baseline arena</h3></div>{arena && <span className="arena-pill">{arena.boardSize}×{arena.boardSize} · {arena.simulations} sims</span>}</div>
        {arena ? <div className="arena-body"><div className="arena-score"><strong>{arena.wins}–{arena.draws}–{arena.losses}</strong><span>{arena.games} games · wins · draws · losses</span></div><div className="arena-track" aria-label={`${arena.wins} wins, ${arena.draws} draws, ${arena.losses} losses`}><span className="arena-win" style={{ width: `${(arena.wins / arena.games) * 100}%` }} /><span className="arena-draw" style={{ width: `${(arena.draws / arena.games) * 100}%` }} /><span className="arena-loss" style={{ width: `${(arena.losses / arena.games) * 100}%` }} /></div></div> : <p className="empty-state">No independent arena result recorded for this checkpoint.</p>}
      </section>
    </div>
  );
}

function FileRow({ label, name, bytes, detail }: { label: string; name: string; bytes: number; detail: string }) {
  return <div className="file-row"><div><span className="file-label">{label}</span><strong>{name}</strong><small>{detail}</small></div><span className="file-size">{formatBytes(bytes)}</span></div>;
}

function MetricGroup({ title, children }: { title: string; children: ReactNode }) {
  return <section className="metric-group"><span className="lab-kicker">{title}</span><div className="metric-list">{children}</div></section>;
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

function OutcomeBars({ outcome, max, total }: { outcome: NonNullable<RunRecord["outcomes"]>; max: number; total: number }) {
  const rows = [["Wins", outcome.wins, "win"], ["Draws", outcome.draws, "draw"], ["Losses", outcome.losses, "loss"]] as const;
  return <div className="outcome-list">{rows.map(([label, value, tone]) => <div className="outcome-row" key={label}><div><span>{label}</span><strong>{value}</strong></div><div className="outcome-bar"><span className={`outcome-fill ${tone}`} style={{ width: `${(value / max) * 100}%` }} /></div><small>{total ? Math.round((value / total) * 100) : 0}%</small></div>)}</div>;
}

function formatBytes(bytes: number) {
  return bytes >= 1024 * 1024 ? `${(bytes / (1024 * 1024)).toFixed(1)} MB` : `${Math.round(bytes / 1024)} KB`;
}

function formatMetric(value: number | null) {
  return value === null ? "—" : value.toFixed(3);
}
