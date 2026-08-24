"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import { LEAGUES, type LeagueArchive, type LeagueMatch, type LeagueStanding } from "./leagues";
import { RUNS, type RunRecord } from "./runs";
import { buildReplayPositions, coordinate, formatReplayAction, parseReplayArchive, type ReplayGame } from "./replay";

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

      <LeagueBrowser />

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

function LeagueBrowser() {
  const [selectedId, setSelectedId] = useState(LEAGUES[0].id);
  const [archive, setArchive] = useState<LeagueArchive | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const selected = LEAGUES.find((league) => league.id === selectedId) ?? LEAGUES[0];

  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    fetch(`/lab/leagues/${selected.archive}`)
      .then((response) => {
        if (!response.ok) throw new Error("League archive unavailable");
        return response.json() as Promise<LeagueArchive>;
      })
      .then((data) => {
        if (cancelled) return;
        setArchive(data);
        setStatus("ready");
      })
      .catch(() => {
        if (cancelled) return;
        setArchive(null);
        setStatus("error");
      });
    return () => { cancelled = true; };
  }, [selected.archive]);

  return (
    <section className="league-card" aria-label="Agent league standings">
      <div className="league-heading">
        <div><span className="lab-kicker">Agent league</span><h2>Head-to-head standings</h2><p>{selected.note}</p></div>
        <a className="league-download" href={`/lab/leagues/${selected.archive}`} download>Download JSON</a>
      </div>
      <div className="league-tabs" role="tablist" aria-label="League board sizes">
        {LEAGUES.map((league) => <button key={league.id} className={`league-tab ${league.id === selected.id ? "selected" : ""}`} onClick={() => setSelectedId(league.id)} type="button" role="tab" aria-selected={league.id === selected.id}>{league.label}</button>)}
      </div>
      {status === "loading" && <p className="empty-state">Loading league archive…</p>}
      {status === "error" && <p className="empty-state">This league archive has not been published yet.</p>}
      {status === "ready" && archive && <LeagueResults archive={archive} />}
    </section>
  );
}

function LeagueResults({ archive }: { archive: LeagueArchive }) {
  const labels = new Map(archive.standings.map((standing) => [standing.id, standing.label]));
  return (
    <div className="league-results">
      <div className="league-meta"><span>{archive.boardSize}×{archive.boardSize} · {archive.reservePerPlayer} pieces/player</span><span>{archive.gamesPerMatch} games/match · {archive.simulations} MCTS sims</span><span>{archive.standings.reduce((total, standing) => total + standing.games, 0) / 2} games total</span></div>
      <div className="league-grid">
        <div className="standings-table" role="table" aria-label="League Elo standings">
          <div className="standings-row standings-header" role="row"><span>#</span><span>Agent</span><span>Elo</span><span>W–D–L</span><span>Pts</span></div>
          {archive.standings.map((standing, index) => <StandingRow key={standing.id} standing={standing} rank={index + 1} />)}
        </div>
        <div className="matchup-table" role="table" aria-label="Head-to-head results">
          <div className="matchup-heading"><span className="lab-kicker">Matchups</span><span>{archive.headToHead.length}</span></div>
          {archive.headToHead.map((match) => <MatchupRow key={`${match.left}-${match.right}`} match={match} labels={labels} />)}
        </div>
      </div>
    </div>
  );
}

function StandingRow({ standing, rank }: { standing: LeagueStanding; rank: number }) {
  return <div className="standings-row" role="row"><span className="standing-rank">{rank}</span><span className="standing-agent"><strong>{standing.label}</strong><small>{standing.kind}</small></span><strong className="standing-rating">{standing.rating}</strong><span>{standing.wins}–{standing.draws}–{standing.losses}</span><span>{standing.points.toFixed(1)}</span></div>;
}

function MatchupRow({ match, labels }: { match: LeagueMatch; labels: Map<string, string> }) {
  const left = match.leftSummary;
  const right = match.rightSummary;
  return <div className="matchup-row"><div><strong>{labels.get(match.left) ?? match.left}</strong><span>{labels.get(match.right) ?? match.right}</span></div><strong className="matchup-score">{left.wins}–{left.draws}–{left.losses}</strong><span className="matchup-vs">vs</span><strong className="matchup-score">{right.wins}–{right.draws}–{right.losses}</strong></div>;
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

      <ReplayViewer run={run} />
    </div>
  );
}

function ReplayViewer({ run }: { run: RunRecord }) {
  const [games, setGames] = useState<ReplayGame[]>([]);
  const [gameIndex, setGameIndex] = useState(0);
  const [ply, setPly] = useState(0);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [playing, setPlaying] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const game = games[gameIndex] ?? null;
  const positions = useMemo(() => game ? buildReplayPositions(game) : [], [game]);
  const position = positions[ply] ?? positions[0] ?? null;

  useEffect(() => {
    setGames([]);
    setGameIndex(0);
    setPly(0);
    setPlaying(false);
    setError(null);
    if (!run.replay) {
      setStatus("idle");
      return;
    }
    let cancelled = false;
    setStatus("loading");
    fetch(`/lab/replays/${run.replay.name}`)
      .then((response) => {
        if (!response.ok) throw new Error("Replay archive unavailable");
        return response.text();
      })
      .then((text) => {
        if (cancelled) return;
        setGames(parseReplayArchive(text, run.boardSize, run.reservePerPlayer));
        setStatus("ready");
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        setError(cause instanceof Error ? cause.message : "Replay archive unavailable");
        setStatus("error");
      });
    return () => { cancelled = true; };
  }, [run.boardSize, run.reservePerPlayer, run.replay, run.replay?.name]);

  useEffect(() => {
    if (!playing || !positions.length) return;
    const timer = setInterval(() => {
      setPly((current) => {
        if (current >= positions.length - 1) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 520);
    return () => clearInterval(timer);
  }, [playing, positions.length]);

  function chooseGame(index: number) {
    setGameIndex(index);
    setPly(0);
    setPlaying(false);
  }

  if (!run.replay) return <section className="replay-card"><ReplayHeading /><p className="empty-state">This legacy checkpoint has no separately saved replay archive.</p></section>;
  if (status === "loading") return <section className="replay-card"><ReplayHeading /><p className="empty-state">Loading replay archive…</p></section>;
  if (status === "error") return <section className="replay-card"><ReplayHeading /><p className="empty-state">{error ?? "Replay archive unavailable."}</p></section>;
  if (!game || !position) return <section className="replay-card"><ReplayHeading /><p className="empty-state">The replay archive contains no games.</p></section>;

  const move = ply > 0 ? game.moves[ply - 1] : null;
  const resultLabel = game.winner ? `${capitalize(game.winner)} path` : "Draw";
  return (
    <section className="replay-card" aria-label="Replay archive viewer">
      <ReplayHeading gameCount={games.length} />
      <div className="replay-layout">
        <div className="replay-games">
          <div className="replay-games-heading"><span className="lab-kicker">Games</span><span>{games.length}</span></div>
          <div className="replay-game-list">
            {games.map((candidate, index) => (
              <button className={`replay-game-item ${index === gameIndex ? "selected" : ""}`} key={`${candidate.seed}-${index}`} onClick={() => chooseGame(index)} type="button" aria-pressed={index === gameIndex}>
                <span><strong>Game {index + 1}</strong><small>seed {candidate.seed}</small></span>
                <span className={`replay-result ${candidate.winner ?? "draw"}`}>{candidate.winner ? `${capitalize(candidate.winner)} win` : "Draw"}<small>{candidate.plies} ply</small></span>
              </button>
            ))}
          </div>
        </div>
        <div className="replay-stage">
          <div className="replay-stage-heading">
            <div><span className="lab-kicker">Game {gameIndex + 1} · {game.reason.replaceAll("-", " ")}</span><h3>{resultLabel}</h3></div>
            <span className="replay-position-count">{ply} / {game.plies}</span>
          </div>
          <div className="replay-board-frame">
            <div className="replay-board" role="grid" aria-label={`${game.boardSize} by ${game.boardSize} replay board`} style={{ gridTemplateColumns: `repeat(${game.boardSize}, 1fr)` }}>
              {position.board.map((piece, index) => {
                const isWinning = position.winningPath.includes(index);
                const isLastMove = Boolean(move && (move.action.to === index || (move.action.kind === "relocate" && move.action.from === index)));
                const isForbidden = position.forbidden.includes(index);
                return <div className={`replay-cell ${piece ?? "empty"} ${isWinning ? "winning" : ""} ${isLastMove ? "last" : ""}`} key={index} role="gridcell" aria-label={`${coordinate(index, game.boardSize)} ${piece ? `${piece} piece` : "empty"}`}><span className="replay-socket" />{piece && <span className={`replay-piece ${piece}`} />}{isForbidden && <span className="replay-forbidden">×</span>}</div>;
              })}
            </div>
          </div>
          <div className="replay-move-summary">
            <span className="replay-turn-dot" data-player={position.lastMove?.player ?? position.turn} />
            <div><strong>{move ? `${capitalize(move.player)} · ${formatReplayAction(move.action, game.boardSize)}` : "Opening position"}</strong><small>{move ? `${move.captured.length ? `Captured ${move.captured.length} · ` : ""}${position.reserve.light} light / ${position.reserve.dark} dark in hand` : `${position.reserve.light} pieces in each reserve`}</small></div>
          </div>
          <div className="replay-controls">
            <button className="replay-control" onClick={() => setPly(0)} disabled={ply === 0} type="button" aria-label="Reset replay">↤</button>
            <button className="replay-control replay-play" onClick={() => setPlaying((value) => !value)} type="button">{playing ? "Pause" : "Play"}</button>
            <button className="replay-control" onClick={() => setPly((value) => Math.min(value + 1, positions.length - 1))} disabled={ply >= positions.length - 1} type="button" aria-label="Next move">→</button>
          </div>
          <input className="replay-scrubber" type="range" min="0" max={Math.max(0, positions.length - 1)} value={ply} onChange={(event) => { setPlaying(false); setPly(Number(event.target.value)); }} aria-label="Replay move" />
          <div className="replay-scrubber-labels"><span>Start</span><span>{move ? `Move ${move.ply}` : "Opening"}</span><span>End</span></div>
        </div>
      </div>
    </section>
  );
}

function ReplayHeading({ gameCount }: { gameCount?: number }) {
  return <div className="replay-heading"><div><span className="lab-kicker">Replay archive</span><h3>Walk the games</h3></div>{gameCount !== undefined && <span className="replay-count">{gameCount} games loaded</span>}</div>;
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

function capitalize(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
