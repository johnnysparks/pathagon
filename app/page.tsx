"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  Action,
  GameState,
  Player,
  createGame,
  createNearWinFixture,
  describeAction,
  playerLabel,
} from "./pathagon";
import { createGameId } from "./game-record";
import { chooseOpponentAction, CNN_OPPONENT, CNN_SEARCH, OPPONENTS, SURVEYOR_OPPONENT, getOpponent } from "./opponents";
import { COACHING_SEARCH, type MoveEvaluation } from "./ai";
import { loadRustEngine, type RustEngine } from "./rust-engine";
import { loadCnnEngine, type CnnEngine } from "./cnn-engine";

const HUMAN: Player = "light";
const AI: Player = "dark";

export default function Home() {
  const [game, setGame] = useState<GameState>(() => createGame());
  const [selected, setSelected] = useState<number | null>(null);
  const [history, setHistory] = useState<GameState[]>([]);
  const [moveHistory, setMoveHistory] = useState<Action[]>([]);
  const [gameId, setGameId] = useState<string | null>(null);
  const [copyStatus, setCopyStatus] = useState<"idle" | "copied" | "error">("idle");
  const [resultDismissed, setResultDismissed] = useState(false);
  const [archiveStatus, setArchiveStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [captureNoticeDismissedPly, setCaptureNoticeDismissedPly] = useState<number | null>(null);
  const [coachingMoves, setCoachingMoves] = useState<MoveEvaluation[]>([]);
  const [coachingAction, setCoachingAction] = useState<Action | null>(null);
  const [coachingEvaluation, setCoachingEvaluation] = useState<MoveEvaluation | null>(null);
  const [coachingStatus, setCoachingStatus] = useState<"idle" | "searching" | "ready">("idle");
  const [opponentId, setOpponentId] = useState(SURVEYOR_OPPONENT.id);
  const [rustEngine, setRustEngine] = useState<RustEngine | null>(null);
  const [engineError, setEngineError] = useState<string | null>(null);
  const [cnnEngine, setCnnEngine] = useState<CnnEngine | null>(null);
  const [cnnError, setCnnError] = useState<string | null>(null);
  const aiTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const resultRef = useRef<HTMLDivElement | null>(null);
  const coachingRequest = useRef(0);
  const longPressTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const fixtureLoaded = useRef(false);
  const recordable = useRef(true);
  const recordedGame = useRef<string | null>(null);

  const actions = useMemo(() => rustEngine?.legalActions(game) ?? [], [game, rustEngine]);
  const actionKeys = useMemo(() => new Set(actions.map(actionKey)), [actions]);
  const humanPlacementTurn = game.turn === HUMAN && game.reserve[HUMAN] > 0;
  const humanMovementTurn = game.turn === HUMAN && game.reserve[HUMAN] === 0;
  const opponent = getOpponent(opponentId);
  const cnnReady = opponent.id !== CNN_OPPONENT.id || Boolean(cnnEngine);
  const thinking = !rustEngine || !cnnReady || (game.turn === AI && !game.winner);
  const resultOpen = Boolean(game.winner) && !resultDismissed;
  const captureCount = game.lastAction?.captured.length ?? 0;
  const captureNotice = captureCount > 0 && captureNoticeDismissedPly !== game.ply
    ? game.lastAction?.player === HUMAN
      ? `Trap! ${captureCount} ${captureCount === 1 ? "piece" : "pieces"} returned to ${opponent.name}.`
      : `Trap! ${opponent.name} returned ${captureCount} ${captureCount === 1 ? "piece" : "pieces"} to your hand.`
    : null;
  const bestCoachingMove = coachingMoves[0] ?? null;

  useEffect(() => {
    let cancelled = false;
    void loadRustEngine().then((engine) => {
      if (!cancelled) setRustEngine(engine);
    }).catch((error: unknown) => {
      if (!cancelled) setEngineError(error instanceof Error ? error.message : String(error));
    });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (opponent.id !== CNN_OPPONENT.id) return;
    let cancelled = false;
    void loadCnnEngine().then((engine) => {
      if (!cancelled) setCnnEngine(engine);
    }).catch((error: unknown) => {
      if (!cancelled) setCnnError(error instanceof Error ? error.message : String(error));
    });
    return () => { cancelled = true; };
  }, [opponent.id]);

  useEffect(() => {
    if (fixtureLoaded.current) return;
    fixtureLoaded.current = true;
    if (new URLSearchParams(window.location.search).get("fixture") === "near-win") {
      recordable.current = false;
      const fixtureTimer = setTimeout(() => {
        setGame(createNearWinFixture());
        setHistory([]);
        setMoveHistory([]);
      }, 0);
      return () => clearTimeout(fixtureTimer);
    }
  }, []);

  useEffect(() => {
    if (!rustEngine || !cnnReady || game.winner || game.turn !== AI) return;
    aiTimer.current = setTimeout(() => {
      setGame((current) => {
        if (current.turn !== AI || current.winner) return current;
        const decision = chooseOpponentAction(rustEngine, opponent, current, cnnEngine ?? undefined);
        if (!decision) return current;
        setHistory((items) => [...items, current]);
        setMoveHistory((items) => [...items, decision]);
        return rustEngine.applyAction(current, decision);
      });
    }, 420);
    return () => {
      if (aiTimer.current) clearTimeout(aiTimer.current);
    };
  }, [cnnEngine, cnnReady, game, opponent, rustEngine]);

  useEffect(() => {
    coachingRequest.current += 1;
    if (!rustEngine || game.winner || game.turn !== HUMAN || thinking) return;
    const request = coachingRequest.current;
    const analysisTimer = setTimeout(() => {
      const moves = rustEngine.analyzeActions(game, COACHING_SEARCH, game.reserve[HUMAN] > 0 ? 49 : 42);
      if (coachingRequest.current !== request) return;
      setCoachingMoves(moves);
    }, 120);
    return () => clearTimeout(analysisTimer);
  }, [game, rustEngine, thinking]);

  useEffect(() => {
    if (!rustEngine || !coachingAction || game.winner || game.turn !== HUMAN || thinking) return;
    const request = coachingRequest.current + 1;
    coachingRequest.current = request;
    const analysisTimer = setTimeout(() => {
      if (coachingRequest.current !== request) return;
      const evaluation = rustEngine.analyzeAction(game, coachingAction, COACHING_SEARCH);
      if (coachingRequest.current !== request) return;
      setCoachingEvaluation(evaluation);
      setCoachingStatus("ready");
    }, 80);
    return () => clearTimeout(analysisTimer);
  }, [coachingAction, game, rustEngine, thinking]);

  useEffect(() => {
    if (!game.winner) return;
    if (!resultOpen) return;
    const focusTimer = setTimeout(() => resultRef.current?.focus(), 40);
    return () => clearTimeout(focusTimer);
  }, [game.winner, resultOpen]);

  useEffect(() => {
    if (!game.winner || !recordable.current || !gameId || moveHistory.length !== game.ply) return;
    const key = `${opponent.id}:${game.winner}:${moveHistory.map(actionKey).join("|")}`;
    if (recordedGame.current === key) return;
    recordedGame.current = key;
    setArchiveStatus("saving");
    void fetch("/api/games", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ id: gameId, opponentId: opponent.id, winner: game.winner, actions: moveHistory }),
    }).then((response) => {
      if (!response.ok) throw new Error("archive rejected");
      setArchiveStatus("saved");
    }).catch(() => setArchiveStatus("error"));
  }, [game.winner, game.ply, moveHistory, opponent.id, gameId]);

  useEffect(() => {
    if (!captureCount) return;
    const capturedPly = game.ply;
    const noticeTimer = setTimeout(() => setCaptureNoticeDismissedPly(capturedPly), 2600);
    return () => clearTimeout(noticeTimer);
  }, [captureCount, game.ply]);

  function play(action: Action) {
    if (!rustEngine || game.turn !== HUMAN || game.winner || thinking) return;
    if (recordable.current && !gameId) setGameId(createGameId());
    setHistory((items) => [...items, game]);
    setMoveHistory((items) => [...items, action]);
    setGame(rustEngine.applyAction(game, action));
    setSelected(null);
    clearCoaching();
  }

  function handleCell(index: number) {
    if (game.turn !== HUMAN || game.winner || thinking) return;
    if (humanPlacementTurn) {
      const action: Action = { kind: "place", to: index };
      if (actionKeys.has(actionKey(action))) play(action);
      return;
    }
    if (!humanMovementTurn) return;
    if (game.board[index] === HUMAN) {
      if (game.lastRelocatedTo[HUMAN] === index) return;
      setSelected(index === selected ? null : index);
      return;
    }
    if (selected !== null) {
      const action: Action = { kind: "relocate", from: selected, to: index };
      if (actionKeys.has(actionKey(action))) play(action);
    }
  }

  function coachingActionForCell(index: number) {
    if (game.turn !== HUMAN || game.winner || thinking) return null;
    if (humanPlacementTurn) {
      const action: Action = { kind: "place", to: index };
      return actionKeys.has(actionKey(action)) ? action : null;
    }
    if (!humanMovementTurn) return null;
    if (selected !== null) {
      const action: Action = { kind: "relocate", from: selected, to: index };
      return actionKeys.has(actionKey(action)) ? action : null;
    }
    if (game.board[index] !== HUMAN) return null;
    return coachingMoves.find((move) => move.action.kind === "relocate" && move.action.from === index)?.action
      ?? actions.find((action) => action.kind === "relocate" && action.from === index)
      ?? null;
  }

  function previewCell(index: number) {
    const action = coachingActionForCell(index);
    setCoachingStatus(action ? "searching" : "idle");
    setCoachingEvaluation(null);
    setCoachingAction(action);
  }

  function clearCoachingPreview() {
    setCoachingAction(null);
    setCoachingEvaluation(null);
    setCoachingStatus("idle");
  }

  function startLongPress(index: number) {
    if (longPressTimer.current) clearTimeout(longPressTimer.current);
    longPressTimer.current = setTimeout(() => previewCell(index), 420);
  }

  function clearLongPress() {
    if (!longPressTimer.current) return;
    clearTimeout(longPressTimer.current);
    longPressTimer.current = null;
  }

  function clearCoaching() {
    clearLongPress();
    setCoachingAction(null);
    setCoachingEvaluation(null);
    setCoachingMoves([]);
    setCoachingStatus("idle");
  }

  function heatForCell(index: number) {
    if (humanPlacementTurn) return coachingMoves.find((move) => move.action.kind === "place" && move.action.to === index);
    if (humanMovementTurn && selected !== null) {
      return coachingMoves.find((move) => move.action.kind === "relocate" && move.action.from === selected && move.action.to === index);
    }
    if (humanMovementTurn && selected === null) {
      return coachingMoves.find((move) => move.action.kind === "relocate" && move.action.from === index);
    }
    return undefined;
  }

  function newGame() {
    if (aiTimer.current) clearTimeout(aiTimer.current);
    setGame(createGame());
    setHistory([]);
    setMoveHistory([]);
    setGameId(null);
    setCopyStatus("idle");
    setSelected(null);
    setResultDismissed(false);
    setCaptureNoticeDismissedPly(null);
    clearCoaching();
    setArchiveStatus("idle");
    recordedGame.current = null;
    recordable.current = true;
  }

  function undoRound() {
    if (!history.length || thinking) return;
    let targetIndex = history.length - 1;
    while (targetIndex > 0 && history[targetIndex].turn !== HUMAN) targetIndex -= 1;
    setGame(history[targetIndex]);
    setHistory(history.slice(0, targetIndex));
    setMoveHistory(moveHistory.slice(0, targetIndex));
    setSelected(null);
    setResultDismissed(false);
    setCaptureNoticeDismissedPly(null);
    clearCoaching();
    setArchiveStatus("idle");
    recordedGame.current = null;
  }

  function changeOpponent(id: string) {
    if (aiTimer.current) clearTimeout(aiTimer.current);
    setOpponentId(id);
    setGame(createGame());
    setHistory([]);
    setMoveHistory([]);
    setGameId(null);
    setCopyStatus("idle");
    setSelected(null);
    setResultDismissed(false);
    setCaptureNoticeDismissedPly(null);
    clearCoaching();
    setArchiveStatus("idle");
    recordedGame.current = null;
    recordable.current = true;
  }

  async function copyGameId() {
    if (!gameId) return;
    try {
      await navigator.clipboard.writeText(gameId);
      setCopyStatus("copied");
    } catch {
      const code = document.querySelector<HTMLElement>("code[data-game-id]");
      if (code) {
        const selection = window.getSelection();
        selection?.removeAllRanges();
        const range = document.createRange();
        range.selectNodeContents(code);
        selection?.addRange(range);
      }
      setCopyStatus("error");
    }
  }

  const status = game.winner
    ? game.winner === HUMAN ? "You win" : `${opponent.name} wins`
    : game.turn === HUMAN
      ? humanMovementTurn
        ? selected === null ? "Choose a piece to move" : "Choose its destination"
        : "Place a light piece"
      : thinking ? `${opponent.name} is choosing…` : `${opponent.name}'s turn`;

  return (
    <main className="app-shell">
      <div className="capture-announcer" role="status" aria-live="polite" aria-atomic="true">
        {captureNotice && <div className="capture-toast">{captureNotice}</div>}
      </div>
      <header className="topbar">
        <div>
          <div className="eyebrow">Fuchs family game · digital v0</div>
          <h1>Pathagon</h1>
        </div>
        <div className="header-actions">
          <Link className="quiet-button lab-nav-link" href="/lab">Learning lab</Link>
          <button className="quiet-button" onClick={undoRound} disabled={!history.length || thinking}>Undo round</button>
          <button className="primary-button" onClick={newGame}>New game</button>
        </div>
      </header>

      <section className="game-layout" aria-label="Pathagon game">
        <div className="play-column">
          <div className="turn-banner" data-winner={Boolean(game.winner)}>
            <span className={`turn-dot ${game.winner ?? game.turn}`} />
            <strong>{status}</strong>
            <span className="turn-detail">{game.winner ? "Game complete" : `Turn ${game.ply + 1}`}</span>
            {game.winner && !resultOpen && (
              <button className="result-link" onClick={() => setResultDismissed(false)}>View result</button>
            )}
          </div>

          {gameId && (
            <div className="game-id-card" aria-label="Game ID">
              <div>
                <span className="stat-label">Game ID</span>
                <code data-game-id>{gameId}</code>
                <p>Keep this token to ask about the replay later. Anyone with it can view the game.</p>
              </div>
              <button className="copy-button" onClick={copyGameId} type="button">
                {copyStatus === "copied" ? "Copied" : copyStatus === "error" ? "Select ID" : "Copy ID"}
              </button>
            </div>
          )}

          {!game.winner && (
            <div className="goal-guide">
              <span>You are light</span>
              <strong>Build forward: connect the near and far light edges</strong>
            </div>
          )}

          <div className={`board-frame ${game.winner ? "game-over" : ""}`} data-winner={game.winner ?? undefined}>
            <span className="goal-edge goal-edge-top light"><span>FAR · LIGHT</span></span>
            <span className="goal-edge goal-edge-bottom light"><span>NEAR · LIGHT</span></span>
            <span className="goal-edge goal-edge-left dark"><span>DARK PATH</span></span>
            <span className="goal-edge goal-edge-right dark"><span>DARK PATH</span></span>
            <div className="board" role="grid" aria-label="Seven by seven Pathagon board">
              {game.board.map((piece, index) => {
                const forbidden = game.forbidden.includes(index) && game.turn === HUMAN;
                const isSelected = selected === index;
                const isLastMove = game.lastAction?.to === index;
                const isWinningPath = game.winningPath.includes(index);
                const legalDestination = selected !== null && actionKeys.has(actionKey({ kind: "relocate", from: selected, to: index }));
                const legalPlacement = humanPlacementTurn && actionKeys.has(actionKey({ kind: "place", to: index }));
                const movable = humanMovementTurn && piece === HUMAN && game.lastRelocatedTo[HUMAN] !== index;
                const heat = heatForCell(index);
                const heatClassName = heat ? heatClass(heat.delta) : "";
                return (
                  <button
                    key={index}
                    className={`cell ${piece ? "occupied" : ""} ${isSelected ? "selected" : ""} ${isLastMove ? "last-move" : ""} ${isWinningPath ? "winning" : ""} ${legalDestination || legalPlacement ? "legal" : ""} ${heatClassName}`}
                    onClick={() => handleCell(index)}
                    onMouseEnter={() => previewCell(index)}
                    onMouseLeave={clearCoachingPreview}
                    onFocus={() => previewCell(index)}
                    onBlur={clearCoachingPreview}
                    onTouchStart={() => startLongPress(index)}
                    onTouchEnd={clearLongPress}
                    onTouchCancel={clearLongPress}
                    role="gridcell"
                    aria-label={cellLabel(index, piece, forbidden, movable)}
                    disabled={!rustEngine || game.turn !== HUMAN || thinking || Boolean(game.winner) || forbidden}
                  >
                    <span className="socket" />
                    {piece && <span className={`piece ${piece}`} />}
                    {forbidden && <span className="forbidden-mark">×</span>}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="board-orientation" role="note" aria-label="Board coordinate orientation">
            <span className="orientation-title">Board map</span>
            <span><strong>Letters</strong> A → G, left to right</span>
            <span><strong>Numbers</strong> 1 → 7, top to bottom</span>
            <span className="orientation-example"><strong>D3</strong> = column D, row 3</span>
          </div>

          <CoachingPanel
            evaluation={coachingEvaluation}
            status={coachingStatus}
            bestMove={bestCoachingMove}
            hasHeatmap={coachingMoves.length > 0}
          />

          <div className="piece-trays">
            <PieceTray label="You" player="light" count={game.reserve.light} active={!game.winner && game.turn === HUMAN} />
            <PieceTray label={opponent.name} player="dark" count={game.reserve.dark} active={!game.winner && game.turn === AI} />
          </div>
        </div>

        <aside className="lab-panel">
          <div className="panel-topline">
            <div className="panel-kicker">Opponent lab</div>
            <label className="opponent-picker">
              <span>Choose opponent</span>
              <select value={opponentId} onChange={(event) => changeOpponent(event.target.value)}>
                {OPPONENTS.map((option) => <option key={option.id} value={option.id}>{option.name}</option>)}
              </select>
            </label>
          </div>
          <div className="opponent-heading">
            <div className="avatar">{opponent.name.split(" ").map((word) => word[0]).join("").slice(0, 2)}</div>
            <div><h2>{opponent.name}</h2><p>{opponent.personality}</p></div>
          </div>
          <div className="rating-card">
            <div><span className="stat-label">Estimated Elo</span><strong>{opponent.elo}</strong></div>
            <span className="engine-pill">{opponent.engine}</span>
          </div>
          <dl className="telemetry">
            <div><dt>Legal actions</dt><dd>{actions.length}</dd></div>
            <div><dt>Phase</dt><dd>{game.winner ? "Complete" : game.reserve[game.turn] ? "Placement" : "Movement"}</dd></div>
            <div><dt>Captured last turn</dt><dd>{game.lastAction?.captured.length ?? 0}</dd></div>
            <div><dt>{opponent.searchDepth === null ? "Search budget" : "Search depth"}</dt><dd>{opponent.searchDepth === null ? `${CNN_SEARCH.simulations} PUCT simulations` : `${opponent.searchDepth} ply`}</dd></div>
          </dl>
          <div className="event-log"><span className="stat-label">Latest event</span><p>{game.lastAction ? describeAction(game.lastAction) : "The board is empty. You have the first move."}</p></div>
          <div className="rules-note"><strong>{engineError || cnnError ? "Engine unavailable" : !rustEngine ? "Loading Rust engine…" : opponent.id === CNN_OPPONENT.id && !cnnEngine ? "Loading CNN model…" : "Rust/WASM engine"}</strong><p>{engineError ?? cnnError ?? "Light connects near-to-far; dark connects side-to-side. Orthogonal paths. Automatic A–B–A captures."}</p></div>
        </aside>
      </section>

      {game.winner && resultOpen && (
        <div className="result-scrim" role="presentation">
          <div
            className="result-card"
            role="dialog"
            aria-modal="true"
            aria-labelledby="result-title"
            aria-describedby="result-description"
            tabIndex={-1}
            ref={resultRef}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                setResultDismissed(true);
                return;
              }
              if (event.key !== "Tab") return;
              const buttons = Array.from(resultRef.current?.querySelectorAll("button") ?? []);
              if (!buttons.length) return;
              const first = buttons[0];
              const last = buttons[buttons.length - 1];
              if (event.shiftKey && document.activeElement === first) {
                event.preventDefault();
                last.focus();
              } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault();
                first.focus();
              }
            }}
          >
            <div className="result-kicker">Game complete</div>
            <h2 id="result-title">
              {game.winner === HUMAN ? "You made the path." : `${opponent.name} made the path.`}
            </h2>
            <p id="result-description">
              {game.winner === HUMAN
                ? `Light connected the near and far edges in ${Math.ceil(game.ply / 2)} ${Math.ceil(game.ply / 2) === 1 ? "move" : "moves"}.`
                : `Dark connected both edges. ${opponent.name} takes this one.`}
            </p>
            <p className={`archive-status ${archiveStatus}`} role="status">
              {archiveStatus === "saving" && "Adding this game to the human strategy archive…"}
              {archiveStatus === "saved" && "Game added to the replay-validated human strategy archive."}
              {archiveStatus === "error" && "This game could not be archived. The result still counts on this board."}
            </p>
            {gameId && (
              <div className="result-game-id">
                <span>Game ID</span>
                <code data-game-id>{gameId}</code>
                <button className="copy-button" onClick={copyGameId} type="button">
                  {copyStatus === "copied" ? "Copied" : copyStatus === "error" ? "Select ID" : "Copy ID"}
                </button>
              </div>
            )}
            <div className="result-actions">
              <button className="result-primary" onClick={newGame}>Play again</button>
              <button className="result-secondary" onClick={() => setResultDismissed(true)}>Review board</button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

function PieceTray({ label, player, count, active }: { label: string; player: Player; count: number; active: boolean }) {
  return <div className={`piece-tray ${active ? "active" : ""}`}><span className={`mini-piece ${player}`} /><div><strong>{label}</strong><span>{count} in hand</span></div></div>;
}

function CoachingPanel({ evaluation, status, bestMove, hasHeatmap }: { evaluation: MoveEvaluation | null; status: "idle" | "searching" | "ready"; bestMove: MoveEvaluation | null; hasHeatmap: boolean }) {
  const beforeSignal = evaluation ? normalizeScore(evaluation.beforeScore) : 0;
  const afterSignal = evaluation ? normalizeScore(evaluation.score) : 0;
  const shift = evaluation ? normalizeShift(evaluation.delta) : 0;
  const beforeNeedleAngle = beforeSignal * 90;
  const afterNeedleAngle = afterSignal * 90;
  const resultTone = evaluation ? shift > 0.12 ? "positive" : shift < -0.12 ? "negative" : "even" : "waiting";
  const headline = !evaluation
    ? status === "searching" ? "Searching the move tree…" : "Hover a legal move to preview it"
    : shift > 0.22 ? "Strong improvement"
      : shift < -0.22 ? "A costly concession"
        : Math.abs(shift) < 0.08 ? "Keeps the balance"
          : shift > 0 ? "Slightly better for you" : "Slightly better for them";

  return (
    <section className={`coaching-panel ${resultTone}`} aria-live="polite" aria-label="Live move coach">
      <div className="coaching-heading">
        <div>
          <span className="panel-kicker">Live move coach</span>
          <h2>{headline}</h2>
        </div>
        <span className="coaching-status"><span className="coach-pulse" /> {status === "searching" ? `Analyzing${evaluation ? ` · ${evaluation.completedDepth}-ply` : "…"}` : hasHeatmap ? "Tree ready" : "Warming up"}</span>
      </div>
      <div className="coaching-body">
        <div className="advantage-gauge" aria-label={evaluation ? `Current ${formatAdvantage(beforeSignal)}; after ${formatAdvantage(afterSignal)}` : "Advantage gauge waiting for a move preview"}>
          <svg viewBox="0 0 220 132" role="img" aria-hidden="true">
            <path className="gauge-track" d="M 28 108 A 82 82 0 0 1 192 108" />
            <path className="gauge-left" d="M 28 108 A 82 82 0 0 1 110 26" />
            <path className="gauge-right" d="M 110 26 A 82 82 0 0 1 192 108" />
            <line className="gauge-needle gauge-needle-before" x1="110" y1="108" x2="110" y2="43" style={{ transform: `rotate(${beforeNeedleAngle}deg)`, transformOrigin: "110px 108px" }} />
            <line className="gauge-needle gauge-needle-after" x1="110" y1="108" x2="110" y2="43" style={{ transform: `rotate(${afterNeedleAngle}deg)`, transformOrigin: "110px 108px" }} />
            <circle className="gauge-hub" cx="110" cy="108" r="7" />
          </svg>
          <div className="gauge-labels"><span>Them</span><strong>After move</strong><span>You</span></div>
          {evaluation && (
            <div className="gauge-readouts">
              <span><i className="gauge-dot gauge-dot-before" /> Now: {formatAdvantage(beforeSignal)}</span>
              <span><i className="gauge-dot gauge-dot-after" /> After: {formatAdvantage(afterSignal)}</span>
            </div>
          )}
        </div>
        <div className="coaching-copy">
          {evaluation ? (
            <>
              <div className="coaching-move"><span>Previewing</span><strong>{formatAction(evaluation.action)}</strong></div>
              <p>{formatShift(evaluation.delta)}</p>
              <span className="coaching-meta">{evaluation.completedDepth}-ply minimax · {evaluation.nodes.toLocaleString()} nodes{evaluation.exhausted ? " · budget capped" : ""}</span>
            </>
          ) : (
            <>
              <p>Move over a square to see how the position shifts. On touch, hold a legal square for a moment.</p>
              {bestMove && <span className="coaching-meta">Best current idea: {formatAction(bestMove.action)}</span>}
            </>
          )}
        </div>
      </div>
      <div className="coaching-legend"><span><i className="legend-swatch good" /> helps your path</span><span><i className="legend-swatch bad" /> helps their path</span></div>
    </section>
  );
}

function actionKey(action: Action) {
  return action.kind === "place" ? `p:${action.to}` : `m:${action.from}:${action.to}`;
}

function heatClass(delta: number) {
  const signal = Math.max(-1, Math.min(1, delta / 420));
  if (signal > 0.52) return "coach-heat coach-heat-strong-good";
  if (signal > 0.12) return "coach-heat coach-heat-good";
  if (signal < -0.52) return "coach-heat coach-heat-strong-bad";
  if (signal < -0.12) return "coach-heat coach-heat-bad";
  return "coach-heat coach-heat-even";
}

function normalizeScore(score: number) {
  if (score >= 1_000_000_000) return 1;
  if (score <= -1_000_000_000) return -1;
  return Math.max(-1, Math.min(1, Math.tanh(score / 3_500)));
}

function normalizeShift(delta: number) {
  return Math.max(-1, Math.min(1, delta / 650));
}

function formatAdvantage(signal: number) {
  if (Math.abs(signal) < 0.08) return "Even";
  return signal > 0 ? `You +${Math.round(signal * 100)}` : `Them +${Math.round(Math.abs(signal) * 100)}`;
}

function formatShift(delta: number) {
  if (Math.abs(delta) < 20) return "The move keeps the advantage balance essentially unchanged.";
  const points = Math.round(Math.abs(delta) / 10);
  return delta > 0 ? `This move shifts the balance ${points} points toward your path.` : `This move shifts the balance ${points} points toward their path.`;
}

function formatAction(action: Action) {
  return action.kind === "place"
    ? coordinate(action.to)
    : `${coordinate(action.from)} → ${coordinate(action.to)}`;
}

function coordinate(index: number) {
  return `${String.fromCharCode(65 + (index % 7))}${Math.floor(index / 7) + 1}`;
}

function cellLabel(index: number, piece: Player | null, forbidden: boolean, movable: boolean) {
  const row = Math.floor(index / 7) + 1;
  const column = (index % 7) + 1;
  if (forbidden) return `Row ${row}, column ${column}, temporarily unavailable`;
  if (piece) return `Row ${row}, column ${column}, ${playerLabel(piece)} piece${movable ? ", selectable" : ""}`;
  return `Row ${row}, column ${column}, empty`;
}
