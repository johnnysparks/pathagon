"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Action,
  GameState,
  Player,
  applyAction,
  createGame,
  createNearWinFixture,
  describeAction,
  legalActions,
  playerLabel,
} from "./pathagon";
import { OPPONENTS, SURVEYOR_OPPONENT, getOpponent } from "./opponents";

const HUMAN: Player = "light";
const AI: Player = "dark";

export default function Home() {
  const [game, setGame] = useState<GameState>(() => createGame());
  const [selected, setSelected] = useState<number | null>(null);
  const [thinking, setThinking] = useState(false);
  const [history, setHistory] = useState<GameState[]>([]);
  const [resultOpen, setResultOpen] = useState(false);
  const [captureNotice, setCaptureNotice] = useState<string | null>(null);
  const [opponentId, setOpponentId] = useState(SURVEYOR_OPPONENT.id);
  const aiTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const resultRef = useRef<HTMLDivElement | null>(null);
  const fixtureLoaded = useRef(false);

  const actions = useMemo(() => legalActions(game), [game]);
  const actionKeys = useMemo(() => new Set(actions.map(actionKey)), [actions]);
  const humanPlacementTurn = game.turn === HUMAN && game.reserve[HUMAN] > 0;
  const humanMovementTurn = game.turn === HUMAN && game.reserve[HUMAN] === 0;
  const opponent = getOpponent(opponentId);

  useEffect(() => {
    if (fixtureLoaded.current) return;
    fixtureLoaded.current = true;
    if (new URLSearchParams(window.location.search).get("fixture") === "near-win") {
      setGame(createNearWinFixture());
      setHistory([]);
    }
  }, []);

  useEffect(() => {
    if (game.winner || game.turn !== AI) {
      setThinking(false);
      return;
    }
    setThinking(true);
    aiTimer.current = setTimeout(() => {
      setGame((current) => {
        if (current.turn !== AI || current.winner) return current;
        const decision = opponent.chooseAction(current);
        if (!decision) return current;
        setHistory((items) => [...items, current]);
        return applyAction(current, decision);
      });
      setThinking(false);
    }, 420);
    return () => {
      if (aiTimer.current) clearTimeout(aiTimer.current);
    };
  }, [game, opponent]);

  useEffect(() => {
    if (!game.winner) return;
    setResultOpen(true);
    const focusTimer = setTimeout(() => resultRef.current?.focus(), 40);
    return () => clearTimeout(focusTimer);
  }, [game.winner]);

  useEffect(() => {
    const captured = game.lastAction?.captured.length ?? 0;
    if (!captured) return;
    const byHuman = game.lastAction?.player === HUMAN;
    const count = `${captured} ${captured === 1 ? "piece" : "pieces"}`;
    setCaptureNotice(
      byHuman
        ? `Trap! ${count} returned to ${opponent.name}.`
        : `Trap! ${opponent.name} returned ${count} to your hand.`,
    );
    const noticeTimer = setTimeout(() => setCaptureNotice(null), 2600);
    return () => clearTimeout(noticeTimer);
  }, [game.ply, game.lastAction, opponent.name]);

  function play(action: Action) {
    if (game.turn !== HUMAN || game.winner || thinking) return;
    setHistory((items) => [...items, game]);
    setGame(applyAction(game, action));
    setSelected(null);
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

  function newGame() {
    if (aiTimer.current) clearTimeout(aiTimer.current);
    setGame(createGame());
    setHistory([]);
    setSelected(null);
    setThinking(false);
    setResultOpen(false);
    setCaptureNotice(null);
  }

  function undoRound() {
    if (!history.length || thinking) return;
    let targetIndex = history.length - 1;
    while (targetIndex > 0 && history[targetIndex].turn !== HUMAN) targetIndex -= 1;
    setGame(history[targetIndex]);
    setHistory(history.slice(0, targetIndex));
    setSelected(null);
    setResultOpen(false);
    setCaptureNotice(null);
  }

  function changeOpponent(id: string) {
    if (aiTimer.current) clearTimeout(aiTimer.current);
    setOpponentId(id);
    setGame(createGame());
    setHistory([]);
    setSelected(null);
    setThinking(false);
    setResultOpen(false);
    setCaptureNotice(null);
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
              <button className="result-link" onClick={() => setResultOpen(true)}>View result</button>
            )}
          </div>

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
                return (
                  <button
                    key={index}
                    className={`cell ${piece ? "occupied" : ""} ${isSelected ? "selected" : ""} ${isLastMove ? "last-move" : ""} ${isWinningPath ? "winning" : ""} ${legalDestination || legalPlacement ? "legal" : ""}`}
                    onClick={() => handleCell(index)}
                    role="gridcell"
                    aria-label={cellLabel(index, piece, forbidden, movable)}
                    disabled={game.turn !== HUMAN || thinking || Boolean(game.winner) || forbidden}
                  >
                    <span className="socket" />
                    {piece && <span className={`piece ${piece}`} />}
                    {forbidden && <span className="forbidden-mark">×</span>}
                  </button>
                );
              })}
            </div>
          </div>

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
            <div><dt>Search depth</dt><dd>{opponent.searchDepth} ply</dd></div>
          </dl>
          <div className="event-log"><span className="stat-label">Latest event</span><p>{game.lastAction ? describeAction(game.lastAction) : "The board is empty. You have the first move."}</p></div>
          <div className="rules-note"><strong>v0 engine rules</strong><p>Light connects near-to-far; dark connects side-to-side. Orthogonal paths. Automatic A–B–A captures.</p></div>
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
                setResultOpen(false);
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
            <div className="result-actions">
              <button className="result-primary" onClick={newGame}>Play again</button>
              <button className="result-secondary" onClick={() => setResultOpen(false)}>Review board</button>
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

function actionKey(action: Action) {
  return action.kind === "place" ? `p:${action.to}` : `m:${action.from}:${action.to}`;
}

function cellLabel(index: number, piece: Player | null, forbidden: boolean, movable: boolean) {
  const row = Math.floor(index / 7) + 1;
  const column = (index % 7) + 1;
  if (forbidden) return `Row ${row}, column ${column}, temporarily unavailable`;
  if (piece) return `Row ${row}, column ${column}, ${playerLabel(piece)} piece${movable ? ", selectable" : ""}`;
  return `Row ${row}, column ${column}, empty`;
}
