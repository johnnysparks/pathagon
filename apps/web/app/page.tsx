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
import {
  chooseOpponentAction,
  CNN_OPPONENT,
  CNN_SEARCH,
  OPPONENTS,
  PATHFINDER_DEADLINE_MS,
  PATHFINDER_DEADLINE_OPTIONS,
  PATHFINDER_DEPTH_OPTIONS,
  PATHFINDER_OPPONENT,
  TRANSITION_PATHFINDER_OPPONENT,
  TRAINED_PATHFINDER_OPPONENT,
  getOpponent,
  pathfinderSearchAtDepth,
  trainedPathfinderSearchAtDepth,
} from "./opponents";
import { COACHING_SEARCH, PATHFINDER_SEARCH, type MoveEvaluation, type SearchConfig } from "./ai";
import { loadRustEngine, type RustEngine } from "./rust-engine";
import { createRustSearchClient, type RustSearchClient, type SearchProgress } from "./rust-search-client";
import { loadCnnEngine, type CnnEngine } from "./cnn-engine";

const HUMAN: Player = "light";
const AI: Player = "dark";

function initialPathfinderDepth(): number {
  if (typeof window === "undefined") return PATHFINDER_SEARCH.depth;
  try {
    const stored = Number(window.localStorage.getItem("pathagon:pathfinder-depth"));
    return PATHFINDER_DEPTH_OPTIONS.includes(stored as (typeof PATHFINDER_DEPTH_OPTIONS)[number])
      ? stored
      : PATHFINDER_SEARCH.depth;
  } catch {
    return PATHFINDER_SEARCH.depth;
  }
}

function initialPathfinderDeadline(): number {
  if (typeof window === "undefined") return PATHFINDER_DEADLINE_MS;
  try {
    const stored = Number(window.localStorage.getItem("pathagon:pathfinder-deadline-ms"));
    return PATHFINDER_DEADLINE_OPTIONS.includes(stored as (typeof PATHFINDER_DEADLINE_OPTIONS)[number])
      ? stored
      : PATHFINDER_DEADLINE_MS;
  } catch {
    return PATHFINDER_DEADLINE_MS;
  }
}

type SearchCheckpoint = Pick<SearchProgress, "completedDepth" | "nodes" | "elapsedMs" | "action">;

type PathfinderMoveTelemetry = {
  ply: number;
  action: Action;
  searchTimeMs: number;
  positions: number;
  targetDepth: number;
  completedDepth: number;
  tableHits: number;
  exhausted: boolean;
  interrupted: boolean;
  modelCard: {
    id: string;
    name: string;
    version: string;
    engine: string;
  };
  config: SearchConfig & { deadlineMs: number };
  checkpoints: SearchCheckpoint[];
};

export default function Home() {
  const [game, setGame] = useState<GameState>(() => createGame());
  const [selected, setSelected] = useState<number | null>(null);
  const [history, setHistory] = useState<GameState[]>([]);
  const [moveHistory, setMoveHistory] = useState<Action[]>([]);
  const [gameId, setGameId] = useState<string | null>(null);
  const [copyStatus, setCopyStatus] = useState<"idle" | "copied" | "error">("idle");
  const [resultDismissed, setResultDismissed] = useState(false);
  const [archiveStatus, setArchiveStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const [archiveError, setArchiveError] = useState<string | null>(null);
  const [captureNoticeDismissedPly, setCaptureNoticeDismissedPly] = useState<number | null>(null);
  const [coachingMoves, setCoachingMoves] = useState<MoveEvaluation[]>([]);
  const [coachingAction, setCoachingAction] = useState<Action | null>(null);
  const [coachingEvaluation, setCoachingEvaluation] = useState<MoveEvaluation | null>(null);
  const [coachingStatus, setCoachingStatus] = useState<"idle" | "searching" | "ready">("idle");
  const [opponentId, setOpponentId] = useState(TRANSITION_PATHFINDER_OPPONENT.id);
  // Keep the first render deterministic for SSR/hydration. The saved browser
  // preference is applied after mount in the effect below.
  const [pathfinderDepth, setPathfinderDepth] = useState<number>(PATHFINDER_SEARCH.depth);
  const [pathfinderDeadlineMs, setPathfinderDeadlineMs] = useState<number>(PATHFINDER_DEADLINE_MS);
  const [pathfinderDepthReady, setPathfinderDepthReady] = useState(false);
  const [pendingOpponentId, setPendingOpponentId] = useState<string | null>(null);
  const [rustEngine, setRustEngine] = useState<RustEngine | null>(null);
  const [searchClient] = useState<RustSearchClient | null>(() =>
    typeof window === "undefined" ? null : createRustSearchClient(),
  );
  const [engineError, setEngineError] = useState<string | null>(null);
  const [cnnEngine, setCnnEngine] = useState<CnnEngine | null>(null);
  const [cnnError, setCnnError] = useState<string | null>(null);
  const [pathfinderProgress, setPathfinderProgress] = useState<SearchProgress | null>(null);
  const [lastPathfinderSearch, setLastPathfinderSearch] = useState<PathfinderMoveTelemetry | null>(null);
  const [pathfinderSearches, setPathfinderSearches] = useState<PathfinderMoveTelemetry[]>([]);
  const [configCopyStatus, setConfigCopyStatus] = useState<"idle" | "copied" | "error">("idle");
  const aiTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const aiRequest = useRef<number | null>(null);
  const activePathfinderSearch = useRef<{ requestId: number; checkpoints: SearchCheckpoint[] } | null>(null);
  const resultRef = useRef<HTMLDivElement | null>(null);
  const opponentSwitchDialogRef = useRef<HTMLDivElement | null>(null);
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
  const pendingOpponent = pendingOpponentId ? getOpponent(pendingOpponentId) : null;
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
  const pathfinderConfig = useMemo(
    () => ({
      ...(opponent.id === TRAINED_PATHFINDER_OPPONENT.id || opponent.id === TRANSITION_PATHFINDER_OPPONENT.id
        ? trainedPathfinderSearchAtDepth(pathfinderDepth)
        : pathfinderSearchAtDepth(pathfinderDepth)),
      deadlineMs: pathfinderDeadlineMs,
    }),
    [opponent.id, pathfinderDeadlineMs, pathfinderDepth],
  );

  useEffect(() => {
    let cancelled = false;
    void loadRustEngine().then((engine) => {
      if (!cancelled) setRustEngine(engine);
    }).catch((error: unknown) => {
      if (!cancelled) setEngineError(error instanceof Error ? error.message : String(error));
    });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => () => {
    aiRequest.current = null;
    searchClient?.terminate();
  }, [searchClient]);

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
    let cancelled = false;
    const current = game;
    const isPathfinder = opponent.id === PATHFINDER_OPPONENT.id
      || opponent.id === TRAINED_PATHFINDER_OPPONENT.id
      || opponent.id === TRANSITION_PATHFINDER_OPPONENT.id;
    const commitDecision = (decision: Action | null, search?: SearchProgress, checkpoints: SearchCheckpoint[] = []) => {
      if (!decision || cancelled) return;
      const telemetry = search && isPathfinder
        ? createPathfinderMoveTelemetry(current.ply + 1, decision, search, checkpoints, opponent, pathfinderDepth, pathfinderDeadlineMs)
        : null;
      setHistory((items) => [...items, current]);
      setMoveHistory((items) => [...items, decision]);
      if (telemetry) {
        setLastPathfinderSearch(telemetry);
        setPathfinderSearches((items) => [...items, telemetry]);
      }
      setPathfinderProgress(null);
      activePathfinderSearch.current = null;
      setGame((latest) => latest === current ? rustEngine.applyAction(current, decision) : latest);
    };
    aiTimer.current = setTimeout(() => {
      if (isPathfinder && searchClient) {
        const checkpoints: SearchCheckpoint[] = [];
        let requestId = 0;
        const request = searchClient.search(current, opponent.id, pathfinderDepth, pathfinderDeadlineMs, (progress) => {
          if (cancelled || aiRequest.current !== requestId) return;
          checkpoints.push({
            action: progress.action,
            completedDepth: progress.completedDepth,
            elapsedMs: progress.elapsedMs,
            nodes: progress.nodes,
          });
          setPathfinderProgress(progress);
        });
        requestId = request.requestId;
        aiRequest.current = request.requestId;
        activePathfinderSearch.current = { requestId: request.requestId, checkpoints };
        void request.promise.then((progress) => {
          if (cancelled || aiRequest.current !== request.requestId) return;
          aiRequest.current = null;
          if (!progress) return;
          commitDecision(progress.action, progress, checkpoints);
        }).catch((error: unknown) => {
          if (cancelled || aiRequest.current !== request.requestId) return;
          aiRequest.current = null;
          console.error("Rust search worker failed; using the local fallback:", error);
          const fallback = chooseOpponentAction(rustEngine, opponent, current, cnnEngine ?? undefined, pathfinderDepth);
          commitDecision(
            fallback,
            isPathfinder && fallback
              ? {
                action: fallback,
                score: 0,
                nodes: 0,
                exhausted: true,
                completedDepth: 0,
                tableHits: 0,
                elapsedMs: 0,
                targetDepth: pathfinderDepth,
              }
              : undefined,
          );
        });
        return;
      }
      commitDecision(chooseOpponentAction(rustEngine, opponent, current, cnnEngine ?? undefined, pathfinderDepth));
    }, 420);
    return () => {
      cancelled = true;
      if (aiTimer.current) clearTimeout(aiTimer.current);
      if (aiRequest.current !== null && searchClient) {
        searchClient.cancel(aiRequest.current);
        aiRequest.current = null;
      }
      activePathfinderSearch.current = null;
      setPathfinderProgress(null);
    };
  }, [cnnEngine, cnnReady, game, opponent, pathfinderDeadlineMs, pathfinderDepth, rustEngine, searchClient]);

  useEffect(() => {
    const hydrationTimer = window.setTimeout(() => {
      setPathfinderDepth(initialPathfinderDepth());
      setPathfinderDeadlineMs(initialPathfinderDeadline());
      setPathfinderDepthReady(true);
    }, 0);
    return () => window.clearTimeout(hydrationTimer);
  }, []);

  useEffect(() => {
    if (!pathfinderDepthReady) return;
    try {
      window.localStorage.setItem("pathagon:pathfinder-depth", String(pathfinderDepth));
      window.localStorage.setItem("pathagon:pathfinder-deadline-ms", String(pathfinderDeadlineMs));
    } catch {
      // Device storage can be disabled; the in-memory control still works.
    }
  }, [pathfinderDeadlineMs, pathfinderDepth, pathfinderDepthReady]);

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
    if (!pendingOpponentId) return;
    const focusTimer = setTimeout(() => opponentSwitchDialogRef.current?.focus(), 40);
    return () => clearTimeout(focusTimer);
  }, [pendingOpponentId]);

  useEffect(() => {
    if (!game.winner || !recordable.current || !gameId || moveHistory.length !== game.ply) return;
    const key = `${opponent.id}:${game.winner}:${moveHistory.map(actionKey).join("|")}`;
    if (recordedGame.current === key) return;
    recordedGame.current = key;
    setArchiveStatus("saving");
    setArchiveError(null);
    void (async () => {
      try {
        const response = await fetch("/api/games", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            id: gameId,
            opponentId: opponent.id,
            winner: game.winner,
            actions: moveHistory,
            metadata: buildPathfinderGameMetadata(opponent, pathfinderDepth, pathfinderDeadlineMs, pathfinderSearches),
          }),
        });
        const payload = await response.json().catch(() => null) as { error?: unknown } | null;
        if (!response.ok) {
          const detail = payload && typeof payload.error === "string"
            ? payload.error
            : `Archive request failed (HTTP ${response.status})`;
          throw new Error(detail);
        }
        setArchiveStatus("saved");
      } catch (error: unknown) {
        const message = error instanceof Error ? error.message : "Unable to archive this game";
        console.error("Human game archive failed:", message);
        setArchiveError(message);
        setArchiveStatus("error");
      }
    })();
  }, [game.winner, game.ply, moveHistory, opponent, pathfinderDeadlineMs, pathfinderDepth, pathfinderSearches, gameId]);

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
    cancelActivePathfinderSearch();
    setPendingOpponentId(null);
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
    setArchiveError(null);
    setLastPathfinderSearch(null);
    setPathfinderSearches([]);
    setConfigCopyStatus("idle");
    recordedGame.current = null;
    recordable.current = true;
  }

  function undoRound() {
    if (!history.length || thinking) return;
    let targetIndex = history.length - 1;
    while (targetIndex > 0 && history[targetIndex].turn !== HUMAN) targetIndex -= 1;
    const targetPly = history[targetIndex].ply;
    setGame(history[targetIndex]);
    setHistory(history.slice(0, targetIndex));
    setMoveHistory(moveHistory.slice(0, targetIndex));
    setSelected(null);
    setResultDismissed(false);
    setCaptureNoticeDismissedPly(null);
    clearCoaching();
    setPathfinderProgress(null);
    setPathfinderSearches((items) => items.filter((item) => item.ply <= targetPly));
    setLastPathfinderSearch(null);
    setArchiveStatus("idle");
    setArchiveError(null);
    recordedGame.current = null;
  }

  function changeOpponent(id: string) {
    if (aiTimer.current) clearTimeout(aiTimer.current);
    cancelActivePathfinderSearch();
    setPendingOpponentId(null);
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
    setLastPathfinderSearch(null);
    setPathfinderSearches([]);
    setConfigCopyStatus("idle");
    recordedGame.current = null;
    recordable.current = true;
  }

  function requestOpponentChange(id: string) {
    if (id === opponentId) return;
    if (game.ply > 0 && !game.winner) {
      setPendingOpponentId(id);
      return;
    }
    changeOpponent(id);
  }

  function confirmOpponentChange() {
    if (pendingOpponentId) changeOpponent(pendingOpponentId);
  }

  function cancelOpponentChange() {
    setPendingOpponentId(null);
  }

  function changePathfinderDepth(depth: number) {
    const tuned = pathfinderSearchAtDepth(depth).depth;
    setPathfinderDepth(tuned);
    setPathfinderProgress(null);
  }

  function changePathfinderDeadline(deadlineMs: number) {
    if (!PATHFINDER_DEADLINE_OPTIONS.includes(deadlineMs as (typeof PATHFINDER_DEADLINE_OPTIONS)[number])) return;
    setPathfinderDeadlineMs(deadlineMs);
    setPathfinderProgress(null);
  }

  function cancelActivePathfinderSearch() {
    if (aiRequest.current !== null && searchClient) {
      searchClient.cancel(aiRequest.current);
      aiRequest.current = null;
    }
    activePathfinderSearch.current = null;
    setPathfinderProgress(null);
  }

  function playCurrentBestMove() {
    const progress = pathfinderProgress;
    const active = activePathfinderSearch.current;
    if (!rustEngine || !progress?.action || !active || active.requestId !== aiRequest.current) return;
    if (game.winner || game.turn !== AI || !actionKeys.has(actionKey(progress.action))) return;

    const checkpoints = [...active.checkpoints];
    cancelActivePathfinderSearch();
    const telemetry = createPathfinderMoveTelemetry(
      game.ply + 1,
      progress.action,
      progress,
      checkpoints,
      opponent,
      pathfinderDepth,
      pathfinderDeadlineMs,
      true,
    );
    setHistory((items) => [...items, game]);
    setMoveHistory((items) => [...items, progress.action!]);
    setLastPathfinderSearch(telemetry);
    setPathfinderSearches((items) => [...items, telemetry]);
    setGame((latest) => latest === game ? rustEngine.applyAction(game, progress.action!) : latest);
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

  async function copyPathfinderConfig() {
    const config = {
      opponent: {
        id: opponent.id,
        name: opponent.name,
        version: opponent.version,
        engine: opponent.engine,
      },
      dials: {
        depth: pathfinderDepth,
        deadlineMs: pathfinderDeadlineMs,
      },
      search: pathfinderConfig,
    };
    try {
      await navigator.clipboard.writeText(JSON.stringify(config, null, 2));
      setConfigCopyStatus("copied");
    } catch {
      setConfigCopyStatus("error");
    }
    window.setTimeout(() => setConfigCopyStatus("idle"), 2200);
  }

  const status = game.winner
    ? game.winner === HUMAN ? "You win" : `${opponent.name} wins`
      : game.turn === HUMAN
        ? humanMovementTurn
          ? selected === null ? "Choose a piece to move" : "Choose its destination"
        : "Place a light piece"
      : thinking ? `${opponent.name} is choosing…` : `${opponent.name}'s turn`;
  const pathfinderDepthIndex = Math.max(0, PATHFINDER_DEPTH_OPTIONS.indexOf(pathfinderDepth as (typeof PATHFINDER_DEPTH_OPTIONS)[number]));

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
              <select value={opponentId} onChange={(event) => requestOpponentChange(event.target.value)}>
                {OPPONENTS.map((option) => <option key={option.id} value={option.id}>{option.shortName ?? option.name}</option>)}
              </select>
            </label>
          </div>
          <div className="opponent-heading">
            <div className="avatar">{opponent.name.split(" ").map((word) => word[0]).join("").slice(0, 2)}</div>
            <div><h2>{opponent.name}</h2><p>{opponent.personality}</p></div>
          </div>
          <div className="rating-card">
            <div><span className="stat-label">Estimated Elo</span><strong>{opponent.elo}</strong></div>
            <span className="engine-pill">{isPathfinderOpponent(opponent.id) ? `${pathfinderDepth}-ply · tactical-safe` : opponent.engine}</span>
          </div>
          {isPathfinderOpponent(opponent.id) && (
            <section className="lookahead-control" aria-labelledby="pathfinder-lookahead-title">
              <div className="lookahead-heading">
                <div>
                  <span className="stat-label">Pathfinder control</span>
                  <strong id="pathfinder-lookahead-title">{pathfinderDepth}-ply look-ahead</strong>
                </div>
                <span className="lookahead-speed">{pathfinderDepthTier(pathfinderDepth)}</span>
              </div>
              <input
                aria-label="Pathfinder look-ahead depth"
                aria-describedby="pathfinder-lookahead-scale pathfinder-lookahead-help"
                aria-valuetext={`${pathfinderDepth}-ply ${pathfinderDepthTier(pathfinderDepth)}`}
                className="lookahead-slider"
                type="range"
                min={0}
                max={PATHFINDER_DEPTH_OPTIONS.length - 1}
                step="1"
                value={pathfinderDepthIndex}
                onChange={(event) => changePathfinderDepth(PATHFINDER_DEPTH_OPTIONS[Number(event.target.value)] ?? pathfinderDepth)}
              />
              <div id="pathfinder-lookahead-scale" className="lookahead-scale"><span>2 · Quick</span><span>4 · Balanced</span><span>20 · Long</span><span>100 · Horizon</span></div>
              <p id="pathfinder-lookahead-help">More ply lets the Pathfinder answer farther ahead. 20, 50, and 100 ply are horizon targets; time and position caps may stop earlier.</p>
              <div className="search-dial-row">
                <label className="search-time-picker">
                  <span>Max search time</span>
                  <select
                    aria-label="Pathfinder maximum search time"
                    value={pathfinderDeadlineMs}
                    onChange={(event) => changePathfinderDeadline(Number(event.target.value))}
                  >
                    {PATHFINDER_DEADLINE_OPTIONS.map((option) => <option key={option} value={option}>{formatSearchTime(option)}</option>)}
                  </select>
                </label>
                <button className="config-copy-button" type="button" onClick={copyPathfinderConfig}>
                  {configCopyStatus === "copied" ? "Config copied" : configCopyStatus === "error" ? "Copy unavailable" : "Copy config"}
                </button>
              </div>
              <span className="search-cap-note">Checkpoints: ~10,000 positions or 500ms · cap: {pathfinderConfig.maxNodes.toLocaleString()}</span>
              {pathfinderProgress && (
                <div className="search-progress" role="status" aria-live="polite">
                  <div className="search-progress-heading">
                    <span className="stat-label">Live search checkpoint</span>
                    <strong>{pathfinderProgress.completedDepth}/{pathfinderProgress.targetDepth} ply</strong>
                  </div>
                  <div className="search-progress-stats">
                    <span>{pathfinderProgress.nodes.toLocaleString()} positions</span>
                    <span>{formatSearchTime(pathfinderProgress.elapsedMs)} elapsed</span>
                  </div>
                  <p>{pathfinderProgress.action ? `Best so far: ${formatAction(pathfinderProgress.action)}` : "Finding a legal first candidate…"}</p>
                  <button className="play-best-button" type="button" onClick={playCurrentBestMove} disabled={!pathfinderProgress.action}>
                    Play current best
                  </button>
                </div>
              )}
              {!pathfinderProgress && lastPathfinderSearch && (
                <div className="search-last-result" role="status">
                  <span className="stat-label">Last dark search</span>
                  <p>{lastPathfinderSearch.completedDepth}/{lastPathfinderSearch.targetDepth} ply · {lastPathfinderSearch.positions.toLocaleString()} positions · {formatSearchTime(lastPathfinderSearch.searchTimeMs)}</p>
                </div>
              )}
            </section>
          )}
          <dl className="telemetry">
            <div><dt>Legal actions</dt><dd>{actions.length}</dd></div>
            <div><dt>Phase</dt><dd>{game.winner ? "Complete" : game.reserve[game.turn] ? "Placement" : "Movement"}</dd></div>
            <div><dt>Captured last turn</dt><dd>{game.lastAction?.captured.length ?? 0}</dd></div>
            <div><dt>{opponent.searchDepth === null ? "Search budget" : "Search depth"}</dt><dd>{isPathfinderOpponent(opponent.id) ? `${pathfinderDepth} ply · ${formatSearchTime(pathfinderDeadlineMs)}` : opponent.searchDepth === null ? `${CNN_SEARCH.simulations} PUCT simulations` : `${opponent.searchDepth} ply`}</dd></div>
          </dl>
          <div className="event-log"><span className="stat-label">Latest event</span><p>{game.lastAction ? describeAction(game.lastAction) : "The board is empty. You have the first move."}</p></div>
          <div className="rules-note"><strong>{engineError || cnnError ? "Engine unavailable" : !rustEngine ? "Loading Rust engine…" : opponent.id === CNN_OPPONENT.id && !cnnEngine ? "Loading CNN model…" : "Rust/WASM engine"}</strong><p>{engineError ?? cnnError ?? "Light connects near-to-far; dark connects side-to-side. Orthogonal paths. Automatic A–B–A captures."}</p></div>
        </aside>
      </section>

      {pendingOpponent && (
        <div className="result-scrim" role="presentation">
          <div
            className="result-card"
            role="dialog"
            aria-modal="true"
            aria-labelledby="opponent-switch-title"
            aria-describedby="opponent-switch-description"
            tabIndex={-1}
            ref={opponentSwitchDialogRef}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                cancelOpponentChange();
                return;
              }
              if (event.key !== "Tab") return;
              const buttons = Array.from(opponentSwitchDialogRef.current?.querySelectorAll("button") ?? []);
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
            <div className="result-kicker">New game required</div>
            <h2 id="opponent-switch-title">Switch opponent?</h2>
            <p id="opponent-switch-description">
              Changing to {pendingOpponent.shortName ?? pendingOpponent.name} will clear this board and its undo history.
            </p>
            <div className="result-actions">
              <button className="result-primary" onClick={confirmOpponentChange}>Start new game</button>
              <button className="result-secondary" autoFocus onClick={cancelOpponentChange}>Keep current game</button>
            </div>
          </div>
        </div>
      )}

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
              {archiveStatus === "error" && `This game could not be archived${archiveError ? `: ${archiveError}` : ""}. The result still counts on this board.`}
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

function isPathfinderOpponent(id: string) {
  return id === PATHFINDER_OPPONENT.id
    || id === TRAINED_PATHFINDER_OPPONENT.id
    || id === TRANSITION_PATHFINDER_OPPONENT.id;
}

function pathfinderDepthTier(depth: number) {
  if (depth <= 3) return "Quick";
  if (depth === 4) return "Balanced";
  if (depth < 20) return "Deep";
  if (depth < 50) return "Long";
  if (depth < 100) return "Extreme";
  return "Horizon";
}

function formatSearchTime(milliseconds: number) {
  if (milliseconds < 1_000) return `${Math.max(0, Math.round(milliseconds))}ms`;
  const seconds = milliseconds / 1_000;
  return `${Number(seconds.toFixed(seconds < 10 ? 1 : 0))}s`;
}

function pathfinderModelCard(opponent: { id: string; name: string; version: string; engine: string }) {
  return {
    id: opponent.id,
    name: opponent.name,
    version: opponent.version,
    engine: opponent.engine,
  };
}

function createPathfinderMoveTelemetry(
  ply: number,
  action: Action,
  progress: SearchProgress,
  checkpoints: SearchCheckpoint[],
  opponent: ReturnType<typeof getOpponent>,
  depth: number,
  deadlineMs: number,
  interrupted = false,
): PathfinderMoveTelemetry {
  const searchConfig = opponent.id === TRAINED_PATHFINDER_OPPONENT.id || opponent.id === TRANSITION_PATHFINDER_OPPONENT.id
    ? trainedPathfinderSearchAtDepth(depth)
    : pathfinderSearchAtDepth(depth);
  return {
    ply,
    action,
    searchTimeMs: progress.elapsedMs,
    positions: progress.nodes,
    targetDepth: progress.targetDepth,
    completedDepth: progress.completedDepth,
    tableHits: progress.tableHits,
    exhausted: progress.exhausted,
    interrupted,
    modelCard: {
      id: opponent.id,
      name: opponent.name,
      version: opponent.version,
      engine: opponent.engine,
    },
    config: { ...searchConfig, deadlineMs },
    checkpoints: checkpoints.map((checkpoint) => ({ ...checkpoint })),
  };
}

function buildPathfinderGameMetadata(
  opponent: ReturnType<typeof getOpponent>,
  depth: number,
  deadlineMs: number,
  searches: PathfinderMoveTelemetry[],
) {
  if (!isPathfinderOpponent(opponent.id)) return {};
  return {
    searchExperiment: "pathfinder-browser-v1",
    modelCard: pathfinderModelCard(opponent),
    dials: { depth, deadlineMs },
    moves: searches,
  };
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
