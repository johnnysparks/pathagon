"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
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
  DEFAULT_OPPONENT_ID,
  OPPONENTS,
  PATHFINDER_DEADLINE_MS,
  PATHFINDER_DEADLINE_OPTIONS,
  PATHFINDER_BEAM_OPTIONS,
  PATHFINDER_DEPTH_OPTIONS,
  PATHFINDER_MAX_NODES_DEFAULT,
  PATHFINDER_MAX_NODES_OPTIONS,
  getOpponent,
  type PlayerFacingOpponent,
  pathfinderMaxNodesForDepth,
  pathfinderSearchAtDepth,
} from "./opponents";
import { DOUBLE_DRAGON_ID, PATHMAN_ID, SEER_ID, TILE_DRIVER_ID, YANN_TILESON_ID } from "./agent-ids";
import { PATHFINDER_SEARCH } from "./ai";
import type { RankedAction, SearchTelemetry } from "./opponent-runtime";
import {
  compactPathfinderGameMetadata,
  type PathfinderMoveTelemetry,
  type SearchCheckpoint,
} from "./archive-metadata";
import { buildGameDebugPayload, formatGameDebugPayload } from "./game-debug";
import { loadRustEngine, type RustEngine, type SearchTrace } from "./rust-engine";
import { createRustSearchClient, type RustSearchClient, type SearchProgress } from "./rust-search-client";
import { loadCnnEngine, type CnnEngine } from "./cnn-engine";
import { loadGnnEngine, loadJepaEngine, loadQAdvEngine, type GnnEngine, type JepaEngine, type QAdvEngine } from "./learned-engine";

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

function initialPathfinderMaxNodes(depth: number): number {
  if (typeof window === "undefined") return PATHFINDER_MAX_NODES_DEFAULT;
  try {
    const stored = Number(window.localStorage.getItem("pathagon:pathfinder-max-nodes"));
    return PATHFINDER_MAX_NODES_OPTIONS.includes(stored as (typeof PATHFINDER_MAX_NODES_OPTIONS)[number])
      ? stored
      : pathfinderMaxNodesForDepth(depth);
  } catch {
    return pathfinderMaxNodesForDepth(depth);
  }
}

function initialPathfinderBeamWidth(depth: number): number {
  if (typeof window === "undefined") return PATHFINDER_SEARCH.beamWidth;
  try {
    const stored = Number(window.localStorage.getItem("pathagon:pathfinder-beam-width"));
    return PATHFINDER_BEAM_OPTIONS.includes(stored as (typeof PATHFINDER_BEAM_OPTIONS)[number])
      ? stored
      : pathfinderSearchAtDepth(depth).beamWidth;
  } catch {
    return pathfinderSearchAtDepth(depth).beamWidth;
  }
}

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
  const [opponentId, setOpponentId] = useState(DEFAULT_OPPONENT_ID);
  const [analystId, setAnalystId] = useState(DEFAULT_OPPONENT_ID);
  const [analystFollowing, setAnalystFollowing] = useState(true);
  const [expandedOpponentId, setExpandedOpponentId] = useState(DEFAULT_OPPONENT_ID);
  const [opponentSettings, setOpponentSettings] = useState<Record<string, Record<string, number>>>({});
  // Keep the first render deterministic for SSR/hydration. The saved browser
  // preference is applied after mount in the effect below.
  const [pathfinderDepth, setPathfinderDepth] = useState<number>(PATHFINDER_SEARCH.depth);
  const [pathfinderDeadlineMs, setPathfinderDeadlineMs] = useState<number>(PATHFINDER_DEADLINE_MS);
  const [pathfinderMaxNodes, setPathfinderMaxNodes] = useState<number>(PATHFINDER_MAX_NODES_DEFAULT);
  const [pathfinderBeamWidth, setPathfinderBeamWidth] = useState<number>(PATHFINDER_SEARCH.beamWidth);
  const [pathfinderDepthReady, setPathfinderDepthReady] = useState(false);
  const [pendingOpponentId, setPendingOpponentId] = useState<string | null>(null);
  const [rustEngine, setRustEngine] = useState<RustEngine | null>(null);
  const [searchClient] = useState<RustSearchClient | null>(() =>
    typeof window === "undefined" ? null : createRustSearchClient(),
  );
  const [engineError, setEngineError] = useState<string | null>(null);
  const [cnnEngine, setCnnEngine] = useState<CnnEngine | null>(null);
  const [cnnError, setCnnError] = useState<string | null>(null);
  const [gnnEngine, setGnnEngine] = useState<GnnEngine | null>(null);
  const [gnnError, setGnnError] = useState<string | null>(null);
  const [qadvEngine, setQadvEngine] = useState<QAdvEngine | null>(null);
  const [qadvError, setQadvError] = useState<string | null>(null);
  const [jepaEngine, setJepaEngine] = useState<JepaEngine | null>(null);
  const [jepaError, setJepaError] = useState<string | null>(null);
  const [pathfinderProgress, setPathfinderProgress] = useState<SearchProgress | null>(null);
  const [decisionTraces, setDecisionTraces] = useState<SearchTrace[]>([]);
  const [decisionRanked, setDecisionRanked] = useState<RankedAction[]>([]);
  const [decisionTelemetry, setDecisionTelemetry] = useState<SearchTelemetry | null>(null);
  const [decisionInterpretation, setDecisionInterpretation] = useState<"relative preference" | "random priority/order">("relative preference");
  const [analystMoves, setAnalystMoves] = useState<RankedAction[]>([]);
  const [analystTelemetry, setAnalystTelemetry] = useState<SearchTelemetry | null>(null);
  const [analystInterpretation, setAnalystInterpretation] = useState<"relative preference" | "random priority/order">("relative preference");
  const [analystStatus, setAnalystStatus] = useState<"idle" | "searching" | "ready" | "unavailable">("idle");
  const [analystError, setAnalystError] = useState<string | null>(null);
  const [decisionDepth, setDecisionDepth] = useState<number | null>(null);
  const [decisionFocusAction, setDecisionFocusAction] = useState<Action | null>(null);
  const [lastPathfinderSearch, setLastPathfinderSearch] = useState<PathfinderMoveTelemetry | null>(null);
  const [pathfinderSearches, setPathfinderSearches] = useState<PathfinderMoveTelemetry[]>([]);
  const aiTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const aiRequest = useRef<number | null>(null);
  const activePathfinderSearch = useRef<{ requestId: number; checkpoints: SearchCheckpoint[] } | null>(null);
  const resultRef = useRef<HTMLDivElement | null>(null);
  const opponentSwitchDialogRef = useRef<HTMLDivElement | null>(null);
  const analystRequest = useRef(0);
  const fixtureLoaded = useRef(false);
  const recordable = useRef(true);
  const recordedGame = useRef<string | null>(null);

  const actions = useMemo(() => rustEngine?.legalActions(game) ?? [], [game, rustEngine]);
  const actionKeys = useMemo(() => new Set(actions.map(actionKey)), [actions]);
  const humanPlacementTurn = game.turn === HUMAN && game.reserve[HUMAN] > 0;
  const humanMovementTurn = game.turn === HUMAN && game.reserve[HUMAN] === 0;
  const opponent = getOpponent(opponentId);
  const pendingOpponent = pendingOpponentId ? getOpponent(pendingOpponentId) : null;
  const analyst = getOpponent(analystId);
  const cnnReady = opponent.id !== SEER_ID || Boolean(cnnEngine);
  const gnnReady = opponent.id !== TILE_DRIVER_ID || Boolean(gnnEngine);
  const qadvReady = opponent.id !== DOUBLE_DRAGON_ID || Boolean(qadvEngine);
  const jepaReady = opponent.id !== YANN_TILESON_ID || Boolean(jepaEngine);
  const activeRuntimeReady = cnnReady && gnnReady && qadvReady && jepaReady;
  const thinking = !rustEngine || !activeRuntimeReady || !opponent.playable || (game.turn === AI && !game.winner);
  const resultOpen = Boolean(game.winner) && !resultDismissed;
  const captureCount = game.lastAction?.captured.length ?? 0;
  const captureNotice = captureCount > 0 && captureNoticeDismissedPly !== game.ply
    ? game.lastAction?.player === HUMAN
      ? `Trap! ${captureCount} ${captureCount === 1 ? "piece" : "pieces"} returned to ${opponent.name}.`
      : `Trap! ${opponent.name} returned ${captureCount} ${captureCount === 1 ? "piece" : "pieces"} to your hand.`
    : null;
  const activeRuntimeControls = useMemo(
    () => runtimeControlsFor(opponent, opponentSettings, opponent.id === PATHMAN_ID ? {
      depth: pathfinderDepth,
      beamWidth: pathfinderBeamWidth,
      maxNodes: pathfinderMaxNodes,
      maxTimeMs: pathfinderDeadlineMs,
    } : undefined),
    [opponent, opponentSettings, pathfinderBeamWidth, pathfinderDeadlineMs, pathfinderDepth, pathfinderMaxNodes],
  );
  const analystRuntimeControls = useMemo(
    () => runtimeControlsFor(analyst, opponentSettings, analyst.id === PATHMAN_ID ? {
      depth: pathfinderDepth,
      beamWidth: pathfinderBeamWidth,
      maxNodes: pathfinderMaxNodes,
      maxTimeMs: pathfinderDeadlineMs,
    } : undefined),
    [analyst, opponentSettings, pathfinderBeamWidth, pathfinderDeadlineMs, pathfinderDepth, pathfinderMaxNodes],
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
    if (opponent.id !== SEER_ID && analyst.id !== SEER_ID) return;
    let cancelled = false;
    void loadCnnEngine().then((engine) => {
      if (!cancelled) setCnnEngine(engine);
    }).catch((error: unknown) => {
      if (!cancelled) setCnnError(error instanceof Error ? error.message : String(error));
    });
    return () => { cancelled = true; };
  }, [analyst.id, opponent.id]);

  useEffect(() => {
    if (opponent.id !== YANN_TILESON_ID && analyst.id !== YANN_TILESON_ID) return;
    let cancelled = false;
    void loadJepaEngine().then((engine) => {
      if (!cancelled) setJepaEngine(engine);
    }).catch((error: unknown) => {
      if (!cancelled) setJepaError(error instanceof Error ? error.message : String(error));
    });
    return () => { cancelled = true; };
  }, [analyst.id, opponent.id]);

  useEffect(() => {
    if (opponent.id !== TILE_DRIVER_ID && analyst.id !== TILE_DRIVER_ID) return;
    let cancelled = false;
    void loadGnnEngine().then((engine) => {
      if (!cancelled) setGnnEngine(engine);
    }).catch((error: unknown) => {
      if (!cancelled) setGnnError(error instanceof Error ? error.message : String(error));
    });
    return () => { cancelled = true; };
  }, [analyst.id, opponent.id]);

  useEffect(() => {
    if (opponent.id !== DOUBLE_DRAGON_ID && analyst.id !== DOUBLE_DRAGON_ID) return;
    let cancelled = false;
    void loadQAdvEngine().then((engine) => {
      if (!cancelled) setQadvEngine(engine);
    }).catch((error: unknown) => {
      if (!cancelled) setQadvError(error instanceof Error ? error.message : String(error));
    });
    return () => { cancelled = true; };
  }, [analyst.id, opponent.id]);

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
    if (!rustEngine || !activeRuntimeReady || !opponent.playable || game.winner || game.turn !== AI) return;
    let cancelled = false;
    const current = game;
    const isPathman = opponent.id === PATHMAN_ID;
    const commitDecision = (decision: Action | null, search?: SearchProgress, checkpoints: SearchCheckpoint[] = []) => {
      if (!decision || cancelled) return;
      const telemetry = search && isPathman
        ? createPathfinderMoveTelemetry(current.ply + 1, decision, search, checkpoints, opponent, pathfinderDepth, pathfinderMaxNodes, pathfinderBeamWidth, pathfinderDeadlineMs)
        : null;
      setHistory((items) => [...items, current]);
      setMoveHistory((items) => [...items, decision]);
      if (telemetry) {
        setLastPathfinderSearch(telemetry);
        setPathfinderSearches((items) => [...items, telemetry]);
      }
      setPathfinderProgress(null);
      setDecisionTraces([]);
      setDecisionRanked([]);
      setDecisionTelemetry(null);
      setDecisionDepth(null);
      setDecisionFocusAction(null);
      activePathfinderSearch.current = null;
      setGame((latest) => latest === current ? rustEngine.applyAction(current, decision) : latest);
    };
    aiTimer.current = setTimeout(() => {
      if (isPathman && searchClient) {
        const checkpoints: SearchCheckpoint[] = [];
        let requestId = 0;
        const request = searchClient.search(current, opponent.id, pathfinderDepth, pathfinderDeadlineMs, pathfinderMaxNodes, pathfinderBeamWidth, (progress) => {
          if (cancelled || aiRequest.current !== requestId) return;
          if (progress.action) {
            checkpoints.push({
              action: progress.action,
              completedDepth: progress.completedDepth,
              elapsedMs: progress.elapsedMs,
              nodes: progress.nodes,
              maxNodes: progress.maxNodes,
              nodeCapReached: progress.nodeCapReached,
            });
          }
          setPathfinderProgress(progress);
        }, (trace) => {
          if (cancelled || aiRequest.current !== requestId) return;
          setDecisionTraces((items) => {
            const next = items.filter((item) => item.depth !== trace.depth);
            next.push(trace);
            return next.sort((left, right) => left.depth - right.depth);
          });
          setDecisionRanked(trace.candidates.map((candidate) => ({ action: candidate.action, preference: candidate.score })));
          setDecisionInterpretation("relative preference");
          setDecisionDepth(null);
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
          const fallback = chooseOpponentAction(rustEngine, opponent, current, cnnEngine ?? undefined, pathfinderDepth, pathfinderDeadlineMs, undefined, pathfinderMaxNodes, pathfinderBeamWidth);
          commitDecision(
            fallback,
            isPathman && fallback
              ? {
                action: fallback,
                score: 0,
                nodes: 0,
                maxNodes: pathfinderMaxNodes,
                nodeCapReached: false,
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
      try {
        const result = opponent.runtime.search(current, { rustEngine, cnnEngine: cnnEngine ?? undefined, gnnEngine: gnnEngine ?? undefined, qadvEngine: qadvEngine ?? undefined, jepaEngine: jepaEngine ?? undefined }, {
          controls: activeRuntimeControls,
          seed: current.ply + 1,
        });
        if (cancelled) return;
        setDecisionRanked(result.ranked);
        setDecisionTelemetry(result.telemetry);
        setDecisionInterpretation(result.interpretation);
        commitDecision(result.action);
      } catch (error: unknown) {
        console.error(`${opponent.name} runtime failed:`, error);
        setEngineError(error instanceof Error ? error.message : String(error));
        setDecisionRanked([]);
        setDecisionTelemetry(null);
      }
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
      setDecisionTraces([]);
      setDecisionRanked([]);
      setDecisionTelemetry(null);
      setDecisionDepth(null);
      setDecisionFocusAction(null);
    };
  }, [activeRuntimeControls, activeRuntimeReady, cnnEngine, game, gnnEngine, jepaEngine, opponent, pathfinderBeamWidth, pathfinderDeadlineMs, pathfinderDepth, pathfinderMaxNodes, qadvEngine, rustEngine, searchClient]);

  useEffect(() => {
    const hydrationTimer = window.setTimeout(() => {
      const hydratedDepth = initialPathfinderDepth();
      setPathfinderDepth(hydratedDepth);
      setPathfinderDeadlineMs(initialPathfinderDeadline());
      setPathfinderMaxNodes(initialPathfinderMaxNodes(hydratedDepth));
      setPathfinderBeamWidth(initialPathfinderBeamWidth(hydratedDepth));
      setPathfinderDepthReady(true);
    }, 0);
    return () => window.clearTimeout(hydrationTimer);
  }, []);

  useEffect(() => {
    if (!pathfinderDepthReady) return;
    try {
      window.localStorage.setItem("pathagon:pathfinder-depth", String(pathfinderDepth));
      window.localStorage.setItem("pathagon:pathfinder-deadline-ms", String(pathfinderDeadlineMs));
      window.localStorage.setItem("pathagon:pathfinder-max-nodes", String(pathfinderMaxNodes));
      window.localStorage.setItem("pathagon:pathfinder-beam-width", String(pathfinderBeamWidth));
    } catch {
      // Device storage can be disabled; the in-memory control still works.
    }
  }, [pathfinderBeamWidth, pathfinderDeadlineMs, pathfinderDepth, pathfinderMaxNodes, pathfinderDepthReady]);

  useEffect(() => {
    const request = analystRequest.current + 1;
    analystRequest.current = request;
    const analysisTimer = setTimeout(() => {
      if (analystRequest.current !== request) return;
      setAnalystMoves([]);
      setAnalystTelemetry(null);
      setAnalystError(null);
      setAnalystStatus("searching");
      if (!rustEngine || !analyst.playable || game.winner) {
        setAnalystStatus(analyst.playable ? "idle" : "unavailable");
        return;
      }
      try {
        const evaluation = analyst.runtime.evaluateBoard(game, { rustEngine, cnnEngine: cnnEngine ?? undefined, gnnEngine: gnnEngine ?? undefined, qadvEngine: qadvEngine ?? undefined, jepaEngine: jepaEngine ?? undefined }, {
          controls: analystRuntimeControls,
          seed: game.ply + 1,
        });
        if (analystRequest.current !== request) return;
        setAnalystMoves(evaluation.ranked);
        setAnalystTelemetry(evaluation.telemetry);
        setAnalystInterpretation(evaluation.interpretation);
        setAnalystStatus("ready");
      } catch (error: unknown) {
        if (analystRequest.current !== request) return;
        setAnalystStatus("unavailable");
        setAnalystError(error instanceof Error ? error.message : String(error));
      }
    }, 160);
    return () => clearTimeout(analysisTimer);
  }, [analyst, analystRuntimeControls, cnnEngine, game, gnnEngine, jepaEngine, qadvEngine, rustEngine]);

  useEffect(() => {
    const hydrationTimer = window.setTimeout(() => {
      try {
        const rawSettings = window.localStorage.getItem("pathagon:opponent-settings-v1");
        if (rawSettings) {
          const savedSettings = JSON.parse(rawSettings) as Record<string, Record<string, number>>;
          setOpponentSettings(savedSettings);
          const savedPathman = savedSettings[PATHMAN_ID];
          if (savedPathman) {
            const pathman = getOpponent(PATHMAN_ID);
            const savedValue = (controlId: string, fallback: number) => {
              const control = pathman.controls.find((candidate) => candidate.id === controlId);
              const index = Math.max(0, Math.min(4, savedPathman[controlId] ?? control?.defaultIndex ?? 2));
              return control?.values[index] ?? fallback;
            };
            setPathfinderDepth(savedValue("depth", PATHFINDER_SEARCH.depth));
            setPathfinderBeamWidth(savedValue("beamWidth", PATHFINDER_SEARCH.beamWidth));
            setPathfinderMaxNodes(savedValue("maxNodes", PATHFINDER_MAX_NODES_DEFAULT));
            setPathfinderDeadlineMs(savedValue("maxTimeMs", PATHFINDER_DEADLINE_MS));
          }
        }
        const storedOpponent = window.localStorage.getItem("pathagon:active-opponent-v1");
        if (storedOpponent) setOpponentId(getOpponent(storedOpponent).id);
        const rawAnalyst = window.localStorage.getItem("pathagon:analyst-v1");
        if (rawAnalyst) {
          const savedAnalyst = JSON.parse(rawAnalyst) as { id?: string; following?: boolean };
          if (savedAnalyst.id) setAnalystId(getOpponent(savedAnalyst.id).id);
          if (typeof savedAnalyst.following === "boolean") setAnalystFollowing(savedAnalyst.following);
        } else if (storedOpponent) {
          setAnalystId(getOpponent(storedOpponent).id);
        }
      } catch {
        // Device storage can be disabled or contain an older schema.
      }
    }, 0);
    return () => window.clearTimeout(hydrationTimer);
  }, []);

  useEffect(() => {
    if (!pathfinderDepthReady) return;
    try {
      window.localStorage.setItem("pathagon:active-opponent-v1", opponent.id);
      window.localStorage.setItem("pathagon:opponent-settings-v1", JSON.stringify(opponentSettings));
      window.localStorage.setItem("pathagon:analyst-v1", JSON.stringify({ id: analyst.id, following: analystFollowing }));
    } catch {
      // Device storage can be disabled; settings remain available in memory.
    }
  }, [analyst.id, analystFollowing, opponent.id, opponentSettings, pathfinderDepthReady]);

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
            metadata: buildPathfinderGameMetadata(opponent, pathfinderDepth, pathfinderMaxNodes, pathfinderBeamWidth, pathfinderDeadlineMs, pathfinderSearches),
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
  }, [game.winner, game.ply, moveHistory, opponent, pathfinderBeamWidth, pathfinderDeadlineMs, pathfinderDepth, pathfinderMaxNodes, pathfinderSearches, gameId]);

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
    clearAnalysis();
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

  function clearAnalysis() {
    analystRequest.current += 1;
    setAnalystMoves([]);
    setAnalystTelemetry(null);
    setAnalystStatus("idle");
    setAnalystError(null);
  }

  function heatForCell(index: number) {
    if (!analystMoves.length) return undefined;
    if (game.turn === HUMAN && humanPlacementTurn) return analystMoves.find((move) => move.action.kind === "place" && move.action.to === index);
    if (game.turn === HUMAN && humanMovementTurn && selected !== null) {
      return analystMoves.find((move) => move.action.kind === "relocate" && move.action.from === selected && move.action.to === index);
    }
    return analystMoves.find((move) => move.action.to === index || (move.action.kind === "relocate" && move.action.from === index));
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
    clearAnalysis();
    setArchiveStatus("idle");
    setArchiveError(null);
    setLastPathfinderSearch(null);
    setPathfinderSearches([]);
    setDecisionTraces([]);
    setDecisionRanked([]);
    setDecisionTelemetry(null);
    setDecisionDepth(null);
    setDecisionFocusAction(null);
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
    clearAnalysis();
    setPathfinderProgress(null);
    setDecisionTraces([]);
    setDecisionRanked([]);
    setDecisionTelemetry(null);
    setDecisionDepth(null);
    setDecisionFocusAction(null);
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
    if (analystFollowing) {
      setAnalystId(opponent.id);
      setAnalystFollowing(false);
    }
    setOpponentId(id);
    setExpandedOpponentId(id);
    setGame(createGame());
    setHistory([]);
    setMoveHistory([]);
    setGameId(null);
    setCopyStatus("idle");
    setSelected(null);
    setResultDismissed(false);
    setCaptureNoticeDismissedPly(null);
    clearAnalysis();
    setArchiveStatus("idle");
    setLastPathfinderSearch(null);
    setPathfinderSearches([]);
    setDecisionTraces([]);
    setDecisionRanked([]);
    setDecisionTelemetry(null);
    setDecisionDepth(null);
    setDecisionFocusAction(null);
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

  function changePathfinderMaxNodes(maxNodes: number) {
    if (!PATHFINDER_MAX_NODES_OPTIONS.includes(maxNodes as (typeof PATHFINDER_MAX_NODES_OPTIONS)[number])) return;
    setPathfinderMaxNodes(maxNodes);
    setPathfinderProgress(null);
  }

  function changePathfinderBeamWidth(beamWidth: number) {
    if (!PATHFINDER_BEAM_OPTIONS.includes(beamWidth as (typeof PATHFINDER_BEAM_OPTIONS)[number])) return;
    setPathfinderBeamWidth(beamWidth);
    setPathfinderProgress(null);
  }

  function cancelActivePathfinderSearch() {
    if (aiRequest.current !== null && searchClient) {
      searchClient.cancel(aiRequest.current);
      aiRequest.current = null;
    }
    activePathfinderSearch.current = null;
    setPathfinderProgress(null);
    setDecisionTraces([]);
    setDecisionRanked([]);
    setDecisionTelemetry(null);
    setDecisionDepth(null);
    setDecisionFocusAction(null);
  }

  function selectAnalyst(id: string) {
    const selectedAnalyst = getOpponent(id);
    if (!selectedAnalyst.playable) return;
    setAnalystId(selectedAnalyst.id);
    setAnalystFollowing(false);
    setExpandedOpponentId(selectedAnalyst.id);
  }

  function updateOpponentControl(id: string, controlId: string, index: number) {
    const card = getOpponent(id);
    const control = card.controls.find((candidate) => candidate.id === controlId);
    if (!control || index < 0 || index > 4) return;
    const value = control.values[index];
    setOpponentSettings((settings) => ({
      ...settings,
      [card.id]: { ...(settings[card.id] ?? {}), [controlId]: index },
    }));
    if (card.id !== PATHMAN_ID) return;
    if (controlId === "depth") changePathfinderDepth(value);
    if (controlId === "beamWidth") changePathfinderBeamWidth(value);
    if (controlId === "maxNodes") changePathfinderMaxNodes(value);
    if (controlId === "maxTimeMs") changePathfinderDeadline(value);
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
      pathfinderMaxNodes,
      pathfinderBeamWidth,
      pathfinderDeadlineMs,
      true,
    );
    setHistory((items) => [...items, game]);
    setMoveHistory((items) => [...items, progress.action!]);
    setLastPathfinderSearch(telemetry);
    setPathfinderSearches((items) => [...items, telemetry]);
    setGame((latest) => latest === game ? rustEngine.applyAction(game, progress.action!) : latest);
  }

  async function copyGameDebug() {
    if (!gameId) return;
    const debugText = getGameDebugText();
    try {
      await navigator.clipboard.writeText(debugText);
      setCopyStatus("copied");
    } catch {
      const debug = document.querySelector<HTMLElement>("[data-game-debug]");
      if (debug) {
        const selection = window.getSelection();
        selection?.removeAllRanges();
        const range = document.createRange();
        range.selectNodeContents(debug);
        selection?.addRange(range);
      }
      setCopyStatus("error");
    }
  }

  function getGameDebugText() {
    if (!gameId) return "";
    return formatGameDebugPayload(buildGameDebugPayload({
      gameId,
      game,
      opponent,
      depth: pathfinderDepth,
      maxNodes: pathfinderMaxNodes,
      beamWidth: pathfinderBeamWidth,
      deadlineMs: pathfinderDeadlineMs,
      actions: moveHistory,
      pathfinderSearches,
      lastPathfinderSearch,
      pathfinderProgress,
      analystId: analyst.id,
      analystName: analyst.name,
      analystStatus,
      analystInterpretation,
      analystRanked: analystMoves,
      analystTelemetry,
      analystError,
      rustEngineReady: Boolean(rustEngine),
      cnnEngineReady: Boolean(cnnEngine),
      engineError,
      cnnError,
      archiveStatus,
      archiveError,
      pageUrl: typeof window === "undefined" ? undefined : window.location.href,
      userAgent: typeof window === "undefined" ? undefined : window.navigator.userAgent,
    }));
  }

  const status = game.winner
    ? game.winner === HUMAN ? "You win" : `${opponent.name} wins`
      : game.turn === HUMAN
        ? humanMovementTurn
          ? selected === null ? "Choose a piece to move" : "Choose its destination"
        : "Place a light piece"
      : thinking ? `${opponent.name} is choosing…` : `${opponent.name}'s turn`;
  const latestDecisionTrace = decisionTraces[decisionTraces.length - 1] ?? null;
  const displayedDecisionTrace = decisionDepth === null
    ? latestDecisionTrace
    : decisionTraces.find((trace) => trace.depth === decisionDepth) ?? latestDecisionTrace;

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
            <div className="game-id-card" aria-label="Game ID and debug log">
              <div>
                <span className="stat-label">Game ID</span>
                <code data-game-id>{gameId}</code>
                <p>Copy the ID and browser debug bundle for analysis.</p>
              </div>
              <button className="copy-button" aria-label="Copy game ID and debug log" onClick={copyGameDebug} type="button">
                {copyStatus === "copied" ? "Debug copied" : copyStatus === "error" ? "Select debug" : "Copy debug"}
              </button>
            </div>
          )}

          {gameId && (
            <pre data-game-debug aria-hidden="true" className="game-debug-copy-buffer">
              {getGameDebugText()}
            </pre>
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
                const heatClassName = heat ? opponentHeatClass(heat, analystMoves, analystInterpretation === "random priority/order") : "";
                const analystRank = heat ? analystMoves.indexOf(heat) + 1 : 0;
                return (
                  <button
                    key={index}
                    className={`cell ${piece ? "occupied" : ""} ${isSelected ? "selected" : ""} ${isLastMove ? "last-move" : ""} ${isWinningPath ? "winning" : ""} ${legalDestination || legalPlacement ? "legal" : ""} ${heatClassName}`}
                    onClick={() => handleCell(index)}
                    role="gridcell"
                    aria-label={`${cellLabel(index, piece, forbidden, movable)}${heat ? `, ${analyst.name} candidate ${analystRank}` : ""}`}
                    disabled={!rustEngine || game.turn !== HUMAN || thinking || Boolean(game.winner) || forbidden}
                  >
                    <span className="socket" />
                    {piece && <span className={`piece ${piece}`} />}
                    {heat && <span className="decision-rank" aria-hidden="true">{analystRank}</span>}
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

          <AnalystSummary
            analyst={analyst}
            status={analystStatus}
            error={analystError}
            ranked={analystMoves}
            telemetry={analystTelemetry}
            interpretation={analystInterpretation}
          />

          <div className="piece-trays">
            <PieceTray label="You" player="light" count={game.reserve.light} active={!game.winner && game.turn === HUMAN} />
            <PieceTray label={opponent.name} player="dark" count={game.reserve.dark} active={!game.winner && game.turn === AI} />
          </div>
        </div>

        <aside className="lab-panel">
          <div className="panel-topline">
            <div><div className="panel-kicker">Opponent lab</div><p className="panel-subtitle">Choose who plays and who studies the board.</p></div>
            <span className="analyst-follow-status">{analystFollowing ? "Analyst follows play" : "Analyst independent"}</span>
          </div>
          <OpponentCards
            opponents={OPPONENTS}
            activeId={opponent.id}
            analystId={analyst.id}
            expandedId={expandedOpponentId}
            settings={opponentSettings}
            onPlay={(id) => requestOpponentChange(id)}
            onAnalyze={selectAnalyst}
            onExpand={(id) => setExpandedOpponentId(id)}
            onControl={updateOpponentControl}
            decision={opponent.id === expandedOpponentId ? <DecisionTheater titleId="active-decision-theater-title" opponentName={opponent.name} trace={displayedDecisionTrace} timeline={decisionTraces} ranked={decisionRanked.length ? decisionRanked : opponent.id === analyst.id ? analystMoves : []} telemetry={decisionTelemetry ?? (opponent.id === analyst.id ? analystTelemetry : null)} interpretation={decisionRanked.length ? decisionInterpretation : opponent.id === analyst.id ? analystInterpretation : "relative preference"} selectedDepth={decisionDepth} focusedAction={decisionFocusAction} searching={Boolean(pathfinderProgress) || (opponent.id === analyst.id && analystStatus === "searching")} onSelectDepth={setDecisionDepth} onFocusAction={setDecisionFocusAction} onPlayBestMove={playCurrentBestMove} canPlayBest={Boolean(pathfinderProgress?.action && activePathfinderSearch.current)} /> : null}
            analystDecision={analyst.id === expandedOpponentId && analyst.id !== opponent.id ? <DecisionTheater titleId="analyst-decision-theater-title" opponentName={analyst.name} trace={null} timeline={[]} ranked={analystMoves} telemetry={analystTelemetry} interpretation={analystInterpretation} selectedDepth={null} focusedAction={null} searching={analystStatus === "searching"} onSelectDepth={() => undefined} onFocusAction={() => undefined} onPlayBestMove={() => undefined} canPlayBest={false} /> : null}
          />
          <dl className="telemetry">
            <div><dt>Legal actions</dt><dd>{actions.length}</dd></div>
            <div><dt>Phase</dt><dd>{game.winner ? "Complete" : game.reserve[game.turn] ? "Placement" : "Movement"}</dd></div>
            <div><dt>Captured last turn</dt><dd>{game.lastAction?.captured.length ?? 0}</dd></div>
            <div><dt>Analyst budget</dt><dd>{analyst.playable ? `${analystRuntimeControls.depth ?? 1}-ply · ${formatSearchTime(analystRuntimeControls.maxTimeMs ?? 0)}` : "Artifact required"}</dd></div>
          </dl>
          <div className="event-log"><span className="stat-label">Latest event</span><p>{game.lastAction ? describeAction(game.lastAction) : "The board is empty. You have the first move."}</p></div>
          <div className="rules-note"><strong>{engineError || cnnError || gnnError || qadvError || jepaError || analystError ? "Engine unavailable" : !rustEngine ? "Loading Rust engine…" : opponent.id === SEER_ID && !cnnEngine ? "Loading CNN model…" : opponent.id === TILE_DRIVER_ID && !gnnEngine ? "Loading GNN model…" : opponent.id === DOUBLE_DRAGON_ID && !qadvEngine ? "Loading Q/Advantage model…" : opponent.id === YANN_TILESON_ID && !jepaEngine ? "Loading JEPA model…" : !opponent.playable ? "Artifact required" : "Rust/WASM engine"}</strong><p>{engineError ?? cnnError ?? gnnError ?? qadvError ?? jepaError ?? analystError ?? opponent.description}</p></div>
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
                <button className="copy-button" aria-label="Copy game ID and debug log" onClick={copyGameDebug} type="button">
                  {copyStatus === "copied" ? "Debug copied" : copyStatus === "error" ? "Select debug" : "Copy debug"}
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
  maxNodes: number,
  beamWidth: number,
  deadlineMs: number,
  interrupted = false,
): PathfinderMoveTelemetry {
  const searchConfig = pathfinderSearchAtDepth(depth, maxNodes, beamWidth);
  return {
    ply,
    action,
    searchTimeMs: progress.elapsedMs,
    positions: progress.nodes,
    maxNodes: progress.maxNodes,
    nodeCapReached: progress.nodeCapReached,
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
  maxNodes: number,
  beamWidth: number,
  deadlineMs: number,
  searches: PathfinderMoveTelemetry[],
) {
  if (opponent.id !== PATHMAN_ID) return {};
  return compactPathfinderGameMetadata(pathfinderModelCard(opponent), depth, maxNodes, beamWidth, deadlineMs, searches);
}

function PieceTray({ label, player, count, active }: { label: string; player: Player; count: number; active: boolean }) {
  return <div className={`piece-tray ${active ? "active" : ""}`}><span className={`mini-piece ${player}`} /><div><strong>{label}</strong><span>{count} in hand</span></div></div>;
}

function runtimeControlIndex(card: PlayerFacingOpponent, settings: Record<string, Record<string, number>>, controlId: string) {
  const control = card.controls.find((candidate) => candidate.id === controlId);
  if (!control) return 2;
  return Math.max(0, Math.min(4, settings[card.id]?.[controlId] ?? control.defaultIndex));
}

function runtimeControlsFor(card: PlayerFacingOpponent, settings: Record<string, Record<string, number>>, overrides?: Record<string, number>) {
  if (overrides) return overrides;
  return Object.fromEntries(card.controls.map((control) => [control.id, control.values[runtimeControlIndex(card, settings, control.id)]]));
}

function OpponentCards({ opponents, activeId, analystId, expandedId, settings, onPlay, onAnalyze, onExpand, onControl, decision, analystDecision }: {
  opponents: readonly PlayerFacingOpponent[];
  activeId: string;
  analystId: string;
  expandedId: string;
  settings: Record<string, Record<string, number>>;
  onPlay: (id: string) => void;
  onAnalyze: (id: string) => void;
  onExpand: (id: string) => void;
  onControl: (id: string, controlId: string, index: number) => void;
  decision: ReactNode;
  analystDecision?: ReactNode;
}) {
  return <section className="opponent-roster" aria-label="Opponent cards">
    {opponents.map((card) => <OpponentCardTile
      key={card.id}
      card={card}
      active={card.id === activeId}
      analyst={card.id === analystId}
      expanded={card.id === expandedId}
      settings={settings}
      onPlay={onPlay}
      onAnalyze={onAnalyze}
      onExpand={onExpand}
      onControl={onControl}
      decision={card.id === activeId ? decision : null}
      analystDecision={card.id === analystId ? analystDecision : null}
    />)}
  </section>;
}

function OpponentCardTile({ card, active, analyst, expanded, settings, onPlay, onAnalyze, onExpand, onControl, decision, analystDecision }: {
  card: PlayerFacingOpponent;
  active: boolean;
  analyst: boolean;
  expanded: boolean;
  settings: Record<string, Record<string, number>>;
  onPlay: (id: string) => void;
  onAnalyze: (id: string) => void;
  onExpand: (id: string) => void;
  onControl: (id: string, controlId: string, index: number) => void;
  decision: ReactNode;
  analystDecision: ReactNode;
}) {
  const [infoOpen, setInfoOpen] = useState(false);
  const statusLabel = card.status === "artifact-pending" ? "Artifact pending" : card.id === PATHMAN_ID ? "Default" : card.nonStrategic ? "Control" : "Ready";
  return <article className={`opponent-card ${active ? "play-active" : ""} ${analyst ? "analyze-active" : ""} ${expanded ? "expanded" : ""} ${card.status === "artifact-pending" ? "artifact-pending" : ""}`}>
    <button className="opponent-card-header" type="button" onClick={() => onExpand(card.id)} aria-expanded={expanded}>
      <span className="opponent-glyph" aria-hidden="true">{card.glyph}</span>
      <span className="opponent-card-identity"><strong>{card.cuteName}</strong><small>{card.technicalName}</small></span>
      <span className="opponent-card-chevron" aria-hidden="true">{expanded ? "−" : "+"}</span>
    </button>
    <div className="opponent-card-badges"><span>{statusLabel}</span><span>{card.capabilities.length} capabilities</span>{card.nonStrategic ? <span>Order only</span> : null}</div>
    <div className="opponent-role-row" aria-label={`${card.cuteName} roles`}>
      <button className={`role-toggle ${active ? "selected" : ""}`} type="button" onClick={() => { onExpand(card.id); onPlay(card.id); }} disabled={!card.playable} aria-pressed={active} title={!card.playable ? `Requires ${card.artifact}` : `Use ${card.cuteName} as the game opponent`}>
        <span>Play</span>{active ? "On" : "Off"}
      </button>
      <button className={`role-toggle ${analyst ? "selected analyst" : ""}`} type="button" onClick={() => { onExpand(card.id); onAnalyze(card.id); }} disabled={!card.playable} aria-pressed={analyst} title={!card.playable ? `Requires ${card.artifact}` : `Use ${card.cuteName} as the board analyst`}>
        <span>Analyze</span>{analyst ? "On" : "Off"}
      </button>
    </div>
    {expanded && <div className="opponent-card-details">
      <div className="opponent-card-description"><span>{card.personality}</span><button className="info-button" type="button" onClick={() => setInfoOpen((open) => !open)} aria-expanded={infoOpen} aria-label={`More information about ${card.cuteName}`}>(i)</button></div>
      {infoOpen && <p className="opponent-info">{card.description}</p>}
      {card.status === "artifact-pending" && <p className="artifact-note">This card stays visible for the shared roster, but Play and Analyze unlock only after the real artifact is promoted: <code>{card.artifact}</code></p>}
      <div className={`opponent-controls ${card.playable ? "" : "disabled"}`} aria-label={`${card.cuteName} controls`}>
        <div className="control-heading"><span>Model-owned controls</span><small>Five-step presets</small></div>
        {card.controls.map((control) => {
          const index = runtimeControlIndex(card, settings, control.id);
          return <label className="model-control" key={control.id}>
            <span className="model-control-label"><strong>{control.label}</strong><button className="info-button" type="button" title={control.info} aria-label={`${control.label}: ${control.info}`}>(i)</button><em>{control.format(control.values[index])}</em></span>
            <input type="range" min={0} max={4} step={1} value={index} disabled={!card.playable} aria-label={`${card.cuteName} ${control.label}`} onChange={(event) => onControl(card.id, control.id, Number(event.target.value))} />
            <span className="model-control-scale"><small>{control.format(control.values[0])}</small><small>{control.format(control.values[2])}</small><small>{control.format(control.values[4])}</small></span>
          </label>;
        })}
      </div>
      {decision}
      {analystDecision}
    </div>}
  </article>;
}

function AnalystSummary({ analyst, status, error, ranked, telemetry, interpretation }: {
  analyst: PlayerFacingOpponent;
  status: "idle" | "searching" | "ready" | "unavailable";
  error: string | null;
  ranked: RankedAction[];
  telemetry: SearchTelemetry | null;
  interpretation: "relative preference" | "random priority/order";
}) {
  const best = ranked[0];
  return <section className="analyst-summary" aria-live="polite" aria-label={`${analyst.name} board evaluation`}>
    <div className="analyst-summary-heading"><div><span className="panel-kicker">Selected analyst</span><h2>{analyst.name} sees…</h2></div><span className={`analyst-status ${status}`}><span />{status === "searching" ? "Thinking" : status === "ready" ? "Live" : status === "unavailable" ? "Unavailable" : "Waiting"}</span></div>
    {status === "unavailable" ? <p className="analyst-empty">{error ?? (analyst.playable ? "The analyst could not evaluate this board." : "This opponent needs its real browser artifact before it can analyze.")}</p> : best ? <div className="analyst-result"><strong>{formatAction(best.action)}</strong><span>{interpretation === "random priority/order" ? "random priority/order" : "top relative preference"}</span><small>{telemetry ? `${telemetry.nodes.toLocaleString()} positions · ${telemetry.depth} ply` : "board evaluation ready"}</small></div> : <p className="analyst-empty">{status === "searching" ? "Ranking legal moves…" : "Select Analyze on any ready opponent card."}</p>}
  </section>;
}

type DecisionTheaterProps = {
  titleId?: string;
  opponentName: string;
  trace: SearchTrace | null;
  timeline: SearchTrace[];
  selectedDepth: number | null;
  focusedAction: Action | null;
  searching: boolean;
  ranked?: RankedAction[];
  telemetry?: SearchTelemetry | null;
  interpretation?: "relative preference" | "random priority/order";
  onSelectDepth: (depth: number | null) => void;
  onFocusAction: (action: Action | null) => void;
  onPlayBestMove: () => void;
  canPlayBest: boolean;
};

function DecisionTheater({
  titleId = "decision-theater-title",
  opponentName,
  trace,
  timeline,
  ranked = [],
  telemetry = null,
  interpretation = "relative preference",
  selectedDepth,
  focusedAction,
  searching,
  onSelectDepth,
  onFocusAction,
  onPlayBestMove,
  canPlayBest,
}: DecisionTheaterProps) {
  const candidates = ranked.length
    ? ranked.slice(0, 8).map((candidate) => ({ action: candidate.action, score: candidate.preference, visits: candidate.visits, randomPriority: candidate.randomPriority }))
    : trace?.candidates.slice(0, 8).map((candidate) => ({ action: candidate.action, score: candidate.score, visits: undefined, randomPriority: undefined })) ?? [];
  const bestScore = candidates[0]?.score ?? 0;
  const worstScore = candidates[candidates.length - 1]?.score ?? bestScore;
  const focusedKey = focusedAction ? actionKey(focusedAction) : null;

  return (
    <section className="decision-theater" aria-labelledby={titleId}>
      <div className="decision-theater-heading">
        <div>
          <span className="stat-label">Decision theater</span>
          <h3 id={titleId}>What {opponentName} sees</h3>
        </div>
        <span className={`decision-live ${searching ? "active" : ""}`}><span />{searching ? "Live" : "Standby"}</span>
      </div>
      <p className="decision-theater-intro">The board glow follows the exact root actions this opponent considered. {interpretation === "random priority/order" ? "This is random priority/order, not strategic goodness or win probability." : "Scores are relative preferences, not win probabilities."}</p>
      {trace || candidates.length ? (
        <>
          <div className="decision-candidate-list" aria-label={`Top moves at search depth ${trace?.depth ?? telemetry?.depth ?? 1}`}>
            {candidates.map((candidate, index) => {
              const delta = candidate.score - bestScore;
              const focused = focusedKey === actionKey(candidate.action);
              return (
                <button
                  key={actionKey(candidate.action)}
                  className={`decision-candidate ${index === 0 ? "best" : ""} ${focused ? "focused" : ""}`}
                  type="button"
                  aria-pressed={focused}
                  onClick={() => onFocusAction(focused ? null : candidate.action)}
                >
                  <span className="decision-candidate-rank">{index + 1}</span>
                  <span className="decision-candidate-main">
                    <span className="decision-candidate-move">{formatAction(candidate.action)}</span>
                    <span className="decision-candidate-bar" aria-hidden="true"><span style={{ width: `${decisionBarWidth(candidate.score, bestScore, worstScore)}%` }} /></span>
                  </span>
                  <span className="decision-candidate-score">{interpretation === "random priority/order" ? `order ${index + 1}` : index === 0 ? "best" : formatDecisionDelta(delta)}{candidate.visits ? ` · ${candidate.visits} visits` : ""}</span>
                </button>
              );
            })}
          </div>
          <div className="decision-depths">
              <div className="decision-depth-heading"><span>{telemetry ? "Search telemetry" : "Completed passes"}</span><span>{telemetry ? `${telemetry.nodes.toLocaleString()} nodes · ${telemetry.elapsedMs}ms` : trace ? selectedDepth === null ? `Following ${trace.depth}-ply` : `Viewing ${selectedDepth}-ply` : "Board ranking"}</span></div>
            <div className="decision-depth-list" role="list" aria-label="Completed search depths">
              {timeline.map((item) => (
                <button
                  key={item.depth}
                  className={`decision-depth ${selectedDepth === item.depth || (selectedDepth === null && item.depth === trace?.depth) ? "active" : ""}`}
                  type="button"
                  aria-label={`View search depth ${item.depth}`}
                  aria-pressed={selectedDepth === item.depth}
                  onClick={() => onSelectDepth(selectedDepth === item.depth ? null : item.depth)}
                >
                  {item.depth}
                </button>
              ))}
            </div>
          </div>
          {canPlayBest && <button className="play-best-button" type="button" onClick={onPlayBestMove}>Play current best</button>}
        </>
      ) : (
        <div className="decision-empty" aria-live="polite">
          <span className="decision-empty-pulse" />
          <span>{searching ? "Waiting for the first completed search pass…" : `The next ${opponentName} turn will populate the theater.`}</span>
        </div>
      )}
    </section>
  );
}

function actionKey(action: Action) {
  return action.kind === "place" ? `p:${action.to}` : `m:${action.from}:${action.to}`;
}

function opponentHeatClass(candidate: RankedAction, ranked: RankedAction[], nonStrategic: boolean) {
  if (nonStrategic) return "opponent-heat opponent-heat-random";
  const rank = Math.max(0, ranked.findIndex((item) => actionKey(item.action) === actionKey(candidate.action)));
  const signal = ranked.length <= 1 ? 1 : 1 - rank / (ranked.length - 1);
  if (signal > 0.84) return "opponent-heat opponent-heat-strong-good";
  if (signal > 0.58) return "opponent-heat opponent-heat-good";
  if (signal > 0.32) return "opponent-heat opponent-heat-even";
  if (signal > 0.1) return "opponent-heat opponent-heat-bad";
  return "opponent-heat opponent-heat-even";
}


function decisionBarWidth(score: number, bestScore: number, worstScore: number) {
  const range = Math.max(1, bestScore - worstScore);
  return Math.round(Math.max(12, Math.min(100, 100 - ((bestScore - score) / range) * 88)));
}

function formatDecisionDelta(delta: number) {
  if (Math.abs(delta) < 1) return "near tie";
  return `Δ ${delta > 0 ? "+" : ""}${Math.round(delta)}`;
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
