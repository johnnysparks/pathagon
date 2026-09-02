use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

#[cfg(target_arch = "wasm32")]
type Deadline = f64;

#[cfg(not(target_arch = "wasm32"))]
type Deadline = Instant;

#[cfg(target_arch = "wasm32")]
fn deadline_after_ms(milliseconds: u32) -> Deadline {
    js_sys::Date::now() + f64::from(milliseconds)
}

#[cfg(not(target_arch = "wasm32"))]
fn deadline_after_ms(milliseconds: u32) -> Deadline {
    Instant::now() + Duration::from_millis(u64::from(milliseconds))
}

#[cfg(target_arch = "wasm32")]
fn deadline_reached(deadline: Deadline) -> bool {
    js_sys::Date::now() >= deadline
}

#[cfg(not(target_arch = "wasm32"))]
fn deadline_reached(deadline: Deadline) -> bool {
    Instant::now() >= deadline
}

use crate::{
    bit, captures_from, column_mask, neighbor_mask_for, row_mask, squares, Action, GameState,
    Player, MAX_CELL_COUNT,
};

const WIN_SCORE: i32 = 1_000_000_000;
// Search bounds must strictly contain every value returned by `evaluate`.
// Terminal scores are intentionally close to +/-WIN_SCORE, so i32::MAX/4 is
// not a valid infinity: a minimizing node could leave its accumulator at
// POS_INF when every child is a terminal win (> POS_INF).
const NEG_INF: i32 = -WIN_SCORE - 1;
const POS_INF: i32 = WIN_SCORE + 1;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EvaluationWeights {
    pub path: i32,
    pub material: i32,
    pub capture: i32,
    pub structure: i32,
    pub threat: i32,
    pub edge: i32,
}

impl Default for EvaluationWeights {
    fn default() -> Self {
        Self {
            path: 240,
            material: 110,
            capture: 700,
            structure: 55,
            threat: 130,
            edge: 80,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SearchConfig {
    pub depth: u8,
    pub max_nodes: u64,
    pub beam_width: usize,
    pub weights: EvaluationWeights,
    pub tactical_proof_horizon: Option<u8>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            depth: 5,
            max_nodes: 256_000,
            beam_width: 256,
            weights: EvaluationWeights::default(),
            tactical_proof_horizon: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SearchResult {
    pub action: Option<Action>,
    pub score: i32,
    pub nodes: u64,
    pub exhausted: bool,
    pub completed_depth: u8,
    pub table_hits: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoveEvaluation {
    pub action: Action,
    pub before_score: i32,
    pub score: i32,
    pub delta: i32,
    pub nodes: u64,
    pub exhausted: bool,
    pub completed_depth: u8,
    pub table_hits: u64,
}

/// A root action and the score found for it at one completed iterative-search
/// pass. Scores for alternatives can be alpha-beta bounds, so consumers should
/// present them as search preferences rather than calibrated win probabilities.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RootSearchCandidate {
    pub action: Action,
    pub score: i32,
}

/// A deliberately coarse callback for browser-facing search telemetry. The
/// callback receives cumulative node and transposition-table-hit counts. It is
/// invoked from the WASM search budget, not from every node, so callers can
/// surface long searches without putting a message boundary in the hot loop.
pub type SearchProgressCallback = Box<dyn FnMut(u64, u64)>;

/// Browser-facing root trace callback. It fires once per completed root
/// iteration and carries only the legal root actions searched in that pass.
pub type SearchTraceCallback = Box<dyn FnMut(u8, u64, u64, Vec<RootSearchCandidate>)>;

#[cfg(target_arch = "wasm32")]
const PROGRESS_NODE_INTERVAL: u64 = 10_000;
#[cfg(target_arch = "wasm32")]
const PROGRESS_TIME_INTERVAL_MS: f64 = 500.0;
#[cfg(target_arch = "wasm32")]
const PROGRESS_POLL_INTERVAL_NODES: u64 = 256;

#[derive(Default)]
struct Budget {
    nodes: u64,
    exhausted: bool,
    table_hits: u64,
    deadline: Option<Deadline>,
    #[cfg(target_arch = "wasm32")]
    progress: Option<SearchProgressCallback>,
    #[cfg(target_arch = "wasm32")]
    next_progress_nodes: u64,
    #[cfg(target_arch = "wasm32")]
    next_progress_at: Deadline,
    #[cfg(target_arch = "wasm32")]
    last_progress_poll: u64,
}

impl Budget {
    #[allow(unused_mut)]
    fn with_deadline_and_progress(
        deadline: Option<Deadline>,
        progress: Option<SearchProgressCallback>,
    ) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let _ = progress;
        let mut budget = Self {
            deadline,
            #[cfg(target_arch = "wasm32")]
            progress,
            ..Self::default()
        };
        #[cfg(target_arch = "wasm32")]
        if budget.progress.is_some() {
            budget.next_progress_nodes = PROGRESS_NODE_INTERVAL;
            budget.next_progress_at = deadline_after_ms(PROGRESS_TIME_INTERVAL_MS as u32);
        }
        budget
    }

    fn count_node(&mut self) {
        self.nodes += 1;
        #[cfg(target_arch = "wasm32")]
        {
            if self.progress.is_none() {
                return;
            }
            let node_due = self.nodes >= self.next_progress_nodes;
            let time_due = if self.nodes.saturating_sub(self.last_progress_poll)
                >= PROGRESS_POLL_INTERVAL_NODES
            {
                self.last_progress_poll = self.nodes;
                js_sys::Date::now() >= self.next_progress_at
            } else {
                false
            };
            if !node_due && !time_due {
                return;
            }
            let now = js_sys::Date::now();
            self.next_progress_nodes =
                (self.nodes / PROGRESS_NODE_INTERVAL + 1) * PROGRESS_NODE_INTERVAL;
            self.next_progress_at = now + PROGRESS_TIME_INTERVAL_MS;
            if let Some(mut callback) = self.progress.take() {
                callback(self.nodes, self.table_hits);
                self.progress = Some(callback);
            }
        }
    }

    fn reached(&mut self, max_nodes: u64) -> bool {
        if self.nodes >= max_nodes || self.deadline.is_some_and(deadline_reached) {
            self.exhausted = true;
            true
        } else {
            false
        }
    }
}

/// Per-search move-ordering hints used by the hybrid root-limited search.
///
/// Pathfinder's static ordering is still the source of truth. These two
/// quiet cut-off moves are only a cheap way to make alpha-beta encounter a
/// previously successful refutation earlier on sibling nodes and during the
/// next iterative-deepening pass. They never change the legal move set.
#[derive(Default)]
struct SearchHints {
    killers: Vec<[Option<Action>; 2]>,
    history: HashMap<(usize, Action, bool), i32>,
}

impl SearchHints {
    fn killers_at(&self, ply: usize) -> [Option<Action>; 2] {
        self.killers.get(ply).copied().unwrap_or([None, None])
    }

    fn record_killer(&mut self, ply: usize, action: Action) {
        if self.killers.len() <= ply {
            self.killers.resize(ply + 1, [None, None]);
        }
        let slot = &mut self.killers[ply];
        if slot[0] != Some(action) {
            slot[1] = slot[0];
            slot[0] = Some(action);
        }
    }

    fn history_at(&self, ply: usize, action: Action, maximizing: bool) -> i32 {
        self.history
            .get(&(ply, action, maximizing))
            .copied()
            .unwrap_or(0)
    }

    fn record_history(&mut self, ply: usize, action: Action, maximizing: bool, depth: u8) {
        let bonus = i32::from(depth).saturating_mul(i32::from(depth)).max(1);
        let entry = self.history.entry((ply, action, maximizing)).or_default();
        *entry = entry.saturating_add(bonus).min(1_000_000);
    }
}

#[derive(Clone, Copy)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy)]
struct TableEntry {
    depth: u8,
    score: i32,
    bound: Bound,
    best_action: Option<Action>,
}

pub fn search_best_action(state: GameState, config: SearchConfig) -> SearchResult {
    search_best_action_with_root_order(state, config, &[])
}

/// Run the ordinary Pathfinder search with both a node ceiling and a
/// wall-clock deadline. Iterative deepening returns the last fully completed
/// depth when either limit is reached; the returned action remains legal.
pub fn search_best_action_with_deadline(
    state: GameState,
    config: SearchConfig,
    deadline_ms: u32,
) -> SearchResult {
    search_best_action_with_root_order_and_root_limit_internal_deadline(
        state,
        config,
        &[],
        false,
        None,
        true,
        Some(deadline_after_ms(deadline_ms.max(1))),
        None,
        None,
    )
}

/// Consult promoted exact endgame data before spending the ordinary search
/// budget.  A sidecar action is usable only when the immutable WDL shard also
/// proves the position is a win and the action is legal after symmetry
/// inversion.  Positions with only a value, or positions absent from gold,
/// continue through the normal Pathfinder search.
pub fn search_best_action_with_golden(
    state: GameState,
    config: SearchConfig,
    golden: &crate::golden::GoldenLookup,
) -> SearchResult {
    if let Some(action) = golden.proven_action(state) {
        return SearchResult {
            action: Some(action),
            score: WIN_SCORE,
            nodes: 0,
            exhausted: false,
            completed_depth: 0,
            table_hits: 1,
        };
    }
    search_best_action(state, config)
}

/// Consult ordered promoted gold layers before ordinary search. This is the
/// multi-ring form of [`search_best_action_with_golden`]: Ring-1 can remain a
/// rollback/control layer while newer exact frontier rings are overlaid
/// without merging or rewriting their immutable files.
pub fn search_best_action_with_golden_layers(
    state: GameState,
    config: SearchConfig,
    golden: &crate::golden::GoldenLookupLayers,
) -> SearchResult {
    if let Some(action) = golden.proven_action(state) {
        return SearchResult {
            action: Some(action),
            score: WIN_SCORE,
            nodes: 0,
            exhausted: false,
            completed_depth: 0,
            table_hits: 1,
        };
    }
    search_best_action(state, config)
}

/// WASM-safe equivalent of [`search_best_action_with_golden`]. The browser
/// supplies immutable bytes fetched from the versioned golden artifacts; the
/// action is returned directly when the WDL row and sidecar agree.
pub fn search_best_action_with_golden_bytes(
    state: GameState,
    config: SearchConfig,
    table_bytes: &[u8],
    sidecar_bytes: Option<&[u8]>,
) -> Result<(SearchResult, Option<crate::golden::GoldenOutcome>, bool), String> {
    let golden = crate::golden::MemoryGoldenLookup::open_bytes(
        table_bytes,
        sidecar_bytes,
        state.config.board_size,
        state.config.reserve_per_player,
    )
    .map_err(|error| error.to_string())?;
    let outcome = golden.lookup(state);
    let action = (outcome == Some(crate::golden::GoldenOutcome::Win))
        .then(|| golden.proven_action(state))
        .flatten();
    let result = if let Some(action) = action {
        SearchResult {
            action: Some(action),
            score: WIN_SCORE,
            nodes: 0,
            exhausted: false,
            completed_depth: 0,
            table_hits: 1,
        }
    } else {
        search_best_action(state, config)
    };
    Ok((result, outcome, action.is_some()))
}

/// WASM-safe multi-layer equivalent. The slices are ordered by lookup
/// priority and copied into an in-memory layered lookup before searching.
pub fn search_best_action_with_golden_layers_bytes(
    state: GameState,
    config: SearchConfig,
    layers: &[(&[u8], Option<&[u8]>)],
) -> Result<(SearchResult, Option<crate::golden::GoldenOutcome>, bool), String> {
    let golden = crate::golden::MemoryGoldenLookupLayers::open_bytes(
        layers,
        state.config.board_size,
        state.config.reserve_per_player,
    )
    .map_err(|error| error.to_string())?;
    let outcome = golden.lookup(state);
    let action = (outcome == Some(crate::golden::GoldenOutcome::Win))
        .then(|| golden.proven_action(state))
        .flatten();
    let result = if let Some(action) = action {
        SearchResult {
            action: Some(action),
            score: WIN_SCORE,
            nodes: 0,
            exhausted: false,
            completed_depth: 0,
            table_hits: 1,
        }
    } else {
        search_best_action(state, config)
    };
    Ok((result, outcome, action.is_some()))
}

/// Run the ordinary Pathfinder search while enabling the transposition-table
/// best-move and killer-move ordering hints. The legal root set and evaluator
/// are unchanged; this is a search-only ablation for measuring whether the
/// incumbent's conservative ordering leaves useful pruning unused.
pub fn search_best_action_with_tt_order(state: GameState, config: SearchConfig) -> SearchResult {
    search_best_action_with_root_order_and_root_limit_internal(
        state,
        config,
        &[],
        false,
        None,
        true,
    )
}

/// Apply Pathfinder's bounded immediate-threat guard to the root order while
/// preserving the complete legal root set and normal alpha-beta budget.
pub fn search_best_action_with_tactical_guard(
    state: GameState,
    config: SearchConfig,
) -> SearchResult {
    let root_order = ordered_root_actions_with_tactical_guard(state, state.turn, config.weights);
    search_best_action_with_root_order_and_root_limit_internal(
        state,
        config,
        &root_order,
        false,
        None,
        true,
    )
}

/// Restrict the root search to moves that do not hand the opponent an
/// immediate winning action. This is safe because the opponent's next move is
/// terminal whenever such an action exists; if every move is unsafe, retain
/// the complete fallback set so the engine still returns a legal move.
pub fn search_best_action_with_tactical_filter(
    state: GameState,
    config: SearchConfig,
) -> SearchResult {
    search_best_action_with_tactical_filter_until(state, config, None, None, None)
}

/// Search through the tactical-safe Pathfinder root with a wall-clock
/// deadline. The recursive search checks the deadline before expanding each
/// node, so the returned move is always the last legal move from a fully
/// completed iterative-deepening pass (or the deterministic root fallback if
/// the first pass cannot finish).
pub fn search_best_action_with_tactical_filter_deadline(
    state: GameState,
    config: SearchConfig,
    deadline_ms: u32,
) -> SearchResult {
    search_best_action_with_tactical_filter_until(
        state,
        config,
        Some(deadline_after_ms(deadline_ms.max(1))),
        None,
        None,
    )
}

/// Deadline-aware tactical-safe search with coarse progress callbacks for the
/// browser worker. The callback cadence is controlled by `Budget`.
pub fn search_best_action_with_tactical_filter_deadline_progress(
    state: GameState,
    config: SearchConfig,
    deadline_ms: u32,
    progress: SearchProgressCallback,
) -> SearchResult {
    search_best_action_with_tactical_filter_until(
        state,
        config,
        Some(deadline_after_ms(deadline_ms.max(1))),
        Some(progress),
        None,
    )
}

/// Deadline-aware tactical-safe search with root traces for the browser
/// decision theater. Progress remains coarse; root candidates arrive only at
/// completed iterative-search depths.
pub fn search_best_action_with_tactical_filter_deadline_trace(
    state: GameState,
    config: SearchConfig,
    deadline_ms: u32,
    progress: SearchProgressCallback,
    trace: SearchTraceCallback,
) -> SearchResult {
    search_best_action_with_tactical_filter_until(
        state,
        config,
        Some(deadline_after_ms(deadline_ms.max(1))),
        Some(progress),
        Some(trace),
    )
}

fn search_best_action_with_tactical_filter_until(
    state: GameState,
    config: SearchConfig,
    deadline: Option<Deadline>,
    progress: Option<SearchProgressCallback>,
    trace: Option<SearchTraceCallback>,
) -> SearchResult {
    let fallback = ordered_root_actions(state, state.turn, config.weights);
    if fallback.is_empty() {
        return search_best_action(state, config);
    }
    let safe = tactical_root_safe_actions(state, state.turn, config.weights);
    let root_limit = (safe.len() < fallback.len()).then_some(safe.len());
    search_best_action_with_root_order_and_root_limit_internal_deadline(
        state, config, &safe, false, root_limit, true, deadline, progress, trace,
    )
}

/// Use the bounded rule-grounded proof solver only for positions with a cheap
/// tactical signal, then fall back to the promoted tactical-safe Pathfinder.
///
/// The proof solver is horizon-limited and may return an unknown result when
/// its budget is exhausted. In that case, or when every root action receives
/// the same proof outcome, the incumbent remains authoritative. The proof
/// history is supplied by self-play so repetition draws are not mislabelled.
pub fn search_best_action_with_tactical_proof(
    state: GameState,
    config: SearchConfig,
    proof_horizon: u8,
    proof_nodes: u64,
    proof_history: &[(crate::endgame::EndgameRepetitionKey, u8)],
) -> SearchResult {
    if proof_horizon == 0
        || proof_nodes == 0
        || state.config.board_size > 7
        || state.legal_action_count() > 512
        || !has_tactical_signal(state)
    {
        return search_best_action_with_tactical_filter(state, config);
    }

    let root_order = ordered_root_actions(state, state.turn, config.weights);
    let analysis = crate::endgame::analyze_with_history_and_root_order(
        state,
        crate::endgame::TacticalProofConfig {
            horizon: proof_horizon,
            max_nodes: proof_nodes,
        },
        proof_history,
        &root_order,
    );
    let Ok(analysis) = analysis else {
        return search_best_action_with_tactical_filter(state, config);
    };
    if analysis.stats.exhausted
        || analysis.outcome != 1
        || analysis.actions.is_empty()
        || analysis
            .actions
            .iter()
            .all(|item| item.outcome == analysis.outcome)
    {
        return search_best_action_with_tactical_filter(state, config);
    }

    let action = root_order
        .iter()
        .copied()
        .find(|action| analysis.optimal_actions.contains(action));
    let Some(action) = action else {
        return search_best_action_with_tactical_filter(state, config);
    };
    let score = match analysis.outcome {
        1 => WIN_SCORE - state.ply as i32,
        -1 => -WIN_SCORE + state.ply as i32,
        _ => 0,
    };
    SearchResult {
        action: Some(action),
        score,
        nodes: analysis.stats.nodes,
        exhausted: false,
        completed_depth: proof_horizon,
        table_hits: analysis.stats.cache_hits,
    }
}

/// Search for the best action while allowing an external policy/sorter to
/// provide a root ordering. The recursive alpha-beta evaluator remains the
/// authority; missing or illegal sorter actions fall back to Pathfinder's
/// deterministic heuristic ordering.
pub fn search_best_action_with_root_order(
    state: GameState,
    config: SearchConfig,
    root_order: &[Action],
) -> SearchResult {
    search_best_action_with_root_order_and_options(state, config, root_order, false)
}

/// Search with an external root order and optional one-ply tactical extension
/// at the normal depth horizon. The extension is deliberately opt-in so the
/// incumbent Pathfinder remains an unchanged control in strength experiments.
pub fn search_best_action_with_root_order_and_options(
    state: GameState,
    config: SearchConfig,
    root_order: &[Action],
    tactical_extension: bool,
) -> SearchResult {
    search_best_action_with_root_order_and_root_limit_internal(
        state,
        config,
        root_order,
        tactical_extension,
        None,
        false,
    )
}

/// Search with a sorter-provided root order and an optional root candidate
/// limit. A limited root is useful for hybrid Pathfinder agents: the sorter
/// chooses a small candidate set, while the recursive alpha-beta evaluator
/// spends the entire node budget comparing those candidates. The ordinary
/// Pathfinder entry point leaves this unset so its baseline behavior is
/// unchanged.
pub fn search_best_action_with_root_order_and_root_limit(
    state: GameState,
    config: SearchConfig,
    root_order: &[Action],
    tactical_extension: bool,
    root_limit: Option<usize>,
) -> SearchResult {
    search_best_action_with_root_order_and_root_limit_internal(
        state,
        config,
        root_order,
        tactical_extension,
        root_limit,
        true,
    )
}

/// Search with an external root order, root limit, and wall-clock deadline.
/// This is the deadline-aware counterpart used by research hybrid agents that
/// add a policy hint without replacing Pathfinder's recursive evaluator.
pub fn search_best_action_with_root_order_and_root_limit_deadline(
    state: GameState,
    config: SearchConfig,
    root_order: &[Action],
    tactical_extension: bool,
    root_limit: Option<usize>,
    deadline_ms: u32,
) -> SearchResult {
    search_best_action_with_root_order_and_root_limit_internal_deadline(
        state,
        config,
        root_order,
        tactical_extension,
        root_limit,
        true,
        Some(deadline_after_ms(deadline_ms.max(1))),
        None,
        None,
    )
}

/// Deadline-aware root-ordered search with coarse progress callbacks. This is
/// the transition-policy entry point used by the browser worker.
pub fn search_best_action_with_root_order_and_root_limit_deadline_progress(
    state: GameState,
    config: SearchConfig,
    root_order: &[Action],
    tactical_extension: bool,
    root_limit: Option<usize>,
    deadline_ms: u32,
    progress: SearchProgressCallback,
) -> SearchResult {
    search_best_action_with_root_order_and_root_limit_internal_deadline(
        state,
        config,
        root_order,
        tactical_extension,
        root_limit,
        true,
        Some(deadline_after_ms(deadline_ms.max(1))),
        Some(progress),
        None,
    )
}

/// Deadline-aware root-ordered search with root traces. This is the generic
/// hook used by hybrid agents whose policy supplies the root ordering.
pub fn search_best_action_with_root_order_and_root_limit_deadline_trace(
    state: GameState,
    config: SearchConfig,
    root_order: &[Action],
    tactical_extension: bool,
    root_limit: Option<usize>,
    deadline_ms: u32,
    progress: SearchProgressCallback,
    trace: SearchTraceCallback,
) -> SearchResult {
    search_best_action_with_root_order_and_root_limit_internal_deadline(
        state,
        config,
        root_order,
        tactical_extension,
        root_limit,
        true,
        Some(deadline_after_ms(deadline_ms.max(1))),
        Some(progress),
        Some(trace),
    )
}

/// Spend a bounded scout budget on the first root actions, then run the normal
/// Pathfinder search over the full root set ordered by those scout scores.
///
/// The scout is deliberately separate from the main search budget: its nodes
/// are charged against the same total ceiling, and the returned result reports
/// the combined count. This gives a pure-Rust control for testing whether a
/// shallow exact root probe is a better ordering signal than a learned sorter.
pub fn search_best_action_with_root_probe(
    state: GameState,
    config: SearchConfig,
    probe_depth: u8,
    probe_nodes: u64,
    probe_actions: usize,
) -> SearchResult {
    if probe_depth == 0 || probe_nodes == 0 || probe_actions == 0 {
        return search_best_action(state, config);
    }
    let probe_config = SearchConfig {
        depth: probe_depth,
        max_nodes: probe_nodes.min(config.max_nodes),
        ..config
    };
    let fallback = ordered_root_actions(state, state.turn, config.weights);
    if fallback.is_empty() {
        return search_best_action(state, config);
    }
    let analyses = analyze_actions(state, probe_config, probe_actions.min(fallback.len()));
    let consumed = analyses
        .last()
        .map_or(0, |result| result.nodes)
        .min(config.max_nodes);
    if analyses.is_empty() || consumed >= config.max_nodes {
        let best = analyses.first().copied().unwrap_or(MoveEvaluation {
            action: fallback[0],
            before_score: evaluate(state, state.turn, config.weights),
            score: evaluate(
                state.apply_legal(fallback[0]).state,
                state.turn,
                config.weights,
            ),
            delta: 0,
            nodes: consumed,
            exhausted: consumed >= config.max_nodes,
            completed_depth: probe_depth,
            table_hits: 0,
        });
        return SearchResult {
            action: Some(best.action),
            score: best.score,
            nodes: consumed,
            exhausted: consumed >= config.max_nodes,
            completed_depth: best.completed_depth,
            table_hits: best.table_hits,
        };
    }
    let mut root_order = analyses
        .iter()
        .map(|result| result.action)
        .collect::<Vec<_>>();
    for action in fallback {
        if !root_order.contains(&action) {
            root_order.push(action);
        }
    }
    let result = search_best_action_with_root_order_and_root_limit(
        state,
        SearchConfig {
            max_nodes: config.max_nodes - consumed,
            ..config
        },
        &root_order,
        false,
        None,
    );
    SearchResult {
        action: result.action,
        score: result.score,
        nodes: consumed.saturating_add(result.nodes),
        exhausted: result.exhausted,
        completed_depth: result.completed_depth,
        table_hits: result.table_hits,
    }
}

fn search_best_action_with_root_order_and_root_limit_internal(
    state: GameState,
    config: SearchConfig,
    root_order: &[Action],
    tactical_extension: bool,
    root_limit: Option<usize>,
    tt_move_order: bool,
) -> SearchResult {
    search_best_action_with_root_order_and_root_limit_internal_deadline(
        state,
        config,
        root_order,
        tactical_extension,
        root_limit,
        tt_move_order,
        None,
        None,
        None,
    )
}

fn search_best_action_with_root_order_and_root_limit_internal_deadline(
    state: GameState,
    config: SearchConfig,
    root_order: &[Action],
    tactical_extension: bool,
    root_limit: Option<usize>,
    tt_move_order: bool,
    deadline: Option<Deadline>,
    progress: Option<SearchProgressCallback>,
    mut trace: Option<SearchTraceCallback>,
) -> SearchResult {
    if state.config.board_size <= 4 {
        if let Some(horizon) = config.tactical_proof_horizon {
            let result = crate::endgame::search_best_action(
                state,
                crate::endgame::TacticalProofConfig {
                    horizon,
                    max_nodes: config.max_nodes,
                },
            );
            let score = match result.outcome {
                1 => WIN_SCORE - i32::from(state.ply),
                -1 => -WIN_SCORE + i32::from(state.ply),
                _ => 0,
            };
            return SearchResult {
                action: result.action,
                score,
                nodes: result.nodes,
                exhausted: result.exhausted,
                completed_depth: result.completed_depth,
                table_hits: result.table_hits,
            };
        }
    }
    let root_player = state.turn;
    let mut initial_actions = root_ordered_actions(state, root_player, config.weights, root_order);
    limit_root_actions(&mut initial_actions, root_limit);
    if initial_actions.is_empty() {
        return SearchResult {
            action: None,
            score: evaluate(state, root_player, config.weights),
            nodes: 0,
            exhausted: false,
            completed_depth: 0,
            table_hits: 0,
        };
    }

    let mut budget = Budget::with_deadline_and_progress(deadline, progress);
    let mut table = HashMap::new();
    let mut hints = SearchHints::default();
    let mut best_action = initial_actions[0];
    let mut best_score = NEG_INF;
    let mut completed_depth = 0;

    for depth in 1..=config.depth {
        let mut actions = root_ordered_actions(state, root_player, config.weights, root_order);
        limit_root_actions(&mut actions, root_limit);
        put_first(&mut actions, best_action);
        let mut iteration_action = actions[0];
        let mut iteration_score = NEG_INF;
        let mut alpha = NEG_INF;
        let mut complete = true;
        let mut iteration_candidates = Vec::with_capacity(actions.len());

        for action in actions {
            if budget.reached(config.max_nodes) {
                complete = false;
                break;
            }
            let next = state.apply_legal(action).state;
            budget.count_node();
            let mut score = minimax(
                next,
                root_player,
                depth - 1,
                alpha,
                POS_INF,
                config,
                tactical_extension,
                tt_move_order,
                1,
                &mut budget,
                &mut table,
                &mut hints,
            );
            // A child searched with the incumbent alpha can legally return an
            // upper bound equal to alpha after a minimizing cutoff. It is not
            // an exact tie, so do not let the root's action-order tie-break
            // replace a known incumbent with a potentially worse move. When
            // the candidate would win that tie-break, re-search it with a
            // full window before comparing the root actions.
            if alpha != NEG_INF && score == alpha && action.order() < iteration_action.order() {
                score = minimax(
                    next,
                    root_player,
                    depth - 1,
                    NEG_INF,
                    POS_INF,
                    config,
                    tactical_extension,
                    tt_move_order,
                    1,
                    &mut budget,
                    &mut table,
                    &mut hints,
                );
            }
            if score > iteration_score
                || (score == iteration_score && action.order() < iteration_action.order())
            {
                iteration_action = action;
                iteration_score = score;
            }
            iteration_candidates.push(RootSearchCandidate { action, score });
            alpha = alpha.max(iteration_score);
            if budget.exhausted {
                complete = false;
                break;
            }
        }
        if !complete {
            break;
        }
        let mut candidates = iteration_candidates;
        candidates.sort_by_key(|candidate| (-candidate.score, candidate.action.order()));
        if let Some(callback) = trace.as_mut() {
            callback(depth, budget.nodes, budget.table_hits, candidates);
        }
        best_action = iteration_action;
        best_score = iteration_score;
        completed_depth = depth;
    }

    if completed_depth == 0 {
        best_score = evaluate(
            state.apply_legal(best_action).state,
            root_player,
            config.weights,
        );
    }
    SearchResult {
        action: Some(best_action),
        score: best_score,
        nodes: budget.nodes,
        exhausted: budget.exhausted,
        completed_depth,
        table_hits: budget.table_hits,
    }
}

/// Expose Pathfinder's cheap deterministic root ordering to hybrid agents.
pub fn ordered_root_actions(
    state: GameState,
    root_player: Player,
    weights: EvaluationWeights,
) -> Vec<Action> {
    ordered_actions(state, root_player, weights)
}

/// Return Pathfinder's ordered root actions after removing moves that allow
/// the opponent to win immediately on the next turn. If there is no safe move
/// (a forced loss) or no risky move, the complete deterministic order is
/// returned so callers can retain the ordinary search behavior.
pub fn tactical_root_safe_actions(
    state: GameState,
    root_player: Player,
    weights: EvaluationWeights,
) -> Vec<Action> {
    let fallback = ordered_actions(state, root_player, weights);
    if fallback.is_empty() || state.legal_action_count() > 512 {
        return fallback;
    }
    let opponent = root_player.other();
    let mut safe = Vec::with_capacity(fallback.len());
    let mut risky = false;
    for action in fallback.iter().copied() {
        let next = state.apply_legal(action).state;
        let allows_win = if next.winner == Some(root_player) {
            false
        } else {
            let opponent_view = if next.turn == opponent {
                next
            } else {
                GameState {
                    turn: opponent,
                    ..next
                }
            };
            !immediate_winning_actions(opponent_view, opponent).is_empty()
        };
        if allows_win {
            risky = true;
        } else {
            safe.push(action);
        }
    }
    if safe.is_empty() || !risky {
        fallback
    } else {
        safe
    }
}

/// Pathfinder's deterministic ordering with a bounded tactical guard.
///
/// The ordinary evaluator already puts captures and wins first, but it can
/// miss the quieter move that removes an opponent's immediate winning reply.
/// When the position is small enough to inspect cheaply, move such forced
/// blocks to the front of the root list. This is intentionally an ordering
/// hint only: the alpha-beta search still evaluates every candidate in its
/// configured beam and remains the authority on the result.
pub fn ordered_root_actions_with_tactical_guard(
    state: GameState,
    root_player: Player,
    weights: EvaluationWeights,
) -> Vec<Action> {
    let fallback = ordered_actions(state, root_player, weights);
    if fallback.is_empty() || state.legal_action_count() > 512 {
        return fallback;
    }

    let opponent = root_player.other();
    let opponent_view = if state.turn == opponent {
        state
    } else {
        GameState {
            turn: opponent,
            ..state
        }
    };
    let baseline_threats = immediate_winning_actions(opponent_view, opponent);

    let mut guarded = Vec::with_capacity(fallback.len());
    let mut has_threat = !baseline_threats.is_empty();
    for action in fallback.iter().copied() {
        let next = state.apply_legal(action).state;
        let next_opponent_view = if next.turn == opponent {
            next
        } else {
            GameState {
                turn: opponent,
                ..next
            }
        };
        let threats = immediate_winning_actions(next_opponent_view, opponent);
        if threats.is_empty() {
            guarded.push(action);
        } else {
            has_threat = true;
        }
    }
    // Keep the incumbent order exactly when no root move creates or leaves an
    // immediate opponent win. Otherwise, put all threat-free moves first and
    // retain the original order for the risky suffix.
    if !has_threat || guarded.is_empty() {
        return fallback;
    }
    for action in fallback {
        if !guarded.contains(&action) {
            guarded.push(action);
        }
    }
    guarded
}

fn immediate_winning_actions(state: GameState, player: Player) -> Vec<Action> {
    if state.winner.is_some() || state.turn != player {
        return Vec::new();
    }
    state
        .legal_actions()
        .into_iter()
        .filter(|action| state.apply_legal(*action).state.winner == Some(player))
        .collect()
}

fn has_tactical_signal(state: GameState) -> bool {
    let own_tactic = state.legal_actions().iter().copied().any(|action| {
        let transition = state.apply_legal(action);
        transition.state.winner == Some(state.turn) || transition.captured.count_ones() >= 2
    });
    if own_tactic {
        return true;
    }
    let mut opponent_view = state;
    opponent_view.turn = state.turn.other();
    opponent_view
        .legal_actions()
        .iter()
        .copied()
        .any(|action| opponent_view.apply_legal(action).state.winner == Some(opponent_view.turn))
}

pub fn analyze_action(
    state: GameState,
    action: Action,
    config: SearchConfig,
) -> Result<MoveEvaluation, String> {
    if !state.legal_actions().contains(&action) {
        return Err("cannot analyze an illegal Pathagon action".to_owned());
    }
    let root_player = state.turn;
    let before_score = evaluate(state, root_player, config.weights);
    let mut budget = Budget::default();
    let mut table = HashMap::new();
    let mut hints = SearchHints::default();
    let next = state.apply_legal(action).state;
    budget.count_node();
    let score = if next.winner.is_some() {
        evaluate(next, root_player, config.weights)
    } else {
        minimax(
            next,
            root_player,
            config.depth.saturating_sub(1),
            NEG_INF,
            POS_INF,
            config,
            false,
            false,
            1,
            &mut budget,
            &mut table,
            &mut hints,
        )
    };
    Ok(MoveEvaluation {
        action,
        before_score,
        score,
        delta: score - before_score,
        nodes: budget.nodes,
        exhausted: budget.exhausted,
        completed_depth: config.depth,
        table_hits: budget.table_hits,
    })
}

pub fn analyze_actions(
    state: GameState,
    config: SearchConfig,
    max_actions: usize,
) -> Vec<MoveEvaluation> {
    let root_player = state.turn;
    let before_score = evaluate(state, root_player, config.weights);
    let mut budget = Budget::default();
    let mut table = HashMap::new();
    let mut hints = SearchHints::default();
    let mut alpha = NEG_INF;
    let mut results = Vec::new();
    let mut complete = true;
    for action in ordered_actions(state, root_player, config.weights)
        .into_iter()
        .take(max_actions)
    {
        if budget.reached(config.max_nodes) {
            complete = false;
        }
        if !complete {
            break;
        }
        let next = state.apply_legal(action).state;
        budget.count_node();
        let score = if next.winner.is_some() {
            evaluate(next, root_player, config.weights)
        } else {
            minimax(
                next,
                root_player,
                config.depth.saturating_sub(1),
                alpha,
                POS_INF,
                config,
                false,
                false,
                1,
                &mut budget,
                &mut table,
                &mut hints,
            )
        };
        results.push(MoveEvaluation {
            action,
            before_score,
            score,
            delta: score - before_score,
            nodes: budget.nodes,
            exhausted: budget.exhausted,
            completed_depth: config.depth,
            table_hits: budget.table_hits,
        });
        alpha = alpha.max(score);
        if budget.exhausted {
            break;
        }
    }
    results.sort_by_key(|result| (-result.score, result.action.order()));
    results
}

/// Choose the same shallow local-pattern move as the browser Lunatic baseline.
/// It deliberately evaluates each legal action once without considering the
/// opponent's reply, making it a cheap breadth opponent for large arenas.
pub fn lunatic_action(state: GameState) -> SearchResult {
    let actions = state.legal_actions();
    if actions.is_empty() {
        return SearchResult {
            action: None,
            score: evaluate(state, state.turn, EvaluationWeights::default()),
            nodes: 0,
            exhausted: false,
            completed_depth: 0,
            table_hits: 0,
        };
    }
    let player = state.turn;
    let before_own_distance = connection_distance(state, player);
    let before_opponent_distance = connection_distance(state, player.other());
    let mut best_action = actions[0];
    let mut best_score = NEG_INF;
    for action in actions.iter().copied() {
        let transition = state.apply_legal(action);
        let score = if transition.state.winner == Some(player) {
            WIN_SCORE
        } else {
            let own_distance = connection_distance(transition.state, player);
            let opponent_distance = connection_distance(transition.state, player.other());
            transition.captured.count_ones() as i32 * 10_000
                + (before_own_distance - own_distance) * 500
                + (opponent_distance - before_opponent_distance) * 350
                + if matches!(action, Action::Relocate { .. }) {
                    10
                } else {
                    0
                }
        };
        if score > best_score || (score == best_score && action.order() < best_action.order()) {
            best_action = action;
            best_score = score;
        }
    }
    SearchResult {
        action: Some(best_action),
        score: best_score,
        nodes: actions.len() as u64,
        exhausted: false,
        completed_depth: 1,
        table_hits: 0,
    }
}

pub fn evaluate(state: GameState, player: Player, weights: EvaluationWeights) -> i32 {
    if state.winner == Some(player) {
        return WIN_SCORE - state.ply as i32;
    }
    let opponent = player.other();
    if state.winner == Some(opponent) {
        return -WIN_SCORE + state.ply as i32;
    }
    let path = connection_distance(state, opponent) - connection_distance(state, player);
    let material =
        state.pieces(player).count_ones() as i32 - state.pieces(opponent).count_ones() as i32;
    let capture_direction = if state.last_player == Some(player) {
        1
    } else {
        -1
    };
    let structure = largest_component(state, player) - largest_component(state, opponent);
    let threats = capture_opportunities(state, player) - capture_opportunities(state, opponent);
    let edges = edge_presence(state, player) - edge_presence(state, opponent);
    path * weights.path
        + material * weights.material
        + capture_direction * state.last_capture as i32 * weights.capture
        + structure * weights.structure
        + threats * weights.threat
        + edges * weights.edge
}

fn minimax(
    state: GameState,
    root_player: Player,
    depth: u8,
    mut alpha: i32,
    mut beta: i32,
    config: SearchConfig,
    tactical_extension: bool,
    tt_move_order: bool,
    ply_from_root: usize,
    budget: &mut Budget,
    table: &mut HashMap<(GameState, Player), TableEntry>,
    hints: &mut SearchHints,
) -> i32 {
    if state.winner.is_some() {
        return evaluate(state, root_player, config.weights);
    }
    if depth == 0 {
        if tactical_extension {
            if let Some(action) = state
                .legal_actions()
                .into_iter()
                .find(|action| state.apply_legal(*action).state.winner == Some(state.turn))
            {
                if !budget.reached(config.max_nodes) {
                    budget.count_node();
                    let score =
                        evaluate(state.apply_legal(action).state, root_player, config.weights);
                    budget.reached(config.max_nodes);
                    return score;
                }
            }
        }
        return evaluate(state, root_player, config.weights);
    }
    if budget.reached(config.max_nodes) {
        return evaluate(state, root_player, config.weights);
    }

    let key = (state, root_player);
    let original_alpha = alpha;
    let original_beta = beta;
    let mut preferred_action = None;
    if let Some(entry) = table.get(&key).copied() {
        preferred_action = entry.best_action;
        if entry.depth >= depth {
            budget.table_hits += 1;
            match entry.bound {
                Bound::Exact => return entry.score,
                Bound::Lower => alpha = alpha.max(entry.score),
                Bound::Upper => beta = beta.min(entry.score),
            }
            if alpha >= beta {
                return entry.score;
            }
        }
    }

    let maximizing = state.turn == root_player;
    let mut actions = ordered_actions(state, root_player, config.weights);
    if tt_move_order {
        if let Some(preferred) = preferred_action {
            put_first(&mut actions, preferred);
        }
        let [killer_one, killer_two] = hints.killers_at(ply_from_root);
        if let Some(killer) = killer_two {
            put_first(&mut actions, killer);
        }
        if let Some(killer) = killer_one {
            put_first(&mut actions, killer);
        }
        actions.sort_by(|left, right| {
            hints
                .history_at(ply_from_root, *right, maximizing)
                .cmp(&hints.history_at(ply_from_root, *left, maximizing))
        });
    }
    actions.truncate(config.beam_width);
    if actions.is_empty() {
        return evaluate(state, root_player, config.weights);
    }
    let mut best = if maximizing { NEG_INF } else { POS_INF };
    let mut best_action = actions[0];
    for action in actions {
        if budget.reached(config.max_nodes) {
            break;
        }
        let next = state.apply_legal(action).state;
        budget.count_node();
        let score = minimax(
            next,
            root_player,
            depth - 1,
            alpha,
            beta,
            config,
            tactical_extension,
            tt_move_order,
            ply_from_root + 1,
            budget,
            table,
            hints,
        );
        if maximizing {
            if score > best || (score == best && action.order() < best_action.order()) {
                best = score;
                best_action = action;
            }
            alpha = alpha.max(best);
        } else {
            if score < best || (score == best && action.order() < best_action.order()) {
                best = score;
                best_action = action;
            }
            beta = beta.min(best);
        }
        if beta <= alpha || budget.reached(config.max_nodes) {
            if beta <= alpha && next.winner.is_none() && next.last_capture == 0 {
                hints.record_history(ply_from_root, action, maximizing, depth);
                hints.record_killer(ply_from_root, action);
            }
            break;
        }
    }
    if !budget.exhausted {
        let bound = if best <= original_alpha {
            Bound::Upper
        } else if best >= original_beta {
            Bound::Lower
        } else {
            Bound::Exact
        };
        table.insert(
            key,
            TableEntry {
                depth,
                score: best,
                bound,
                best_action: Some(best_action),
            },
        );
    }
    budget.reached(config.max_nodes);
    best
}

fn ordered_actions(
    state: GameState,
    root_player: Player,
    weights: EvaluationWeights,
) -> Vec<Action> {
    let maximizing = state.turn == root_player;
    let mut scored: Vec<(Action, i32)> = state
        .legal_actions()
        .into_iter()
        .map(|action| {
            let transition = state.apply_legal(action);
            let score = if transition.state.winner == Some(state.turn) {
                2_000_000_000
            } else {
                transition.captured.count_ones() as i32 * 10_000
                    + evaluate(transition.state, root_player, weights)
            };
            (action, score)
        })
        .collect();
    scored.sort_by(|left, right| {
        let score_order = if maximizing {
            right.1.cmp(&left.1)
        } else {
            left.1.cmp(&right.1)
        };
        score_order.then_with(|| left.0.order().cmp(&right.0.order()))
    });
    scored.into_iter().map(|(action, _)| action).collect()
}

fn root_ordered_actions(
    state: GameState,
    root_player: Player,
    weights: EvaluationWeights,
    root_order: &[Action],
) -> Vec<Action> {
    let fallback = ordered_actions(state, root_player, weights);
    if root_order.is_empty() {
        return fallback;
    }
    let mut merged = Vec::with_capacity(fallback.len());
    for action in root_order.iter().copied() {
        if fallback.contains(&action) && !merged.contains(&action) {
            merged.push(action);
        }
    }
    for action in fallback {
        if !merged.contains(&action) {
            merged.push(action);
        }
    }
    merged
}

fn limit_root_actions(actions: &mut Vec<Action>, root_limit: Option<usize>) {
    if let Some(limit) = root_limit {
        actions.truncate(limit.max(1));
    }
}

pub(crate) fn connection_distance(state: GameState, player: Player) -> i32 {
    let opponent = player.other();
    let board_size = state.config.board_size;
    let cell_count = state.config.cells();
    let own_pieces = state.pieces(player);
    let opponent_pieces = state.pieces(opponent);
    let far_edge = if player == Player::Light {
        row_mask(board_size, 0)
    } else {
        column_mask(board_size, board_size - 1)
    };
    let mut distance = [u8::MAX; MAX_CELL_COUNT as usize];
    let mut buckets = [0_u64; MAX_CELL_COUNT as usize + 1];
    for index in 0..board_size {
        let square = if player == Player::Light {
            (board_size - 1) * board_size + index
        } else {
            index * board_size
        };
        let square_bit = bit(square);
        if opponent_pieces & square_bit != 0 {
            continue;
        }
        let cost = u8::from(own_pieces & square_bit == 0);
        distance[square as usize] = cost;
        buckets[cost as usize] |= square_bit;
    }
    let mut current_distance = 0_usize;
    while current_distance <= cell_count as usize {
        while buckets[current_distance] != 0 {
            let square = buckets[current_distance].trailing_zeros() as u8;
            buckets[current_distance] &= !bit(square);
            if bit(square) & far_edge != 0 {
                return current_distance as i32;
            }
            for next in squares(neighbor_mask_for(board_size, square)) {
                let next_bit = bit(next);
                if opponent_pieces & next_bit != 0 {
                    continue;
                }
                let step = u8::from(own_pieces & next_bit == 0);
                let next_distance = current_distance as u8 + step;
                if next_distance >= distance[next as usize] {
                    continue;
                }
                let previous_distance = distance[next as usize];
                if previous_distance != u8::MAX {
                    buckets[previous_distance as usize] &= !next_bit;
                }
                distance[next as usize] = next_distance;
                buckets[next_distance as usize] |= next_bit;
            }
        }
        current_distance += 1;
    }
    cell_count as i32
}

pub(crate) fn largest_component(state: GameState, player: Player) -> i32 {
    let mut remaining = state.pieces(player);
    let mut largest = 0;
    while let Some(first) = squares(remaining).next() {
        let mut stack = [0_u8; MAX_CELL_COUNT as usize];
        let mut stack_len = 1_usize;
        stack[0] = first;
        remaining &= !bit(first);
        let mut size = 0;
        while stack_len != 0 {
            stack_len -= 1;
            let square = stack[stack_len];
            size += 1;
            let adjacent = neighbor_mask_for(state.config.board_size, square) & remaining;
            for next in squares(adjacent) {
                remaining &= !bit(next);
                stack[stack_len] = next;
                stack_len += 1;
            }
        }
        largest = largest.max(size);
    }
    largest
}

pub(crate) fn capture_opportunities(state: GameState, player: Player) -> i32 {
    let occupied = state.light | state.dark | state.forbidden;
    let mut victims = 0;
    for origin in 0..state.config.cells() {
        if occupied & bit(origin) != 0 {
            continue;
        }
        victims |= captures_from(state, origin, player);
    }
    victims.count_ones() as i32
}

pub(crate) fn edge_presence(state: GameState, player: Player) -> i32 {
    let board_size = state.config.board_size;
    let mut near = false;
    let mut far = false;
    for index in 0..board_size {
        let near_square = if player == Player::Light {
            (board_size - 1) * board_size + index
        } else {
            index * board_size
        };
        let far_square = if player == Player::Light {
            index
        } else {
            index * board_size + board_size - 1
        };
        near |= state.board_at(near_square) == Some(player);
        far |= state.board_at(far_square) == Some(player);
    }
    i32::from(near) + i32::from(far)
}

fn put_first(actions: &mut Vec<Action>, preferred: Action) {
    if let Some(index) = actions.iter().position(|action| *action == preferred) {
        actions.swap(0, index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    fn exhaustive_beam_score(
        state: GameState,
        root_player: Player,
        depth: u8,
        config: SearchConfig,
    ) -> i32 {
        if state.winner.is_some() || depth == 0 {
            return evaluate(state, root_player, config.weights);
        }
        let actions = ordered_actions(state, root_player, config.weights);
        if actions.is_empty() {
            return evaluate(state, root_player, config.weights);
        }
        let maximizing = state.turn == root_player;
        let mut best = if maximizing { NEG_INF } else { POS_INF };
        for action in actions.into_iter().take(config.beam_width) {
            let score = exhaustive_beam_score(
                state.apply_legal(action).state,
                root_player,
                depth - 1,
                config,
            );
            if maximizing {
                best = best.max(score);
            } else {
                best = best.min(score);
            }
        }
        if best == NEG_INF || best == POS_INF {
            evaluate(state, root_player, config.weights)
        } else {
            best
        }
    }

    fn exhaustive_beam_best_action(state: GameState, config: SearchConfig) -> (Action, i32) {
        let root_player = state.turn;
        let actions = ordered_actions(state, root_player, config.weights);
        let mut best_action = actions[0];
        let mut best_score = NEG_INF;
        for action in actions {
            let score = exhaustive_beam_score(
                state.apply_legal(action).state,
                root_player,
                config.depth.saturating_sub(1),
                config,
            );
            if score > best_score || (score == best_score && action.order() < best_action.order()) {
                best_action = action;
                best_score = score;
            }
        }
        (best_action, best_score)
    }

    fn exhaustive_root_order_best_action(
        state: GameState,
        config: SearchConfig,
        root_order: &[Action],
    ) -> (Action, i32) {
        let root_player = state.turn;
        let actions = root_ordered_actions(state, root_player, config.weights, root_order);
        let mut best_action = actions[0];
        let mut best_score = NEG_INF;
        for action in actions {
            let score = exhaustive_beam_score(
                state.apply_legal(action).state,
                root_player,
                config.depth.saturating_sub(1),
                config,
            );
            if score > best_score || (score == best_score && action.order() < best_action.order()) {
                best_action = action;
                best_score = score;
            }
        }
        (best_action, best_score)
    }

    fn next_test_seed(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *seed
    }

    fn sampled_nonterminal_position(board_size: u8, seed: u64, target_plies: usize) -> GameState {
        for attempt in 0..256_u64 {
            let mut state = GameState::with_board_size(board_size);
            let mut cursor = seed.wrapping_add(attempt);
            for _ in 0..target_plies {
                if state.winner.is_some() {
                    break;
                }
                let actions = state.legal_actions();
                if actions.is_empty() {
                    break;
                }
                let action = actions[(next_test_seed(&mut cursor) as usize) % actions.len()];
                state = state.apply_legal(action).state;
            }
            if state.winner.is_none() && !state.legal_actions().is_empty() {
                return state;
            }
        }
        panic!(
            "could not sample a non-terminal {board_size}x{board_size} position after {target_plies} plies"
        );
    }

    fn search_test_weights(index: usize) -> EvaluationWeights {
        match index % 4 {
            0 => EvaluationWeights::default(),
            1 => EvaluationWeights {
                path: 241,
                material: 112,
                capture: 887,
                structure: 40,
                threat: 154,
                edge: 74,
            },
            2 => EvaluationWeights {
                path: 1,
                material: 1,
                capture: 1,
                structure: 1,
                threat: 1,
                edge: 1,
            },
            _ => EvaluationWeights {
                path: -240,
                material: 0,
                capture: 1_100,
                structure: -55,
                threat: 260,
                edge: 0,
            },
        }
    }

    fn assert_search_matches_reference(state: GameState, config: SearchConfig, context: &str) {
        let (expected_action, expected_score) = exhaustive_beam_best_action(state, config);
        let actual = search_best_action(state, config);
        assert!(
            !actual.exhausted,
            "{context}: search exhausted its test budget"
        );
        assert_eq!(
            actual.completed_depth, config.depth,
            "{context}: incomplete depth"
        );
        assert_eq!(
            actual.action,
            Some(expected_action),
            "{context}: action mismatch; expected {expected_action:?}, got {:?}",
            actual.action
        );
        assert_eq!(
            actual.score, expected_score,
            "{context}: score mismatch for {:?}",
            actual.action
        );
        assert!(
            actual
                .action
                .is_some_and(|action| state.legal_actions().contains(&action)),
            "{context}: search returned an illegal action"
        );
    }

    fn reference_connection_distance(state: GameState, player: Player) -> i32 {
        let opponent = player.other();
        let board_size = state.config.board_size;
        let cell_count = state.config.cells();
        let mut distance = vec![u8::MAX; cell_count as usize];
        let mut queue = VecDeque::new();
        for index in 0..board_size {
            let square = if player == Player::Light {
                (board_size - 1) * board_size + index
            } else {
                index * board_size
            };
            if state.board_at(square) == Some(opponent) {
                continue;
            }
            let cost = u8::from(state.board_at(square) != Some(player));
            distance[square as usize] = cost;
            if cost == 0 {
                queue.push_front(square);
            } else {
                queue.push_back(square);
            }
        }
        while let Some(square) = queue.pop_front() {
            let row = square / board_size;
            let column = square % board_size;
            if (player == Player::Light && row == 0)
                || (player == Player::Dark && column == board_size - 1)
            {
                return distance[square as usize] as i32;
            }
            for next in squares(neighbor_mask_for(board_size, square)) {
                if state.board_at(next) == Some(opponent) {
                    continue;
                }
                let step = u8::from(state.board_at(next) != Some(player));
                let next_distance = distance[square as usize].saturating_add(step);
                if next_distance >= distance[next as usize] {
                    continue;
                }
                distance[next as usize] = next_distance;
                if step == 0 {
                    queue.push_front(next);
                } else {
                    queue.push_back(next);
                }
            }
        }
        cell_count as i32
    }

    #[test]
    fn allocation_free_connection_distance_matches_reference_search() {
        for board_size in crate::MIN_BOARD_SIZE..=crate::MAX_BOARD_SIZE {
            let mut state = GameState::with_board_size(board_size);
            let mut seed = u32::from(board_size).wrapping_mul(0x243f_6a88);
            for _ in 0..96 {
                for player in [Player::Light, Player::Dark] {
                    assert_eq!(
                        connection_distance(state, player),
                        reference_connection_distance(state, player),
                        "connection distance mismatch on {board_size}x{board_size} for {}",
                        player.as_str()
                    );
                }
                let actions = state.legal_actions();
                let Some(action) = actions.get((seed as usize) % actions.len().max(1)).copied()
                else {
                    break;
                };
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                state = state.apply_legal(action).state;
                if state.winner.is_some() {
                    break;
                }
            }
        }
    }

    #[test]
    fn iterative_search_respects_budget() {
        let result = search_best_action(
            GameState::new(),
            SearchConfig {
                depth: 5,
                max_nodes: 120,
                beam_width: 49,
                ..SearchConfig::default()
            },
        );
        assert!(result.action.is_some());
        assert!(result.nodes <= 120);
        assert!(result.exhausted);
        assert!((1..5).contains(&result.completed_depth));
    }

    #[test]
    fn alpha_beta_matches_exhaustive_beam_reference_on_small_positions() {
        let profiles = [(2_u8, 2_usize), (3, 2), (3, 4), (4, 8)];
        for (profile_index, (depth, beam_width)) in profiles.into_iter().enumerate() {
            let config = SearchConfig {
                depth,
                max_nodes: 200_000,
                beam_width,
                ..SearchConfig::default()
            };
            let mut state = GameState::with_board_size(3);
            let mut seed = 0x9e37_79b9_u32.wrapping_add(profile_index as u32);
            for position_index in 0..16 {
                if state.legal_actions().is_empty() {
                    state = GameState::with_board_size(3);
                    continue;
                }
                let (expected_action, expected_score) = exhaustive_beam_best_action(state, config);
                let actual = search_best_action(state, config);
                assert!(!actual.exhausted);
                assert_eq!(actual.completed_depth, depth);
                if actual.action != Some(expected_action) || actual.score != expected_score {
                    let breakdown = ordered_actions(state, state.turn, config.weights)
                        .into_iter()
                        .map(|action| {
                            let score = exhaustive_beam_score(
                                state.apply_legal(action).state,
                                state.turn,
                                config.depth.saturating_sub(1),
                                config,
                            );
                            format!("{action:?}={score}")
                        })
                        .collect::<Vec<_>>();
                    eprintln!(
                        "search/reference mismatch profile={depth}/{beam_width} position={position_index} state={state:?} expected={expected_action:?}/{expected_score} actual={:?}/{} nodes={} hits={} breakdown={breakdown:?}",
                        actual.action,
                        actual.score,
                        actual.nodes,
                        actual.table_hits
                    );
                }
                assert_eq!(actual.action, Some(expected_action));
                assert_eq!(actual.score, expected_score);

                let actions = state.legal_actions();
                let action = actions[(seed as usize) % actions.len()];
                seed = seed
                    .wrapping_mul(1_664_525)
                    .wrapping_add(1_013_904_223)
                    .wrapping_add(position_index);
                state = state.apply_legal(action).state;
                if state.winner.is_some() {
                    state = GameState::with_board_size(3);
                }
            }
        }
    }

    #[test]
    fn alpha_beta_matches_reference_across_reachable_position_matrix() {
        let profiles = [
            (1_u8, 1_usize),
            (1, 4),
            (2, 1),
            (2, 4),
            (3, 2),
            (3, 8),
            (4, 2),
            (4, 8),
            (5, 1),
            (5, 4),
            (5, 8),
        ];
        let samples = [
            (3_u8, 0_usize),
            (3, 1),
            (3, 2),
            (3, 4),
            (3, 6),
            (3, 8),
            (3, 10),
            (3, 12),
            (4, 0),
            (4, 2),
            (4, 5),
            (4, 8),
            (4, 12),
            (4, 16),
            (5, 0),
            (5, 3),
            (5, 6),
            (5, 10),
        ];

        for (position_index, (board_size, target_plies)) in samples.into_iter().enumerate() {
            let state = sampled_nonterminal_position(
                board_size,
                0x9e37_79b9_7f4a_7c15_u64.wrapping_add(position_index as u64 * 97),
                target_plies,
            );
            for (profile_index, (depth, beam_width)) in profiles.into_iter().enumerate() {
                let config = SearchConfig {
                    depth,
                    max_nodes: 5_000_000,
                    beam_width,
                    weights: search_test_weights(position_index + profile_index),
                    ..SearchConfig::default()
                };
                let context = format!(
                    "matrix position={position_index} board={board_size} plies={target_plies} profile={depth}/{beam_width}"
                );
                assert_search_matches_reference(state, config, &context);
            }
        }
    }

    #[test]
    fn root_ordering_cannot_change_a_completed_search_result() {
        for position_index in 0..16_usize {
            let state = sampled_nonterminal_position(
                3,
                0xa5a5_5a5a_1234_5678_u64.wrapping_add(position_index as u64 * 131),
                [0_usize, 1, 3, 5, 7, 9, 11, 13][position_index % 8],
            );
            let config = SearchConfig {
                depth: 4,
                max_nodes: 2_000_000,
                beam_width: 8,
                weights: search_test_weights(position_index),
                ..SearchConfig::default()
            };
            let legal = state.legal_actions();
            let reversed = legal.iter().rev().copied().collect::<Vec<_>>();
            let mut rotated = legal.clone();
            let rotation = position_index % rotated.len();
            rotated.rotate_left(rotation);
            let mut noisy = reversed.clone();
            noisy.push(reversed[0]);
            noisy.push(Action::Place { to: 63 });

            for (order_index, root_order) in [&reversed[..], &rotated[..], &noisy[..]]
                .into_iter()
                .enumerate()
            {
                let expected = exhaustive_root_order_best_action(state, config, root_order);
                let actual = search_best_action_with_root_order(state, config, root_order);
                assert!(
                    !actual.exhausted,
                    "position={position_index} order={order_index}"
                );
                assert_eq!(actual.completed_depth, config.depth);
                assert_eq!(
                    (actual.action, actual.score),
                    (Some(expected.0), expected.1),
                    "root order changed result at position={position_index} order={order_index}"
                );
            }
        }
    }

    #[test]
    fn tt_move_ordering_matches_the_exhaustive_value_when_the_beam_is_complete() {
        for position_index in 0..16_usize {
            let state = sampled_nonterminal_position(
                3,
                0x0123_4567_89ab_cdef_u64.wrapping_add(position_index as u64 * 173),
                [0_usize, 2, 4, 6, 8, 10, 12, 12][position_index % 8],
            );
            let config = SearchConfig {
                depth: 5,
                max_nodes: 5_000_000,
                beam_width: 64,
                weights: search_test_weights(position_index + 1),
                ..SearchConfig::default()
            };
            let expected = exhaustive_beam_best_action(state, config);
            let actual = search_best_action_with_tt_order(state, config);
            assert!(
                !actual.exhausted,
                "TT search exhausted at position={position_index}"
            );
            assert_eq!(actual.completed_depth, config.depth);
            assert_eq!(
                (actual.action, actual.score),
                (Some(expected.0), expected.1),
                "TT move ordering changed the completed result at position={position_index}"
            );
        }
    }

    #[test]
    fn bounded_search_respects_budget_and_keeps_results_inside_search_bounds() {
        let states = [
            sampled_nonterminal_position(3, 0x1111_2222_3333_4444, 6),
            sampled_nonterminal_position(4, 0x5555_6666_7777_8888, 9),
            sampled_nonterminal_position(5, 0x9999_aaaa_bbbb_cccc, 12),
        ];
        for (state_index, state) in states.into_iter().enumerate() {
            for depth in [1_u8, 3, 5] {
                for beam_width in [0_usize, 1, 2, 8, 64] {
                    for max_nodes in [0_u64, 1, 2, 7, 31, 120, 512, 2_000] {
                        let config = SearchConfig {
                            depth,
                            max_nodes,
                            beam_width,
                            weights: search_test_weights(state_index + depth as usize),
                            ..SearchConfig::default()
                        };
                        let result = search_best_action(state, config);
                        assert!(
                            result.nodes <= max_nodes,
                            "budget overrun state={state_index} depth={depth} beam={beam_width} budget={max_nodes}: {}",
                            result.nodes
                        );
                        assert!(result.completed_depth <= depth);
                        assert!(result.score > NEG_INF && result.score < POS_INF);
                        assert!(
                            result
                                .action
                                .is_some_and(|action| state.legal_actions().contains(&action)),
                            "invalid fallback state={state_index} depth={depth} beam={beam_width} budget={max_nodes}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn analyze_actions_returns_a_sorted_unique_legal_prefix() {
        for position_index in 0..12_usize {
            let state = sampled_nonterminal_position(
                4,
                0xdead_beef_cafe_babe_u64.wrapping_add(position_index as u64 * 211),
                [0_usize, 1, 3, 5, 7, 9][position_index % 6],
            );
            let config = SearchConfig {
                depth: 3,
                max_nodes: 2_000_000,
                beam_width: 8,
                weights: search_test_weights(position_index + 2),
                ..SearchConfig::default()
            };
            let legal = state.legal_actions();
            for requested in [0_usize, 1, 3, legal.len(), legal.len() + 5] {
                let analyses = analyze_actions(state, config, requested);
                assert_eq!(analyses.len(), requested.min(legal.len()));
                assert!(analyses.iter().all(|item| legal.contains(&item.action)));
                let unique = analyses
                    .iter()
                    .map(|item| item.action)
                    .collect::<std::collections::HashSet<_>>();
                assert_eq!(unique.len(), analyses.len());
                assert!(analyses.windows(2).all(|pair| {
                    pair[0].score > pair[1].score
                        || (pair[0].score == pair[1].score
                            && pair[0].action.order() < pair[1].action.order())
                }));
                assert!(analyses
                    .iter()
                    .all(|item| item.score > NEG_INF && item.score < POS_INF));
            }
        }
    }

    #[test]
    fn completed_search_never_exposes_internal_bound_as_a_score() {
        // This is the first Extra High game position where the referee
        // previously published POS_INF (536_870_911) as Pathfinder's score.
        let mut state = GameState::new();
        for to in [
            44_u8, 9, 10, 2, 3, 8, 17, 11, 7, 10, 13, 6, 20, 5, 1, 15, 14, 0, 16, 2, 4, 18,
        ] {
            let action = Action::Place { to };
            assert!(state.legal_actions().contains(&action));
            state = state.apply_legal(action).state;
        }
        let result = search_best_action(
            state,
            SearchConfig {
                depth: 5,
                max_nodes: 2_000,
                beam_width: 8,
                ..SearchConfig::default()
            },
        );
        assert_ne!(result.score, NEG_INF);
        assert_ne!(result.score, POS_INF);
    }

    #[test]
    fn root_probe_respects_the_shared_budget() {
        let state = GameState::new();
        let result = search_best_action_with_root_probe(
            state,
            SearchConfig {
                depth: 4,
                max_nodes: 120,
                beam_width: 8,
                ..SearchConfig::default()
            },
            2,
            24,
            8,
        );
        assert!(result.action.is_some());
        assert!(result
            .action
            .is_some_and(|action| state.legal_actions().contains(&action)));
        assert!(result.nodes <= 120);
    }

    #[test]
    fn tt_order_preserves_the_full_root_set() {
        let state = GameState::new();
        let result = search_best_action_with_tt_order(
            state,
            SearchConfig {
                depth: 3,
                max_nodes: 120,
                beam_width: 8,
                ..SearchConfig::default()
            },
        );
        assert!(result.action.is_some());
        assert!(result
            .action
            .is_some_and(|action| state.legal_actions().contains(&action)));
        assert!(result.nodes <= 120);
    }

    #[test]
    fn tactical_guard_preserves_the_full_root_set() {
        let state = GameState::new();
        let result = search_best_action_with_tactical_guard(
            state,
            SearchConfig {
                depth: 3,
                max_nodes: 120,
                beam_width: 8,
                ..SearchConfig::default()
            },
        );
        assert!(result.action.is_some());
        assert!(result
            .action
            .is_some_and(|action| state.legal_actions().contains(&action)));
        assert!(result.nodes <= 120);
    }

    #[test]
    fn tactical_filter_keeps_a_legal_safe_root_or_falls_back() {
        let state = GameState::new();
        let result = search_best_action_with_tactical_filter(
            state,
            SearchConfig {
                depth: 3,
                max_nodes: 120,
                beam_width: 8,
                ..SearchConfig::default()
            },
        );
        assert!(result.action.is_some());
        assert!(result
            .action
            .is_some_and(|action| state.legal_actions().contains(&action)));
        assert!(result.nodes <= 120);
    }

    #[test]
    fn deadline_returns_a_legal_last_completed_move() {
        let bits = |squares: &[u8]| {
            squares
                .iter()
                .fold(0_u64, |mask, square| mask | (1_u64 << square))
        };
        let state = GameState {
            config: crate::BoardConfig::DEFAULT,
            light: bits(&[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]),
            dark: bits(&[28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 29, 31, 33]),
            reserve: [0, 0],
            turn: Player::Light,
            forbidden: 0,
            last_relocated_to: [None, None],
            last_capture: 0,
            last_player: None,
            winner: None,
            ply: 40,
        };
        let result = search_best_action_with_tactical_filter_deadline(
            state,
            SearchConfig {
                depth: 8,
                max_nodes: 1_500_000,
                beam_width: 48,
                ..SearchConfig::default()
            },
            1,
        );
        assert!(result.exhausted);
        assert!(result
            .action
            .is_some_and(|action| state.legal_actions().contains(&action)));
        assert!(result.completed_depth <= 8);
        assert!(result.nodes <= 1_500_000);
    }

    #[test]
    fn ordinary_deadline_returns_a_legal_last_completed_move() {
        let bits = |squares: &[u8]| {
            squares
                .iter()
                .fold(0_u64, |mask, square| mask | (1_u64 << square))
        };
        let state = GameState {
            config: crate::BoardConfig::DEFAULT,
            light: bits(&[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]),
            dark: bits(&[28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 29, 31, 33]),
            reserve: [0, 0],
            turn: Player::Light,
            forbidden: 0,
            last_relocated_to: [None, None],
            last_capture: 0,
            last_player: None,
            winner: None,
            ply: 40,
        };
        let result = search_best_action_with_deadline(
            state,
            SearchConfig {
                depth: 8,
                max_nodes: 1_500_000,
                beam_width: 48,
                ..SearchConfig::default()
            },
            1,
        );
        assert!(result.exhausted);
        assert!(result
            .action
            .is_some_and(|action| state.legal_actions().contains(&action)));
        assert!(result.completed_depth <= 8);
        assert!(result.nodes <= 1_500_000);
    }

    #[test]
    fn tactical_filter_keeps_the_forced_block_fixture() {
        let config = crate::BoardConfig::new(4, 5)
            .expect("valid board config")
            .with_max_plies(64)
            .expect("valid ply limit");
        let state = GameState {
            config,
            light: [5_u8, 7, 9, 11, 15]
                .into_iter()
                .fold(0, |mask, square| mask | (1_u64 << square)),
            dark: [1_u8, 2, 3, 6, 10]
                .into_iter()
                .fold(0, |mask, square| mask | (1_u64 << square)),
            reserve: [0, 0],
            turn: Player::Light,
            forbidden: 0,
            last_relocated_to: [None, None],
            last_capture: 0,
            last_player: None,
            winner: None,
            ply: 20,
        };
        let result = search_best_action_with_tactical_filter(
            state,
            SearchConfig {
                depth: 3,
                max_nodes: 10_000,
                beam_width: 8,
                ..SearchConfig::default()
            },
        );
        assert!([
            Action::Relocate { from: 5, to: 0 },
            Action::Relocate { from: 7, to: 0 },
            Action::Relocate { from: 9, to: 0 },
            Action::Relocate { from: 11, to: 0 },
            Action::Relocate { from: 15, to: 0 },
        ]
        .contains(&result.action.expect("filter returns an action")));
    }

    #[test]
    fn proof_guided_search_selects_a_bounded_seven_by_seven_win() {
        let config = crate::BoardConfig::new(7, 14)
            .expect("valid board config")
            .with_max_plies(180)
            .expect("valid ply limit");
        let bits = |squares: &[u8]| {
            squares
                .iter()
                .fold(0_u64, |mask, square| mask | (1_u64 << square))
        };
        let state = GameState {
            config,
            light: bits(&[7, 14, 21, 28, 35, 42, 48]),
            dark: bits(&[1, 2, 3, 4, 5, 6]),
            reserve: [0, 0],
            turn: Player::Light,
            forbidden: 0,
            last_relocated_to: [None, None],
            last_capture: 0,
            last_player: None,
            winner: None,
            ply: 20,
        };
        let result = search_best_action_with_tactical_proof(
            state,
            SearchConfig {
                depth: 4,
                max_nodes: 2_000,
                beam_width: 8,
                ..SearchConfig::default()
            },
            1,
            1_000,
            &[],
        );
        assert_eq!(result.action, Some(Action::Relocate { from: 48, to: 0 }));
        assert_eq!(result.score, WIN_SCORE - state.ply as i32);
        assert_eq!(result.completed_depth, 1);
        assert!(!result.exhausted);
    }

    #[test]
    fn external_root_order_is_respected_under_a_tiny_budget() {
        let state = GameState::new();
        let preferred = state
            .legal_actions()
            .last()
            .copied()
            .expect("opening position has legal actions");
        let result = search_best_action_with_root_order(
            state,
            SearchConfig {
                depth: 1,
                max_nodes: 1,
                beam_width: 1,
                ..SearchConfig::default()
            },
            &[preferred],
        );
        assert_eq!(result.action, Some(preferred));
        assert_eq!(result.nodes, 1);
        assert!(result.exhausted);
    }

    #[test]
    fn root_candidate_limit_restricts_hybrid_search_to_sorted_pool() {
        let state = GameState::new();
        let preferred = state
            .legal_actions()
            .last()
            .copied()
            .expect("opening position has legal actions");
        let result = search_best_action_with_root_order_and_root_limit(
            state,
            SearchConfig {
                depth: 1,
                max_nodes: 100,
                beam_width: 8,
                ..SearchConfig::default()
            },
            &[preferred],
            false,
            Some(1),
        );
        assert_eq!(result.action, Some(preferred));
        assert!(!result.exhausted);
    }

    #[test]
    fn tactical_root_guard_preserves_legality_and_fallback_order() {
        let state = GameState::new();
        let guarded = ordered_root_actions_with_tactical_guard(
            state,
            state.turn,
            EvaluationWeights::default(),
        );
        assert_eq!(guarded.len(), state.legal_actions().len());
        assert!(guarded
            .iter()
            .all(|action| state.legal_actions().contains(action)));
        assert_eq!(
            guarded
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>(),
            state.legal_actions().into_iter().collect()
        );
    }

    #[test]
    fn lunatic_returns_a_legal_action() {
        let state = GameState::new();
        let result = lunatic_action(state);
        assert!(result.action.is_some());
        assert!(state.legal_actions().contains(&result.action.unwrap()));
        assert_eq!(result.completed_depth, 1);
    }

    #[test]
    fn boundary_search_helpers_cover_empty_terminal_and_probe_fallbacks() {
        let no_moves = GameState {
            config: crate::BoardConfig::new(3, 4).unwrap(),
            light: (1_u64 << 0) | (1_u64 << 1) | (1_u64 << 3) | (1_u64 << 4),
            dark: (1_u64 << 2) | (1_u64 << 5) | (1_u64 << 6) | (1_u64 << 7),
            reserve: [0, 0],
            turn: Player::Light,
            forbidden: 1_u64 << 8,
            last_relocated_to: [None, None],
            last_capture: 0,
            last_player: None,
            winner: None,
            ply: 3,
        };
        let default_config = SearchConfig {
            depth: 2,
            max_nodes: 64,
            beam_width: 4,
            ..SearchConfig::default()
        };
        assert!(lunatic_action(no_moves).action.is_none());
        assert!(analyze_actions(no_moves, default_config, 4).is_empty());
        assert!(
            tactical_root_safe_actions(no_moves, Player::Light, default_config.weights).is_empty()
        );
        assert!(ordered_root_actions_with_tactical_guard(
            no_moves,
            Player::Light,
            default_config.weights
        )
        .is_empty());
        assert!(!has_tactical_signal(no_moves));
        assert!(immediate_winning_actions(no_moves, Player::Dark).is_empty());

        let state = GameState::new();
        let action = state.legal_actions()[0];
        let evaluation = analyze_action(state, action, default_config).unwrap();
        assert_eq!(evaluation.action, action);
        assert_eq!(evaluation.delta, evaluation.score - evaluation.before_score);
        assert!(analyze_action(state, Action::Place { to: 63 }, default_config).is_err());
        assert!(analyze_actions(state, default_config, 0).is_empty());
        assert!(
            search_best_action_with_root_probe(state, default_config, 0, 8, 2,)
                .action
                .is_some()
        );
        let exhausted_probe = search_best_action_with_root_probe(
            state,
            SearchConfig {
                max_nodes: 0,
                ..default_config
            },
            2,
            8,
            2,
        );
        assert!(exhausted_probe.exhausted);
        assert_eq!(exhausted_probe.nodes, 0);
        assert!(
            search_best_action_with_root_order_and_options(state, default_config, &[], true,)
                .action
                .is_some()
        );
        assert!(search_best_action_with_root_order_and_root_limit_deadline(
            state,
            default_config,
            &[],
            false,
            Some(1),
            0,
        )
        .action
        .is_some());

        let mut terminal = state;
        terminal.winner = Some(Player::Light);
        assert!(lunatic_action(terminal).action.is_none());
        assert!(immediate_winning_actions(terminal, Player::Light).is_empty());
    }

    #[test]
    fn search_supports_variable_board_sizes() {
        for size in 4..=7 {
            let state = GameState::with_board_size(size);
            let result = search_best_action(
                state,
                SearchConfig {
                    depth: 1,
                    max_nodes: 256,
                    beam_width: 64,
                    ..SearchConfig::default()
                },
            );
            assert!(
                result.action.is_some(),
                "{size}x{size} search returned no action"
            );
            assert!(
                state.legal_actions().contains(&result.action.unwrap()),
                "{size}x{size} search returned an illegal action"
            );
        }
    }

    #[test]
    fn optional_tactical_proof_mode_selects_a_forced_block() {
        let config = crate::BoardConfig::new(4, 5)
            .expect("valid board config")
            .with_max_plies(64)
            .expect("valid ply limit");
        let state = GameState {
            config,
            light: [5_u8, 7, 9, 11, 15]
                .into_iter()
                .fold(0, |mask, square| mask | (1_u64 << square)),
            dark: [1_u8, 2, 3, 6, 10]
                .into_iter()
                .fold(0, |mask, square| mask | (1_u64 << square)),
            reserve: [0, 0],
            turn: Player::Light,
            forbidden: 0,
            last_relocated_to: [None, None],
            last_capture: 0,
            last_player: None,
            winner: None,
            ply: 20,
        };
        let result = search_best_action(
            state,
            SearchConfig {
                tactical_proof_horizon: Some(3),
                max_nodes: 100_000,
                ..SearchConfig::default()
            },
        );
        assert!([5_u8, 7, 9, 11, 15]
            .into_iter()
            .map(|from| Action::Relocate { from, to: 0 })
            .any(|action| result.action == Some(action)));
        assert_eq!(result.score, 0);
        assert_eq!(result.completed_depth, 3);
        assert!(!result.exhausted);
    }
}
