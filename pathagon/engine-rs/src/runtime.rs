//! Stable state/action boundary for the browser engine adapter.
//!
//! The rules engine itself uses compact bitboards. The browser and the Python
//! learner use portable square arrays instead. This module is the translation
//! seam between those representations. It deliberately contains no UI or
//! search-policy assumptions, so the same functions can later be exposed with
//! wasm-bindgen and used by native self-play.

use serde::{Deserialize, Serialize};

use crate::contract::{
    ContractAction, ContractPlayer, GameConfig, PlayerNumbers, PlayerSquares, Position,
    CONTRACT_VERSION,
};
use crate::search::{
    analyze_action, analyze_actions, lunatic_action, search_best_action,
    search_best_action_with_golden_bytes, search_best_action_with_tactical_filter,
    search_best_action_with_tactical_filter_deadline,
    search_best_action_with_tactical_filter_deadline_progress,
    search_best_action_with_tactical_filter_deadline_trace, MoveEvaluation, SearchConfig,
    SearchProgressCallback, SearchResult, SearchTraceCallback,
};
use crate::transition_policy::{RankedTransitionAction, TransitionPolicyModel};
use crate::{bit_squares, Action, BoardConfig, GameState, Player};

/// Hard ceiling for browser-supplied search budgets. Native research jobs use
/// their own explicit budgets; this guard protects the synchronous WASM
/// boundary from accidentally requesting an unbounded table.
pub const MAX_RUNTIME_SEARCH_NODES: u64 = 10_000_000;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RuntimePosition {
    #[serde(rename = "contractVersion")]
    pub contract_version: u8,
    pub config: GameConfig,
    pub board: Vec<Option<ContractPlayer>>,
    pub reserve: PlayerNumbers,
    pub turn: ContractPlayer,
    pub forbidden: Vec<u8>,
    #[serde(rename = "lastRelocatedTo")]
    pub last_relocated_to: PlayerSquares,
    #[serde(rename = "lastCapture", default)]
    pub last_capture: u8,
    #[serde(rename = "lastPlayer", default)]
    pub last_player: Option<ContractPlayer>,
    pub winner: Option<ContractPlayer>,
    #[serde(rename = "winningPath", default)]
    pub winning_path: Vec<u8>,
    pub ply: u16,
}

/// The complete result of applying one action at the runtime boundary.
/// `position` is the post-move state; the other fields make the transition
/// auditable without replaying the action or inferring captures from boards.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RuntimeActionResult {
    pub player: ContractPlayer,
    pub action: ContractAction,
    pub captured: Vec<u8>,
    pub position: RuntimePosition,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RuntimeSearchConfig {
    pub depth: u8,
    #[serde(rename = "maxNodes")]
    pub max_nodes: u64,
    #[serde(rename = "beamWidth")]
    pub beam_width: usize,
    pub weights: crate::search::EvaluationWeights,
    #[serde(rename = "tacticalProofHorizon", default)]
    pub tactical_proof_horizon: Option<u8>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RuntimeSearchResult {
    pub action: Option<ContractAction>,
    pub score: i32,
    pub nodes: u64,
    pub exhausted: bool,
    #[serde(rename = "completedDepth")]
    pub completed_depth: u8,
    #[serde(rename = "tableHits")]
    pub table_hits: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RuntimeMoveEvaluation {
    pub action: ContractAction,
    #[serde(rename = "beforeScore")]
    pub before_score: i32,
    pub score: i32,
    pub delta: i32,
    pub nodes: u64,
    pub exhausted: bool,
    #[serde(rename = "completedDepth")]
    pub completed_depth: u8,
    #[serde(rename = "tableHits")]
    pub table_hits: u64,
}

/// Browser-facing policy ranking. Actions are already legal and tactical-safe
/// when this response is produced; the recursive search still decides the
/// final move when called through `search_transition_policy_json`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct RuntimeTransitionPolicyAction {
    pub action: ContractAction,
    pub safe: bool,
    #[serde(rename = "immediateWin")]
    pub immediate_win: bool,
    pub score: f32,
}

impl From<RankedTransitionAction> for RuntimeTransitionPolicyAction {
    fn from(item: RankedTransitionAction) -> Self {
        Self {
            action: item.action.into(),
            safe: item.safe,
            immediate_win: item.immediate_win,
            score: item.score,
        }
    }
}

impl From<MoveEvaluation> for RuntimeMoveEvaluation {
    fn from(result: MoveEvaluation) -> Self {
        Self {
            action: result.action.into(),
            before_score: result.before_score,
            score: result.score,
            delta: result.delta,
            nodes: result.nodes,
            exhausted: result.exhausted,
            completed_depth: result.completed_depth,
            table_hits: result.table_hits,
        }
    }
}

impl From<SearchResult> for RuntimeSearchResult {
    fn from(result: SearchResult) -> Self {
        Self {
            action: result.action.map(Into::into),
            score: result.score,
            nodes: result.nodes,
            exhausted: result.exhausted,
            completed_depth: result.completed_depth,
            table_hits: result.table_hits,
        }
    }
}

impl From<GameState> for RuntimePosition {
    fn from(state: GameState) -> Self {
        let cells = state.config.cells();
        let board = (0..cells)
            .map(|square| state.board_at(square).map(ContractPlayer::from))
            .collect();
        Self {
            contract_version: CONTRACT_VERSION,
            config: GameConfig {
                rules_version: crate::contract::RULES_VERSION.to_owned(),
                board_size: state.config.board_size,
                reserve_per_player: state.config.reserve_per_player,
                max_plies: state.config.max_plies,
                repetition_limit: 3,
            },
            board,
            reserve: PlayerNumbers {
                light: u16::from(state.reserve[Player::Light.index()]),
                dark: u16::from(state.reserve[Player::Dark.index()]),
            },
            turn: state.turn.into(),
            forbidden: crate::bit_squares(state.forbidden),
            last_relocated_to: PlayerSquares {
                light: state.last_relocated_to[Player::Light.index()],
                dark: state.last_relocated_to[Player::Dark.index()],
            },
            last_capture: state.last_capture,
            last_player: state.last_player.map(ContractPlayer::from),
            winner: state.winner.map(ContractPlayer::from),
            winning_path: state
                .winner
                .map_or_else(Vec::new, |player| crate::winning_path(state, player)),
            ply: state.ply,
        }
    }
}

impl TryFrom<RuntimePosition> for GameState {
    type Error = String;

    fn try_from(position: RuntimePosition) -> Result<Self, Self::Error> {
        let contract_position = Position {
            contract_version: position.contract_version,
            config: position.config.clone(),
            board: position.board.clone(),
            reserve: position.reserve.clone(),
            turn: position.turn,
            forbidden: position.forbidden.clone(),
            last_relocated_to: position.last_relocated_to.clone(),
            winner: position.winner,
            ply: position.ply,
        };
        contract_position.validate()?;
        if position.last_capture > 4 {
            return Err("last capture exceeds the four orthogonal directions".to_owned());
        }

        let config = BoardConfig::from_contract(&position.config)?;
        let mut light = 0_u64;
        let mut dark = 0_u64;
        for (square, piece) in position.board.iter().enumerate() {
            let bit = 1_u64 << square;
            match piece {
                Some(ContractPlayer::Light) => light |= bit,
                Some(ContractPlayer::Dark) => dark |= bit,
                None => {}
            }
        }
        let forbidden = position
            .forbidden
            .iter()
            .fold(0_u64, |mask, square| mask | (1_u64 << square));
        Ok(GameState {
            config,
            light,
            dark,
            reserve: [
                u8::try_from(position.reserve.light)
                    .map_err(|_| "light reserve exceeds u8".to_owned())?,
                u8::try_from(position.reserve.dark)
                    .map_err(|_| "dark reserve exceeds u8".to_owned())?,
            ],
            turn: position.turn.into(),
            forbidden,
            last_relocated_to: [
                position.last_relocated_to.light,
                position.last_relocated_to.dark,
            ],
            last_capture: position.last_capture,
            last_player: position.last_player.map(Player::from),
            winner: position.winner.map(Player::from),
            ply: position.ply,
        })
    }
}

impl From<Player> for ContractPlayer {
    fn from(player: Player) -> Self {
        match player {
            Player::Light => Self::Light,
            Player::Dark => Self::Dark,
        }
    }
}

impl From<ContractPlayer> for Player {
    fn from(player: ContractPlayer) -> Self {
        match player {
            ContractPlayer::Light => Self::Light,
            ContractPlayer::Dark => Self::Dark,
        }
    }
}

impl From<Action> for ContractAction {
    fn from(action: Action) -> Self {
        match action {
            Action::Place { to } => Self::Place { to },
            Action::Relocate { from, to } => Self::Relocate { from, to },
        }
    }
}

impl From<ContractAction> for Action {
    fn from(action: ContractAction) -> Self {
        match action {
            ContractAction::Place { to } => Self::Place { to },
            ContractAction::Relocate { from, to } => Self::Relocate { from, to },
        }
    }
}

impl From<RuntimeSearchConfig> for SearchConfig {
    fn from(config: RuntimeSearchConfig) -> Self {
        Self {
            depth: config.depth,
            max_nodes: config.max_nodes.min(MAX_RUNTIME_SEARCH_NODES),
            beam_width: config.beam_width,
            weights: config.weights,
            tactical_proof_horizon: config.tactical_proof_horizon,
        }
    }
}

pub fn parse_position(json: &str) -> Result<GameState, String> {
    let position: RuntimePosition =
        serde_json::from_str(json).map_err(|error| error.to_string())?;
    position.try_into()
}

pub fn position_json(state: GameState) -> Result<String, String> {
    serde_json::to_string(&RuntimePosition::from(state)).map_err(|error| error.to_string())
}

pub fn legal_actions_json(state_json: &str) -> Result<String, String> {
    let state = parse_position(state_json)?;
    let actions: Vec<ContractAction> = state.legal_actions().into_iter().map(Into::into).collect();
    serde_json::to_string(&actions).map_err(|error| error.to_string())
}

pub fn apply_action_json(state_json: &str, action_json: &str) -> Result<String, String> {
    let state = parse_position(state_json)?;
    let action: ContractAction =
        serde_json::from_str(action_json).map_err(|error| error.to_string())?;
    let transition = state.apply(action.into()).map_err(str::to_owned)?;
    position_json(transition.state)
}

pub fn apply_action_transition_json(state_json: &str, action_json: &str) -> Result<String, String> {
    let state = parse_position(state_json)?;
    let action: ContractAction =
        serde_json::from_str(action_json).map_err(|error| error.to_string())?;
    let player = state.turn;
    let transition = state.apply(action.clone().into()).map_err(str::to_owned)?;
    let response = RuntimeActionResult {
        player: player.into(),
        action,
        captured: bit_squares(transition.captured),
        position: RuntimePosition::from(transition.state),
    };
    serde_json::to_string(&response).map_err(|error| error.to_string())
}

pub fn search_best_action_json(state_json: &str, config_json: &str) -> Result<String, String> {
    let state = parse_position(state_json)?;
    let config: RuntimeSearchConfig =
        serde_json::from_str(config_json).map_err(|error| error.to_string())?;
    let result = search_best_action(state, config.into());
    let response = RuntimeSearchResult::from(result);
    serde_json::to_string(&response).map_err(|error| error.to_string())
}

/// Browser-facing gold-aware search. Immutable table and action-book bytes
/// are supplied by the caller; an exact action short-circuits search, while
/// an absent or value-only row remains available as explicit metadata and the
/// unknown branches use ordinary search.
pub fn search_best_action_with_golden_bytes_json(
    state_json: &str,
    config_json: &str,
    table_bytes: &[u8],
    sidecar_bytes: Option<&[u8]>,
) -> Result<String, String> {
    let state = parse_position(state_json)?;
    let config: RuntimeSearchConfig =
        serde_json::from_str(config_json).map_err(|error| error.to_string())?;
    let (result, outcome, exact_action) =
        search_best_action_with_golden_bytes(state, config.into(), table_bytes, sidecar_bytes)?;
    let mut response = serde_json::to_value(RuntimeSearchResult::from(result))
        .map_err(|error| error.to_string())?;
    if let serde_json::Value::Object(object) = &mut response {
        object.insert(
            "goldenOutcome".to_owned(),
            outcome
                .map(|value| serde_json::json!(value.as_str()))
                .unwrap_or(serde_json::Value::Null),
        );
        object.insert("goldenAction".to_owned(), serde_json::json!(exact_action));
    }
    serde_json::to_string(&response).map_err(|error| error.to_string())
}

/// Browser-facing promoted Pathfinder entry point. The underlying evaluator
/// and alpha-beta search are unchanged; the tactical-safe root filter only
/// removes moves that hand the opponent an immediate win when a safe
/// alternative exists.
pub fn search_best_action_with_tactical_filter_json(
    state_json: &str,
    config_json: &str,
) -> Result<String, String> {
    let state = parse_position(state_json)?;
    let config: RuntimeSearchConfig =
        serde_json::from_str(config_json).map_err(|error| error.to_string())?;
    let result = search_best_action_with_tactical_filter(state, config.into());
    let response = RuntimeSearchResult::from(result);
    serde_json::to_string(&response).map_err(|error| error.to_string())
}

pub fn search_best_action_with_tactical_filter_deadline_json(
    state_json: &str,
    config_json: &str,
    deadline_ms: u32,
) -> Result<String, String> {
    let state = parse_position(state_json)?;
    let config: RuntimeSearchConfig =
        serde_json::from_str(config_json).map_err(|error| error.to_string())?;
    let result =
        search_best_action_with_tactical_filter_deadline(state, config.into(), deadline_ms);
    let response = RuntimeSearchResult::from(result);
    serde_json::to_string(&response).map_err(|error| error.to_string())
}

pub fn search_best_action_with_tactical_filter_deadline_progress_json(
    state_json: &str,
    config_json: &str,
    deadline_ms: u32,
    progress: SearchProgressCallback,
) -> Result<String, String> {
    let state = parse_position(state_json)?;
    let config: RuntimeSearchConfig =
        serde_json::from_str(config_json).map_err(|error| error.to_string())?;
    let result = search_best_action_with_tactical_filter_deadline_progress(
        state,
        config.into(),
        deadline_ms,
        progress,
    );
    let response = RuntimeSearchResult::from(result);
    serde_json::to_string(&response).map_err(|error| error.to_string())
}

pub fn search_best_action_with_tactical_filter_deadline_trace_json(
    state_json: &str,
    config_json: &str,
    deadline_ms: u32,
    progress: SearchProgressCallback,
    trace: SearchTraceCallback,
) -> Result<String, String> {
    let state = parse_position(state_json)?;
    let config: RuntimeSearchConfig =
        serde_json::from_str(config_json).map_err(|error| error.to_string())?;
    let result = search_best_action_with_tactical_filter_deadline_trace(
        state,
        config.into(),
        deadline_ms,
        progress,
        trace,
    );
    serde_json::to_string(&RuntimeSearchResult::from(result)).map_err(|error| error.to_string())
}

pub fn lunatic_action_json(state_json: &str) -> Result<String, String> {
    let state = parse_position(state_json)?;
    serde_json::to_string(&RuntimeSearchResult::from(lunatic_action(state)))
        .map_err(|error| error.to_string())
}

pub fn analyze_action_json(
    state_json: &str,
    action_json: &str,
    config_json: &str,
) -> Result<String, String> {
    let state = parse_position(state_json)?;
    let action: ContractAction =
        serde_json::from_str(action_json).map_err(|error| error.to_string())?;
    let config: RuntimeSearchConfig =
        serde_json::from_str(config_json).map_err(|error| error.to_string())?;
    let result: RuntimeMoveEvaluation = analyze_action(state, action.into(), config.into())?.into();
    serde_json::to_string(&result).map_err(|error| error.to_string())
}

pub fn analyze_actions_json(
    state_json: &str,
    config_json: &str,
    max_actions: usize,
) -> Result<String, String> {
    let state = parse_position(state_json)?;
    let config: RuntimeSearchConfig =
        serde_json::from_str(config_json).map_err(|error| error.to_string())?;
    let results: Vec<RuntimeMoveEvaluation> = analyze_actions(state, config.into(), max_actions)
        .into_iter()
        .map(Into::into)
        .collect();
    serde_json::to_string(&results).map_err(|error| error.to_string())
}

/// Rank legal roots through the explicit action-transition model. This is a
/// model-inspection endpoint; use `search_transition_policy_json` to combine
/// the ranking with the rules-authoritative alpha-beta search.
pub fn rank_transition_policy_json(
    state_json: &str,
    model: &TransitionPolicyModel,
    max_actions: usize,
) -> Result<String, String> {
    let state = parse_position(state_json)?;
    let ranked = model
        .ranked_actions(state, crate::search::EvaluationWeights::default())
        .into_iter()
        .take(if max_actions == 0 {
            usize::MAX
        } else {
            max_actions
        })
        .map(Into::into)
        .collect::<Vec<RuntimeTransitionPolicyAction>>();
    serde_json::to_string(&ranked).map_err(|error| error.to_string())
}

/// Search with the packaged explicit transition scorer. The model only
/// orders the tactical-safe root; legal-action generation and recursive
/// evaluation remain in the Rust rules/search engine.
pub fn search_transition_policy_json(
    state_json: &str,
    config_json: &str,
    model: &TransitionPolicyModel,
    deadline_ms: u32,
) -> Result<String, String> {
    let state = parse_position(state_json)?;
    let config: RuntimeSearchConfig =
        serde_json::from_str(config_json).map_err(|error| error.to_string())?;
    let result = model.search(state, config.into(), Some(deadline_ms.max(1)));
    serde_json::to_string(&RuntimeSearchResult::from(result)).map_err(|error| error.to_string())
}

pub fn search_transition_policy_with_progress_json(
    state_json: &str,
    config_json: &str,
    model: &TransitionPolicyModel,
    deadline_ms: u32,
    progress: SearchProgressCallback,
) -> Result<String, String> {
    let state = parse_position(state_json)?;
    let config: RuntimeSearchConfig =
        serde_json::from_str(config_json).map_err(|error| error.to_string())?;
    let result = model.search_with_progress(state, config.into(), deadline_ms, progress);
    serde_json::to_string(&RuntimeSearchResult::from(result)).map_err(|error| error.to_string())
}

pub fn search_transition_policy_with_trace_json(
    state_json: &str,
    config_json: &str,
    model: &TransitionPolicyModel,
    deadline_ms: u32,
    progress: SearchProgressCallback,
    trace: SearchTraceCallback,
) -> Result<String, String> {
    let state = parse_position(state_json)?;
    let config: RuntimeSearchConfig =
        serde_json::from_str(config_json).map_err(|error| error.to_string())?;
    let result = model.search_with_trace(state, config.into(), deadline_ms, progress, trace);
    serde_json::to_string(&RuntimeSearchResult::from(result)).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply_transition(state: GameState, action: ContractAction) -> RuntimeActionResult {
        let state_json = position_json(state).expect("serialize transition position");
        let action_json = serde_json::to_string(&action).expect("encode transition action");
        let transition = apply_action_transition_json(&state_json, &action_json)
            .expect("apply transition at runtime boundary");
        serde_json::from_str(&transition).expect("decode transition result")
    }

    #[test]
    fn runtime_position_round_trips_rule_state() {
        let mut state = GameState::new();
        state.reserve = [13, 13];
        state = state.apply_legal(Action::Place { to: 23 }).state;
        let json = position_json(state).expect("serialize runtime position");
        let restored = parse_position(&json).expect("parse runtime position");
        assert_eq!(restored, state);
    }

    #[test]
    fn runtime_action_boundary_matches_native_rules() {
        let state_json = position_json(GameState::new()).expect("serialize initial state");
        let actions: Vec<ContractAction> =
            serde_json::from_str(&legal_actions_json(&state_json).expect("legal actions"))
                .expect("decode legal actions");
        assert_eq!(actions.len(), 49);
        let next = apply_action_json(
            &state_json,
            &serde_json::to_string(&actions[24]).expect("encode action"),
        )
        .expect("apply action");
        assert_eq!(parse_position(&next).expect("parse next state").ply, 1);
    }

    #[test]
    fn runtime_action_transition_reports_ordinary_placement() {
        let transition = apply_transition(GameState::new(), ContractAction::Place { to: 24 });

        assert_eq!(transition.player, ContractPlayer::Light);
        assert_eq!(transition.action, ContractAction::Place { to: 24 });
        assert!(transition.captured.is_empty());
        assert_eq!(transition.position.board[24], Some(ContractPlayer::Light));
        assert_eq!(transition.position.reserve.light, 13);
        assert_eq!(transition.position.reserve.dark, 14);
        assert!(transition.position.forbidden.is_empty());
        assert_eq!(transition.position.last_capture, 0);
        assert_eq!(transition.position.last_player, Some(ContractPlayer::Light));
        assert_eq!(transition.position.turn, ContractPlayer::Dark);
        assert_eq!(transition.position.winner, None);
        assert_eq!(transition.position.ply, 1);
    }

    #[test]
    fn runtime_action_transition_reports_capture_and_post_state() {
        let mut state = GameState::new();
        state.light = 1_u64 << 21;
        state.dark = 1_u64 << 22;
        state.turn = Player::Light;
        let state_json = position_json(state).expect("serialize capture position");
        let action_json = serde_json::to_string(&ContractAction::Place { to: 23 })
            .expect("encode capture action");
        let transition =
            apply_action_transition_json(&state_json, &action_json).expect("apply transition");
        let transition: RuntimeActionResult =
            serde_json::from_str(&transition).expect("decode transition");
        assert_eq!(transition.player, ContractPlayer::Light);
        assert_eq!(transition.action, ContractAction::Place { to: 23 });
        assert_eq!(transition.captured, vec![22]);
        assert_eq!(transition.position.forbidden, vec![22]);
        assert_eq!(transition.position.reserve.dark, 15);
        assert_eq!(transition.position.board[21], Some(ContractPlayer::Light));
        assert_eq!(transition.position.board[22], None);
        assert_eq!(transition.position.board[23], Some(ContractPlayer::Light));
        assert_eq!(transition.position.last_capture, 1);
        assert_eq!(transition.position.last_player, Some(ContractPlayer::Light));
        assert_eq!(transition.position.turn, ContractPlayer::Dark);
        assert_eq!(transition.position.ply, 1);
    }

    #[test]
    fn runtime_action_transition_reports_relocation_capture() {
        let mut state = GameState::new();
        state.light = (1_u64 << 21) | (1_u64 << 30);
        state.dark = 1_u64 << 22;
        state.reserve = [0, 0];
        state.turn = Player::Light;

        let transition = apply_transition(state, ContractAction::Relocate { from: 30, to: 23 });

        assert_eq!(transition.player, ContractPlayer::Light);
        assert_eq!(
            transition.action,
            ContractAction::Relocate { from: 30, to: 23 }
        );
        assert_eq!(transition.captured, vec![22]);
        assert_eq!(transition.position.board[21], Some(ContractPlayer::Light));
        assert_eq!(transition.position.board[22], None);
        assert_eq!(transition.position.board[23], Some(ContractPlayer::Light));
        assert_eq!(transition.position.board[30], None);
        assert_eq!(transition.position.forbidden, vec![22]);
        assert_eq!(transition.position.last_relocated_to.light, Some(23));
        assert_eq!(transition.position.last_capture, 1);
        assert_eq!(transition.position.reserve.light, 0);
        assert_eq!(transition.position.reserve.dark, 1);
        assert_eq!(transition.position.turn, ContractPlayer::Dark);
        assert_eq!(transition.position.ply, 1);
    }

    #[test]
    fn runtime_action_transition_reports_all_four_direction_captures() {
        let mut state = GameState::new();
        // Target 24 has an opposing stone on each orthogonal near square and
        // a Light stone two squares away to support each capture ray.
        state.light = (1_u64 << 10) | (1_u64 << 22) | (1_u64 << 26) | (1_u64 << 38) | (1_u64 << 40);
        state.dark = (1_u64 << 17) | (1_u64 << 23) | (1_u64 << 25) | (1_u64 << 31);
        state.reserve = [0, 0];
        state.turn = Player::Light;

        let transition = apply_transition(state, ContractAction::Relocate { from: 40, to: 24 });

        assert_eq!(transition.captured, vec![17, 23, 25, 31]);
        assert_eq!(transition.position.forbidden, vec![17, 23, 25, 31]);
        assert_eq!(transition.position.last_capture, 4);
        assert_eq!(transition.position.reserve.dark, 4);
        assert_eq!(transition.position.board[24], Some(ContractPlayer::Light));
        assert_eq!(transition.position.board[40], None);
        for square in [17, 23, 25, 31] {
            assert_eq!(transition.position.board[square], None);
        }
        assert_eq!(transition.position.winner, None);
        assert_eq!(transition.position.turn, ContractPlayer::Dark);
    }

    #[test]
    fn runtime_action_transition_reports_terminal_winner_and_path() {
        let mut state = GameState::new();
        // Light already spans rows 1 through 6 in column 0; P0 completes the
        // top-to-bottom path and must be reported as a terminal transition.
        state.light = (1_u64 << 7)
            | (1_u64 << 14)
            | (1_u64 << 21)
            | (1_u64 << 28)
            | (1_u64 << 35)
            | (1_u64 << 42);
        state.dark = 1_u64 << 1;
        state.reserve = [8, 13];
        state.turn = Player::Light;

        let transition = apply_transition(state, ContractAction::Place { to: 0 });

        assert_eq!(transition.captured, Vec::<u8>::new());
        assert_eq!(transition.position.board[0], Some(ContractPlayer::Light));
        assert_eq!(transition.position.winner, Some(ContractPlayer::Light));
        assert_eq!(transition.position.winning_path.len(), 7);
        assert!(transition.position.winning_path.contains(&0));
        assert!(transition.position.winning_path.contains(&42));
        assert_eq!(transition.position.turn, ContractPlayer::Dark);
        assert_eq!(transition.position.ply, 1);
    }

    #[test]
    fn runtime_analysis_boundary_returns_canonical_scores() {
        let state_json = position_json(GameState::new()).expect("serialize initial state");
        let config = serde_json::to_string(&RuntimeSearchConfig {
            depth: 1,
            max_nodes: 100,
            beam_width: 16,
            weights: crate::search::EvaluationWeights::default(),
            tactical_proof_horizon: None,
        })
        .expect("encode search config");
        let actions: Vec<ContractAction> =
            serde_json::from_str(&legal_actions_json(&state_json).expect("legal actions"))
                .expect("decode legal actions");
        let one = analyze_action_json(
            &state_json,
            &serde_json::to_string(&actions[0]).expect("encode action"),
            &config,
        )
        .expect("analyze action");
        let one: RuntimeMoveEvaluation = serde_json::from_str(&one).expect("decode evaluation");
        assert_eq!(one.action, actions[0]);

        let many = analyze_actions_json(&state_json, &config, 7).expect("analyze actions");
        let many: Vec<RuntimeMoveEvaluation> =
            serde_json::from_str(&many).expect("decode evaluations");
        assert_eq!(many.len(), 7);
        assert!(many.windows(2).all(|pair| pair[0].score >= pair[1].score));
    }

    #[test]
    fn runtime_search_config_exposes_optional_tactical_proof_horizon() {
        let config: RuntimeSearchConfig = serde_json::from_str(
            r#"{"depth":4,"maxNodes":32000,"beamWidth":256,"weights":{"path":240,"material":110,"capture":700,"structure":55,"threat":130,"edge":80},"tacticalProofHorizon":3}"#,
        )
        .expect("decode tactical proof search config");
        assert_eq!(SearchConfig::from(config).tactical_proof_horizon, Some(3));
    }

    #[test]
    fn runtime_search_config_clamps_browser_node_budget() {
        let config: RuntimeSearchConfig = serde_json::from_str(
            r#"{"depth":100,"maxNodes":999999999,"beamWidth":2,"weights":{"path":240,"material":110,"capture":700,"structure":55,"threat":130,"edge":80}}"#,
        )
        .expect("decode long-horizon search config");
        assert_eq!(
            SearchConfig::from(config).max_nodes,
            MAX_RUNTIME_SEARCH_NODES
        );
    }

    #[test]
    fn runtime_gold_bytes_expose_exact_value_without_fabricating_an_action() {
        let mut state = GameState::new();
        state.light = (1_u64 << 7)
            | (1_u64 << 14)
            | (1_u64 << 21)
            | (1_u64 << 28)
            | (1_u64 << 35)
            | (1_u64 << 42);
        state.dark = 1_u64 << 1;
        state.reserve = [8, 13];
        state.turn = Player::Light;
        let mut table = crate::golden::canonical_position_key(state);
        table.push(crate::golden::WIN);
        let position = position_json(state).expect("serialize gold lookup position");
        let config = serde_json::to_string(&RuntimeSearchConfig {
            depth: 1,
            max_nodes: 100,
            beam_width: 16,
            weights: crate::search::EvaluationWeights::default(),
            tactical_proof_horizon: None,
        })
        .expect("encode gold lookup config");
        let response: serde_json::Value = serde_json::from_str(
            &search_best_action_with_golden_bytes_json(&position, &config, &table, None)
                .expect("run gold-aware runtime search"),
        )
        .expect("decode gold-aware runtime response");
        assert_eq!(response["goldenOutcome"], "win");
        assert_eq!(response["goldenAction"], false);
        assert!(response["action"].is_object());
    }

    #[test]
    fn transition_policy_runtime_endpoints_keep_actions_legal() {
        let model = TransitionPolicyModel::from_json(
            &serde_json::json!({
                "schemaVersion": 1,
                "model": "tanh-action-state-transition-policy-v2",
                "encoding": "explicit-source-kind",
                "featureOrder": (0..crate::transition_policy::FEATURE_COUNT).map(|i| format!("f{i}")).collect::<Vec<_>>(),
                "mean": vec![0.0; crate::transition_policy::FEATURE_COUNT],
                "scale": vec![1.0; crate::transition_policy::FEATURE_COUNT],
                "layers": [
                    {"weights": vec![vec![0.0; crate::transition_policy::FEATURE_COUNT]], "bias": [0.0]},
                    {"weights": vec![vec![0.0]], "bias": [0.0]},
                    {"weights": vec![vec![0.0]], "bias": [0.0]}
                ]
            })
            .to_string(),
        )
        .expect("valid transition model");
        let position = position_json(GameState::new()).expect("serialize position");
        let ranked: Vec<RuntimeTransitionPolicyAction> = serde_json::from_str(
            &rank_transition_policy_json(&position, &model, 7).expect("rank roots"),
        )
        .expect("decode ranked roots");
        assert_eq!(ranked.len(), 7);
        assert!(ranked.iter().all(|item| {
            let action: Action = item.action.clone().into();
            GameState::new().legal_actions().contains(&action)
        }));
        let config = serde_json::json!({
            "depth": 3,
            "maxNodes": 32_000,
            "beamWidth": 256,
            "weights": {
                "path": 241,
                "material": 112,
                "capture": 887,
                "structure": 40,
                "threat": 154,
                "edge": 74
            }
        })
        .to_string();
        let result: RuntimeSearchResult = serde_json::from_str(
            &search_transition_policy_json(&position, &config, &model, 25)
                .expect("search transition policy"),
        )
        .expect("decode transition-policy search result");
        let action = result.action.expect("initial position has an action");
        assert!(GameState::new().legal_actions().contains(&action.into()));
    }

    #[test]
    fn runtime_json_boundaries_cover_errors_and_all_search_endpoints() {
        assert!(parse_position("not json").is_err());
        let position = position_json(GameState::new()).expect("serialize position");
        let mut malformed: serde_json::Value = serde_json::from_str(&position).unwrap();
        malformed["lastCapture"] = serde_json::json!(5);
        assert!(parse_position(&malformed.to_string()).is_err());
        assert!(legal_actions_json("not json").is_err());
        assert!(apply_action_json(&position, "not json").is_err());
        assert!(apply_action_json(
            &position,
            &serde_json::to_string(&ContractAction::Place { to: 49 }).unwrap()
        )
        .is_err());
        assert!(apply_action_transition_json(&position, "not json").is_err());

        let config = serde_json::json!({
            "depth": 1,
            "maxNodes": 32,
            "beamWidth": 8,
            "weights": {"path": 240, "material": 110, "capture": 700, "structure": 55, "threat": 130, "edge": 80}
        })
        .to_string();
        assert!(search_best_action_json(&position, &config).is_ok());
        assert!(search_best_action_json(&position, "not json").is_err());
        assert!(search_best_action_with_tactical_filter_json(&position, &config).is_ok());
        assert!(
            search_best_action_with_tactical_filter_deadline_json(&position, &config, 1).is_ok()
        );
        assert!(
            search_best_action_with_tactical_filter_deadline_progress_json(
                &position,
                &config,
                1,
                Box::new(|_, _| {}),
            )
            .is_ok()
        );
        let trace_depths = std::rc::Rc::new(std::cell::RefCell::new(Vec::<u8>::new()));
        let trace_depths_for_callback = std::rc::Rc::clone(&trace_depths);
        let trace_config = serde_json::json!({
            "depth": 1,
            "maxNodes": 5_000,
            "beamWidth": 16,
            "weights": {"path": 240, "material": 110, "capture": 700, "structure": 55, "threat": 130, "edge": 80}
        })
        .to_string();
        assert!(
            search_best_action_with_tactical_filter_deadline_trace_json(
                &position,
                &trace_config,
                500,
                Box::new(|_, _| {}),
                Box::new(move |depth, _, _, candidates| {
                    assert!(!candidates.is_empty());
                    trace_depths_for_callback.borrow_mut().push(depth);
                }),
            )
            .is_ok()
        );
        assert_eq!(&*trace_depths.borrow(), &[1]);
        assert!(lunatic_action_json(&position).is_ok());
        assert!(analyze_action_json(&position, "not json", &config).is_err());
        assert!(analyze_actions_json(&position, &config, 0).is_ok());
        assert!(
            search_best_action_with_golden_bytes_json(&position, &config, &[1, 2, 3], None)
                .is_err()
        );

        let model = TransitionPolicyModel::from_json(
            &serde_json::json!({
                "schemaVersion": 1,
                "model": "tanh-action-state-transition-policy-v2",
                "encoding": "explicit-source-kind",
                "featureOrder": (0..crate::transition_policy::FEATURE_COUNT).map(|i| format!("f{i}")).collect::<Vec<_>>(),
                "mean": vec![0.0; crate::transition_policy::FEATURE_COUNT],
                "scale": vec![1.0; crate::transition_policy::FEATURE_COUNT],
                "layers": [
                    {"weights": vec![vec![0.0; crate::transition_policy::FEATURE_COUNT]], "bias": [0.0]},
                    {"weights": vec![vec![0.0]], "bias": [0.0]},
                    {"weights": vec![vec![0.0]], "bias": [0.0]}
                ]
            })
            .to_string(),
        )
        .unwrap();
        let ranked: Vec<RuntimeTransitionPolicyAction> =
            serde_json::from_str(&rank_transition_policy_json(&position, &model, 0).unwrap())
                .unwrap();
        assert_eq!(ranked.len(), 49);
        assert!(search_transition_policy_with_progress_json(
            &position,
            &config,
            &model,
            1,
            Box::new(|_, _| {}),
        )
        .is_ok());
        assert!(rank_transition_policy_json("not json", &model, 1).is_err());
    }
}
