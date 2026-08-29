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
    search_best_action_with_tactical_filter, MoveEvaluation, SearchConfig, SearchResult,
};
use crate::{Action, BoardConfig, GameState, Player};

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
            max_nodes: config.max_nodes,
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

pub fn search_best_action_json(state_json: &str, config_json: &str) -> Result<String, String> {
    let state = parse_position(state_json)?;
    let config: RuntimeSearchConfig =
        serde_json::from_str(config_json).map_err(|error| error.to_string())?;
    let result = search_best_action(state, config.into());
    let response = RuntimeSearchResult::from(result);
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

#[cfg(test)]
mod tests {
    use super::*;

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
            r#"{"depth":4,"maxNodes":90000,"beamWidth":40,"weights":{"path":240,"material":110,"capture":700,"structure":55,"threat":130,"edge":80},"tacticalProofHorizon":3}"#,
        )
        .expect("decode tactical proof search config");
        assert_eq!(SearchConfig::from(config).tactical_proof_horizon, Some(3));
    }
}
