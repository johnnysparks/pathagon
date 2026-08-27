//! Tensor ABI shared by the Python learner and Rust inference backends.
//!
//! This is intentionally a data-only boundary. Rust remains the authority for
//! legal actions and state features; a policy/value model only scores the
//! already-generated action list.

use crate::qadv::{transition_features, TRANSITION_FEATURE_COUNT};
use crate::{Action, GameState, Player};

pub const DEPLOYED_BOARD_SIZE: u8 = 7;
pub const DEPLOYED_CELL_COUNT: usize = (DEPLOYED_BOARD_SIZE * DEPLOYED_BOARD_SIZE) as usize;
pub const BOARD_FEATURE_COUNT: usize = 16;
pub const GLOBAL_FEATURE_COUNT: usize = 8;
pub const ACTION_FEATURE_COUNT: usize = 3;
pub const MAX_ACTIONS: usize = DEPLOYED_CELL_COUNT * DEPLOYED_CELL_COUNT;
pub const GNN_NODE_FEATURE_COUNT: usize = 21;
pub const GNN_GRAPH_NODE_COUNT: usize = DEPLOYED_CELL_COUNT + 4;
pub const QADV_TRANSITION_FEATURE_COUNT: usize = TRANSITION_FEATURE_COUNT;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ActionSpec {
    pub kind: u8,
    pub from: u8,
    pub to: u8,
}

impl From<Action> for ActionSpec {
    fn from(action: Action) -> Self {
        match action {
            Action::Place { to } => Self {
                kind: 0,
                from: 0,
                to,
            },
            Action::Relocate { from, to } => Self { kind: 1, from, to },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PolicyValueInputs {
    /// Channel-first, row-major board tensor: [16, 7, 7].
    pub board_features: Vec<f32>,
    /// Global feature tensor: [8].
    pub global_features: [f32; GLOBAL_FEATURE_COUNT],
    /// Padded action tensor: [2401, 3], in legal-action order.
    pub action_specs: Vec<ActionSpec>,
    /// 1.0 for a legal action slot and 0.0 for padding: [2401].
    pub action_mask: Vec<f32>,
}

impl PolicyValueInputs {
    pub fn from_state(state: GameState) -> Result<Self, String> {
        if state.config.board_size != DEPLOYED_BOARD_SIZE {
            return Err(format!(
                "deployed policy model requires {}x{} board, received {}x{}",
                DEPLOYED_BOARD_SIZE,
                DEPLOYED_BOARD_SIZE,
                state.config.board_size,
                state.config.board_size,
            ));
        }

        let cells = DEPLOYED_CELL_COUNT;
        let mut board_features = vec![0.0_f32; BOARD_FEATURE_COUNT * cells];
        let size_feature = f32::from(state.config.board_size) / 7.0;
        let denominator = f32::from(state.config.board_size.saturating_sub(1));
        for square in 0..cells as u8 {
            let row = square / state.config.board_size;
            let column = square % state.config.board_size;
            let piece_channel = match state.board_at(square) {
                None => 0,
                Some(Player::Light) => 1,
                Some(Player::Dark) => 2,
            };
            set_feature(&mut board_features, piece_channel, square, 1.0);
            set_feature(
                &mut board_features,
                3,
                square,
                f32::from((state.forbidden & (1_u64 << square)) != 0),
            );
            set_feature(
                &mut board_features,
                4,
                square,
                f32::from(state.last_relocated_to[Player::Light.index()] == Some(square)),
            );
            set_feature(
                &mut board_features,
                5,
                square,
                f32::from(state.last_relocated_to[Player::Dark.index()] == Some(square)),
            );
            set_feature(&mut board_features, 6, square, f32::from(row) / denominator);
            set_feature(
                &mut board_features,
                7,
                square,
                f32::from(column) / denominator,
            );
            set_feature(&mut board_features, 8, square, f32::from(row == 0));
            set_feature(
                &mut board_features,
                9,
                square,
                f32::from(row + 1 == state.config.board_size),
            );
            set_feature(&mut board_features, 10, square, f32::from(column == 0));
            set_feature(
                &mut board_features,
                11,
                square,
                f32::from(column + 1 == state.config.board_size),
            );
            set_feature(&mut board_features, 12, square, 1.0);
            set_feature(&mut board_features, 13, square, size_feature);
            set_feature(
                &mut board_features,
                14,
                square,
                f32::from(state.turn == Player::Light),
            );
            set_feature(
                &mut board_features,
                15,
                square,
                f32::from(state.turn == Player::Dark),
            );
        }

        let global_features = [
            f32::from(state.reserve[Player::Light.index()])
                / f32::from(state.config.reserve_per_player),
            f32::from(state.reserve[Player::Dark.index()])
                / f32::from(state.config.reserve_per_player),
            f32::from(state.turn == Player::Light),
            f32::from(state.turn == Player::Dark),
            f32::from(state.last_capture) / 4.0,
            f32::from(state.last_player == Some(Player::Light)),
            f32::from(state.last_player == Some(Player::Dark)),
            f32::from(state.ply) / f32::from(state.config.max_plies),
        ];

        let legal_actions = state.legal_actions();
        if legal_actions.len() > MAX_ACTIONS {
            return Err(format!(
                "legal action count exceeds model capacity: {}",
                legal_actions.len()
            ));
        }
        let mut action_specs = vec![
            ActionSpec {
                kind: 0,
                from: 0,
                to: 0
            };
            MAX_ACTIONS
        ];
        let mut action_mask = vec![0.0_f32; MAX_ACTIONS];
        for (index, action) in legal_actions.into_iter().enumerate() {
            action_specs[index] = action.into();
            action_mask[index] = 1.0;
        }

        Ok(Self {
            board_features,
            global_features,
            action_specs,
            action_mask,
        })
    }
}

/// Tensor ABI for the fixed 7x7 graph exported from the Python QAdv trunk.
///
/// The native graph model keeps the four typed boundary nodes explicit. Its
/// policy/value path is shared by QAdv checkpoints, and the separate
/// `GnnQAdvInputs` ABI appends deterministic transition features for the
/// action-value head.
#[derive(Clone, Debug, PartialEq)]
pub struct GnnPolicyValueInputs {
    pub node_features: Vec<f32>,
    pub global_features: [f32; GLOBAL_FEATURE_COUNT],
    pub action_specs: Vec<ActionSpec>,
    pub action_mask: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GnnQAdvInputs {
    pub node_features: Vec<f32>,
    pub global_features: [f32; GLOBAL_FEATURE_COUNT],
    pub action_specs: Vec<ActionSpec>,
    pub action_mask: Vec<f32>,
    /// Padded transition features: [2401, 24], in legal-action order.
    pub transition_features: Vec<f32>,
}

impl GnnQAdvInputs {
    pub fn from_state(state: GameState) -> Result<Self, String> {
        let base = GnnPolicyValueInputs::from_state(state)?;
        let actions = state.legal_actions();
        let rows = transition_features(state, &actions);
        let mut padded = vec![0.0_f32; MAX_ACTIONS * QADV_TRANSITION_FEATURE_COUNT];
        padded[..rows.len()].copy_from_slice(&rows);
        Ok(Self {
            node_features: base.node_features,
            global_features: base.global_features,
            action_specs: base.action_specs,
            action_mask: base.action_mask,
            transition_features: padded,
        })
    }
}

impl GnnPolicyValueInputs {
    pub fn from_state(state: GameState) -> Result<Self, String> {
        if state.config.board_size != DEPLOYED_BOARD_SIZE {
            return Err(format!(
                "GNN policy model requires {}x{} board, received {}x{}",
                DEPLOYED_BOARD_SIZE,
                DEPLOYED_BOARD_SIZE,
                state.config.board_size,
                state.config.board_size,
            ));
        }
        let mut node_features = vec![0.0_f32; GNN_GRAPH_NODE_COUNT * GNN_NODE_FEATURE_COUNT];
        let denominator = f32::from(state.config.board_size.saturating_sub(1));
        let size_feature = f32::from(state.config.board_size) / 7.0;
        let mut set = |node: usize, channel: usize, value: f32| {
            node_features[node * GNN_NODE_FEATURE_COUNT + channel] = value;
        };
        for square in 0..DEPLOYED_CELL_COUNT as u8 {
            let row = square / state.config.board_size;
            let column = square % state.config.board_size;
            let piece_channel = match state.board_at(square) {
                None => 0,
                Some(Player::Light) => 1,
                Some(Player::Dark) => 2,
            };
            set(usize::from(square), piece_channel, 1.0);
            set(
                usize::from(square),
                3,
                f32::from(state.forbidden & (1_u64 << square) != 0),
            );
            set(
                usize::from(square),
                4,
                f32::from(state.last_relocated_to[Player::Light.index()] == Some(square)),
            );
            set(
                usize::from(square),
                5,
                f32::from(state.last_relocated_to[Player::Dark.index()] == Some(square)),
            );
            set(usize::from(square), 6, f32::from(row) / denominator);
            set(usize::from(square), 7, f32::from(column) / denominator);
            set(usize::from(square), 8, f32::from(row == 0));
            set(
                usize::from(square),
                9,
                f32::from(row + 1 == state.config.board_size),
            );
            set(usize::from(square), 10, f32::from(column == 0));
            set(
                usize::from(square),
                11,
                f32::from(column + 1 == state.config.board_size),
            );
            set(usize::from(square), 12, 1.0);
            set(usize::from(square), 14, size_feature);
            set(
                usize::from(square),
                15,
                f32::from(state.turn == Player::Light),
            );
            set(
                usize::from(square),
                16,
                f32::from(state.turn == Player::Dark),
            );
        }
        for boundary in 0..4 {
            let node = DEPLOYED_CELL_COUNT + boundary;
            set(node, 13, 1.0);
            set(node, 14, size_feature);
            set(node, 17 + boundary, 1.0);
        }
        let global_features = [
            f32::from(state.reserve[Player::Light.index()])
                / f32::from(state.config.reserve_per_player),
            f32::from(state.reserve[Player::Dark.index()])
                / f32::from(state.config.reserve_per_player),
            f32::from(state.turn == Player::Light),
            f32::from(state.turn == Player::Dark),
            f32::from(state.last_capture) / 4.0,
            f32::from(state.last_player == Some(Player::Light)),
            f32::from(state.last_player == Some(Player::Dark)),
            f32::from(state.ply) / f32::from(state.config.max_plies),
        ];
        let legal_actions = state.legal_actions();
        if legal_actions.len() > MAX_ACTIONS {
            return Err(format!(
                "legal action count exceeds model capacity: {}",
                legal_actions.len()
            ));
        }
        let mut action_specs = vec![
            ActionSpec {
                kind: 0,
                from: 0,
                to: 0
            };
            MAX_ACTIONS
        ];
        let mut action_mask = vec![0.0_f32; MAX_ACTIONS];
        for (index, action) in legal_actions.into_iter().enumerate() {
            action_specs[index] = action.into();
            action_mask[index] = 1.0;
        }
        Ok(Self {
            node_features,
            global_features,
            action_specs,
            action_mask,
        })
    }
}

fn set_feature(features: &mut [f32], channel: usize, square: u8, value: f32) {
    features[channel * DEPLOYED_CELL_COUNT + usize::from(square)] = value;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BoardConfig;

    #[test]
    fn initial_position_matches_python_tensor_dimensions() {
        let inputs =
            PolicyValueInputs::from_state(GameState::new()).expect("encode initial position");
        assert_eq!(inputs.board_features.len(), 16 * 49);
        assert_eq!(inputs.global_features.len(), 8);
        assert_eq!(inputs.action_specs.len(), 2401);
        assert_eq!(
            inputs
                .action_mask
                .iter()
                .filter(|value| **value == 1.0)
                .count(),
            49
        );
        assert_eq!(
            inputs.action_specs[24],
            ActionSpec {
                kind: 0,
                from: 0,
                to: 24
            }
        );
        assert_eq!(inputs.board_features[12 * 49 + 24], 1.0);
    }

    #[test]
    fn movement_actions_preserve_rust_legal_action_order() {
        let mut state = GameState::new();
        state.light = (0..14).fold(0_u64, |mask, square| mask | (1_u64 << square));
        state.dark = (28..42).fold(0_u64, |mask, square| mask | (1_u64 << square));
        state.reserve = [0, 0];
        let inputs = PolicyValueInputs::from_state(state).expect("encode movement position");
        assert_eq!(
            inputs.action_specs[0],
            ActionSpec {
                kind: 1,
                from: 0,
                to: 14
            }
        );
        assert_eq!(
            inputs
                .action_mask
                .iter()
                .filter(|value| **value == 1.0)
                .count(),
            14 * 21
        );
    }

    #[test]
    fn ply_feature_uses_the_configured_game_limit() {
        let config = BoardConfig::new(7, 14)
            .expect("valid board config")
            .with_max_plies(196)
            .expect("valid maximum plies");
        let mut state = GameState::with_config(config);
        state.ply = 98;
        let inputs = PolicyValueInputs::from_state(state).expect("encode configured position");
        assert!((inputs.global_features[7] - 0.5).abs() < f32::EPSILON);
    }
}
