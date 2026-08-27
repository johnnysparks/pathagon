//! Exact Rust counterpart of the Python Q/advantage transition features.
//!
//! The feature order and normalization are part of the native QAdv ONNX ABI.
//! Keep this module deliberately explicit: it is used for numerical parity
//! checks, not as a second learned evaluator.

use crate::search::{capture_opportunities, connection_distance, edge_presence, largest_component};
use crate::{Action, GameState, Player};

pub const TRANSITION_FEATURE_COUNT: usize = 24;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PositionSignals {
    own_distance: i32,
    opponent_distance: i32,
    own_component: i32,
    opponent_component: i32,
    own_threats: i32,
    opponent_threats: i32,
    own_edges: i32,
    opponent_edges: i32,
    own_pieces: i32,
    opponent_pieces: i32,
    own_reserve: i32,
    opponent_reserve: i32,
    mobility: i32,
}

fn signals(state: GameState, player: Player) -> PositionSignals {
    let opponent = player.other();
    PositionSignals {
        own_distance: connection_distance(state, player),
        opponent_distance: connection_distance(state, opponent),
        own_component: largest_component(state, player),
        opponent_component: largest_component(state, opponent),
        own_threats: capture_opportunities(state, player),
        opponent_threats: capture_opportunities(state, opponent),
        own_edges: edge_presence(state, player),
        opponent_edges: edge_presence(state, opponent),
        own_pieces: state.pieces(player).count_ones() as i32,
        opponent_pieces: state.pieces(opponent).count_ones() as i32,
        own_reserve: i32::from(state.reserve[player.index()]),
        opponent_reserve: i32::from(state.reserve[opponent.index()]),
        mobility: state.legal_actions().len() as i32,
    }
}

fn normalized_delta(value: i32, scale: i32) -> f32 {
    value as f32 / scale.max(1) as f32
}

fn neighbor_counts(state: GameState, square: u8, player: Player) -> (i32, i32, i32) {
    let mut own = 0;
    let mut opponent = 0;
    let mut empty = 0;
    let board_size = state.config.board_size;
    let row = square / board_size;
    let column = square % board_size;
    let mut visit = |next: u8| match state.board_at(next) {
        Some(piece) if piece == player => own += 1,
        Some(_) => opponent += 1,
        None => empty += 1,
    };
    if row > 0 {
        visit(square - board_size);
    }
    if row + 1 < board_size {
        visit(square + board_size);
    }
    if column > 0 {
        visit(square - 1);
    }
    if column + 1 < board_size {
        visit(square + 1);
    }
    (own, opponent, empty)
}

/// Return flat `[action][feature]` rows in the caller-supplied action order.
pub fn transition_features(state: GameState, actions: &[Action]) -> Vec<f32> {
    if actions.is_empty() {
        return Vec::new();
    }
    let player = state.turn;
    let before = signals(state, player);
    let size_scale = i32::from(state.config.cells());
    let coordinate_scale = i32::from(state.config.board_size.saturating_sub(1)).max(1);
    let mut rows = Vec::with_capacity(actions.len() * TRANSITION_FEATURE_COUNT);
    for action in actions {
        let next_state = state.apply_legal(*action).state;
        let after = signals(next_state, player);
        let destination = action.destination();
        let row = i32::from(destination / state.config.board_size);
        let column = i32::from(destination % state.config.board_size);
        let (from_row, from_column) = match action {
            Action::Place { .. } => (0.0, 0.0),
            Action::Relocate { from, .. } => (
                i32::from(*from / state.config.board_size) as f32 / coordinate_scale as f32,
                i32::from(*from % state.config.board_size) as f32 / coordinate_scale as f32,
            ),
        };
        let (own_neighbors, opponent_neighbors, empty_neighbors) =
            neighbor_counts(next_state, destination, player);
        rows.extend([
            f32::from(matches!(action, Action::Place { .. })),
            f32::from(matches!(action, Action::Relocate { .. })),
            from_row,
            from_column,
            row as f32 / coordinate_scale as f32,
            column as f32 / coordinate_scale as f32,
            normalized_delta(i32::from(next_state.last_capture), 4),
            normalized_delta(before.own_distance - after.own_distance, size_scale),
            normalized_delta(
                after.opponent_distance - before.opponent_distance,
                size_scale,
            ),
            normalized_delta(after.own_component - before.own_component, size_scale),
            normalized_delta(
                before.opponent_component - after.opponent_component,
                size_scale,
            ),
            normalized_delta(after.own_threats - before.own_threats, 4),
            normalized_delta(before.opponent_threats - after.opponent_threats, 4),
            normalized_delta(after.own_edges - before.own_edges, 2),
            normalized_delta(before.opponent_edges - after.opponent_edges, 2),
            normalized_delta(after.own_pieces - before.own_pieces, size_scale),
            normalized_delta(before.opponent_pieces - after.opponent_pieces, size_scale),
            normalized_delta(
                after.own_reserve - before.own_reserve,
                i32::from(state.config.reserve_per_player),
            ),
            normalized_delta(
                after.opponent_reserve - before.opponent_reserve,
                i32::from(state.config.reserve_per_player),
            ),
            normalized_delta(after.mobility - before.mobility, size_scale * size_scale),
            f32::from(next_state.winner == Some(player)),
            normalized_delta(own_neighbors, 4),
            normalized_delta(opponent_neighbors, 4),
            normalized_delta(empty_neighbors, 4),
        ]);
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_placement_features_have_expected_shape_and_coordinates() {
        let state = GameState::new();
        let actions = state.legal_actions();
        let features = transition_features(state, &actions);
        assert_eq!(features.len(), actions.len() * TRANSITION_FEATURE_COUNT);
        assert_eq!(&features[..2], &[1.0, 0.0]);
        assert_eq!(features[4], 0.0);
        assert_eq!(features[5], 0.0);
        let center = 24 * TRANSITION_FEATURE_COUNT;
        assert_eq!(&features[center..center + 2], &[1.0, 0.0]);
        assert!((features[center + 4] - 0.5).abs() < f32::EPSILON);
        assert!((features[center + 5] - 0.5).abs() < f32::EPSILON);
    }
}
