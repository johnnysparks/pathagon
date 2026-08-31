//! Small, inspectable action-transition policy used by research agents.
//!
//! The model is deliberately separate from the production evaluator. It scores
//! the legal afterstates supplied by the rules engine and is only a root-order
//! hint; Pathfinder's tactical filter and alpha-beta search remain authoritative.

use std::collections::HashSet;
use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::search::{
    evaluate, search_best_action_with_root_order_and_root_limit,
    search_best_action_with_root_order_and_root_limit_deadline,
    search_best_action_with_root_order_and_root_limit_deadline_progress,
    search_best_action_with_tactical_filter, search_best_action_with_tactical_filter_deadline,
    search_best_action_with_tactical_filter_deadline_progress, tactical_root_safe_actions,
    EvaluationWeights, SearchProgressCallback, SearchResult,
};
use crate::{Action, GameState, Player};

pub const FEATURE_COUNT: usize = 32;

/// A legal root action together with the policy score and the tactical facts
/// used to order it. The rules engine still owns legality; this is a
/// transparent, inspectable hint for the search root.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RankedTransitionAction {
    pub action: Action,
    pub safe: bool,
    pub immediate_win: bool,
    pub score: f32,
}

#[derive(Clone, Debug, Deserialize)]
struct DenseLayer {
    weights: Vec<Vec<f32>>,
    bias: Vec<f32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct TransitionPolicyModel {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u8,
    pub model: String,
    pub encoding: String,
    #[serde(rename = "featureOrder")]
    pub feature_order: Vec<String>,
    pub mean: Vec<f32>,
    pub scale: Vec<f32>,
    layers: Vec<DenseLayer>,
}

impl TransitionPolicyModel {
    pub fn from_json(text: &str) -> Result<Self, String> {
        let model: Self = serde_json::from_str(text).map_err(|error| error.to_string())?;
        model.validate()?;
        Ok(model)
    }

    /// Load a UTF-8 model artifact without relying on a filesystem. This is
    /// the stable constructor used by the browser/WASM package.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let text = std::str::from_utf8(bytes)
            .map_err(|error| format!("transition-policy model is not UTF-8: {error}"))?;
        Self::from_json(text)
    }

    pub fn from_path(path: &Path) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|error| format!("read {path:?}: {error}"))?;
        Self::from_json(&text)
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1
            || !matches!(
                self.model.as_str(),
                "tanh-action-state-transition-policy-v2" | "tanh-unified-move-policy-v2"
            )
        {
            return Err("unsupported transition-policy model schema".to_owned());
        }
        let expected_encoding = if self.model == "tanh-unified-move-policy-v2" {
            "virtual-offboard-source"
        } else {
            "explicit-source-kind"
        };
        if self.encoding != expected_encoding {
            return Err("transition-policy encoding does not match model".to_owned());
        }
        if self.feature_order.len() != FEATURE_COUNT
            || self.mean.len() != FEATURE_COUNT
            || self.scale.len() != FEATURE_COUNT
        {
            return Err("transition-policy feature metadata has the wrong length".to_owned());
        }
        if self.layers.len() != 3 {
            return Err("transition-policy model must contain three dense layers".to_owned());
        }
        let mut input = FEATURE_COUNT;
        for (index, layer) in self.layers.iter().enumerate() {
            if layer.weights.len() != layer.bias.len() || layer.weights.is_empty() {
                return Err(format!(
                    "transition-policy layer {index} has invalid output shape"
                ));
            }
            if layer.weights.iter().any(|row| row.len() != input) {
                return Err(format!(
                    "transition-policy layer {index} has invalid input shape"
                ));
            }
            if layer
                .weights
                .iter()
                .flatten()
                .chain(layer.bias.iter())
                .any(|value| !value.is_finite())
            {
                return Err(format!(
                    "transition-policy layer {index} contains non-finite values"
                ));
            }
            input = layer.bias.len();
        }
        if input != 1
            || self
                .scale
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err("transition-policy output or feature scale is invalid".to_owned());
        }
        Ok(())
    }

    pub fn score(&self, state: GameState, action: Action, safe: bool) -> f32 {
        let raw = action_features(
            state,
            action,
            safe,
            self.encoding == "virtual-offboard-source",
        );
        let mut values = raw
            .into_iter()
            .zip(self.mean.iter().copied().zip(self.scale.iter().copied()))
            .map(|(value, (mean, scale))| (value - mean) / scale)
            .collect::<Vec<_>>();
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let output = layer
                .weights
                .iter()
                .zip(layer.bias.iter().copied())
                .map(|(weights, bias)| {
                    weights
                        .iter()
                        .copied()
                        .zip(values.iter().copied())
                        .map(|(weight, value)| weight * value)
                        .sum::<f32>()
                        + bias
                })
                .collect::<Vec<_>>();
            values = if layer_index + 1 == self.layers.len() {
                output
            } else {
                output.into_iter().map(f32::tanh).collect()
            };
        }
        values[0]
    }

    /// Rank the tactical-safe legal root using this policy. Every returned
    /// action came from the rules engine's legal action set; immediate wins
    /// remain ahead of learned scores, matching the native research agent.
    pub fn ranked_actions(
        &self,
        state: GameState,
        weights: EvaluationWeights,
    ) -> Vec<RankedTransitionAction> {
        let fallback = tactical_root_safe_actions(state, state.turn, weights);
        if fallback.is_empty() {
            return Vec::new();
        }
        let safe_set = fallback.iter().copied().collect::<HashSet<_>>();
        let mut ranked = fallback
            .iter()
            .copied()
            .map(|action| RankedTransitionAction {
                action,
                safe: safe_set.contains(&action),
                immediate_win: state.apply_legal(action).state.winner == Some(state.turn),
                score: self.score(state, action, safe_set.contains(&action)),
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .immediate_win
                .cmp(&left.immediate_win)
                .then_with(|| right.score.total_cmp(&left.score))
                .then_with(|| left.action.order().cmp(&right.action.order()))
        });
        ranked
    }

    /// Run the same rules-authoritative search used by the native transition
    /// policy agent. On unsupported board sizes, fall back to the promoted
    /// tactical Pathfinder rather than applying a 7×7-trained hint.
    pub fn search(
        &self,
        state: GameState,
        config: crate::search::SearchConfig,
        deadline_ms: Option<u32>,
    ) -> SearchResult {
        if state.config.board_size != 7 {
            return deadline_ms.map_or_else(
                || search_best_action_with_tactical_filter(state, config),
                |deadline| {
                    search_best_action_with_tactical_filter_deadline(state, config, deadline)
                },
            );
        }
        let ranked = self.ranked_actions(state, config.weights);
        if ranked.is_empty() {
            return SearchResult {
                action: None,
                score: 0,
                nodes: 0,
                exhausted: false,
                completed_depth: 0,
                table_hits: 0,
            };
        }
        let root_order = ranked.iter().map(|item| item.action).collect::<Vec<_>>();
        let root_limit = Some(root_order.len());
        deadline_ms.map_or_else(
            || {
                search_best_action_with_root_order_and_root_limit(
                    state,
                    config,
                    &root_order,
                    false,
                    root_limit,
                )
            },
            |deadline| {
                search_best_action_with_root_order_and_root_limit_deadline(
                    state,
                    config,
                    &root_order,
                    false,
                    root_limit,
                    deadline,
                )
            },
        )
    }

    /// Deadline-aware transition-policy search with coarse browser progress.
    /// The callback is fed by the recursive search budget and receives
    /// cumulative nodes and table hits for the current pass.
    pub fn search_with_progress(
        &self,
        state: GameState,
        config: crate::search::SearchConfig,
        deadline_ms: u32,
        progress: SearchProgressCallback,
    ) -> SearchResult {
        if state.config.board_size != 7 {
            return search_best_action_with_tactical_filter_deadline_progress(
                state,
                config,
                deadline_ms,
                progress,
            );
        }
        let ranked = self.ranked_actions(state, config.weights);
        if ranked.is_empty() {
            return SearchResult {
                action: None,
                score: 0,
                nodes: 0,
                exhausted: false,
                completed_depth: 0,
                table_hits: 0,
            };
        }
        let root_order = ranked.iter().map(|item| item.action).collect::<Vec<_>>();
        search_best_action_with_root_order_and_root_limit_deadline_progress(
            state,
            config,
            &root_order,
            false,
            Some(root_order.len()),
            deadline_ms,
            progress,
        )
    }
}

/// Build the exact 32-feature vector used by the research trainer.
pub fn action_features(
    state: GameState,
    action: Action,
    safe: bool,
    virtual_source: bool,
) -> [f32; FEATURE_COUNT] {
    let player = state.turn;
    let next = state.apply_legal(action).state;
    let unit = |index: usize| {
        let mut weights = EvaluationWeights {
            path: 0,
            material: 0,
            capture: 0,
            structure: 0,
            threat: 0,
            edge: 0,
        };
        match index {
            0 => weights.path = 1,
            1 => weights.material = 1,
            2 => weights.capture = 1,
            3 => weights.structure = 1,
            4 => weights.threat = 1,
            _ => weights.edge = 1,
        }
        evaluate(next, player, weights) as f32
    };
    let destination = action.destination();
    let size = f32::from(state.config.board_size.saturating_sub(1).max(1));
    let row = destination / state.config.board_size;
    let column = destination % state.config.board_size;
    let (relocate, from_row, from_column) = match action {
        Action::Place { .. } if virtual_source => (0.0, 7.0 / size, 7.0 / size),
        Action::Place { .. } => (0.0, 0.0, 0.0),
        Action::Relocate { from, .. } => (
            1.0,
            f32::from(from / state.config.board_size) / size,
            f32::from(from % state.config.board_size) / size,
        ),
    };
    let dark = player == Player::Dark;
    let own_progress = if dark {
        f32::from(column) / size
    } else {
        f32::from(state.config.board_size - 1 - row) / size
    };
    let own_from_progress = match action {
        Action::Place { .. } => 0.0,
        Action::Relocate { from, .. } if dark => f32::from(from % state.config.board_size) / size,
        Action::Relocate { from, .. } => {
            f32::from(state.config.board_size - 1 - from / state.config.board_size) / size
        }
    };
    let edge = f32::from(
        row == 0
            || row + 1 == state.config.board_size
            || column == 0
            || column + 1 == state.config.board_size,
    );
    let corner = f32::from(
        (row == 0 || row + 1 == state.config.board_size)
            && (column == 0 || column + 1 == state.config.board_size),
    );
    [
        unit(0),
        unit(1),
        unit(2),
        unit(3),
        unit(4),
        unit(5),
        f32::from(next.last_capture) / 4.0,
        f32::from(next.winner == Some(player)),
        f32::from(safe),
        relocate,
        f32::from(row) / size,
        f32::from(column) / size,
        if virtual_source || matches!(action, Action::Relocate { .. }) {
            from_row
        } else {
            0.0
        },
        if virtual_source || matches!(action, Action::Relocate { .. }) {
            from_column
        } else {
            0.0
        },
        own_progress,
        own_from_progress,
        f32::from((i16::from(row) - 3).unsigned_abs() + (i16::from(column) - 3).unsigned_abs())
            / 6.0,
        edge,
        corner,
        f32::from(dark),
        state.pieces(player).count_ones() as f32 / f32::from(state.config.reserve_per_player),
        state.pieces(player.other()).count_ones() as f32
            / f32::from(state.config.reserve_per_player),
        f32::from(state.reserve[player.index()]) / f32::from(state.config.reserve_per_player),
        f32::from(state.reserve[player.other().index()])
            / f32::from(state.config.reserve_per_player),
        state.legal_action_count() as f32
            / (f32::from(state.config.cells()) * f32::from(state.config.cells())),
        f32::from(state.last_capture) / 4.0,
        f32::from(state.ply) / f32::from(state.config.max_plies),
        f32::from(state.last_player == Some(player)),
        f32::from((state.light | state.dark).count_ones() < 8),
        f32::from(
            (state.light | state.dark).count_ones() >= 8
                && u32::from(state.reserve[0]) + u32::from(state.reserve[1]) > 0
                && (state.light | state.dark).count_ones() < 20,
        ),
        f32::from(state.reserve[0] + state.reserve[1] == 0),
        f32::from(
            (state.light | state.dark).count_ones() >= 20
                && u32::from(state.reserve[0]) + u32::from(state.reserve[1]) > 0,
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BoardConfig;

    fn fixture() -> String {
        serde_json::json!({
            "schemaVersion": 1,
            "model": "tanh-action-state-transition-policy-v2",
            "encoding": "explicit-source-kind",
            "featureOrder": (0..FEATURE_COUNT).map(|index| format!("f{index}")).collect::<Vec<_>>(),
            "mean": vec![0.0; FEATURE_COUNT],
            "scale": vec![1.0; FEATURE_COUNT],
            "layers": [
                {"weights": vec![vec![0.0; FEATURE_COUNT]; 2], "bias": [0.0, 0.0]},
                {"weights": vec![vec![0.0; 2]; 2], "bias": [0.0, 0.0]},
                {"weights": vec![vec![0.0; 2]], "bias": [0.0]}
            ]
        })
        .to_string()
    }

    #[test]
    fn validates_and_scores_a_model() {
        let model = TransitionPolicyModel::from_json(&fixture()).expect("valid model");
        let state = GameState::with_config(BoardConfig::DEFAULT);
        assert_eq!(model.score(state, Action::Place { to: 0 }, true), 0.0);
    }

    #[test]
    fn feature_vector_matches_fixed_width() {
        let state = GameState::with_config(BoardConfig::DEFAULT);
        let action = state.legal_actions()[24];
        assert_eq!(
            action_features(state, action, true, false).len(),
            FEATURE_COUNT
        );
    }
}
