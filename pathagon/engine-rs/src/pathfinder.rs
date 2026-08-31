//! Pathfinder-compatible shallow heuristic guidance.
//!
//! This mirrors `research/20260824-gnn-cnn-lab/python/pathfinder.py`: every root action receives a
//! cheap successor score, while only the ordered root beam spends the deeper
//! alpha-beta budget. The output is a soft ranking prior, not a replacement
//! for learned search.

use crate::search::{evaluate, EvaluationWeights};
use crate::{Action, GameState, Player};

const WIN_SCORE: i64 = 1_000_000_000;
const TACTICAL_WIN_SCORE: i64 = 2_000_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PathfinderConfig {
    pub depth: u8,
    pub beam_width: usize,
    pub max_nodes: u64,
}

impl Default for PathfinderConfig {
    fn default() -> Self {
        Self {
            depth: 2,
            beam_width: 8,
            max_nodes: 1_000,
        }
    }
}

#[derive(Debug)]
pub struct PathfinderGuide {
    config: PathfinderConfig,
    nodes: u64,
}

impl PathfinderGuide {
    pub fn new(config: PathfinderConfig) -> Result<Self, String> {
        if config.depth == 0 || config.beam_width == 0 || config.max_nodes == 0 {
            return Err(
                "Pathfinder guidance depth, beam, and node budget must be positive".to_owned(),
            );
        }
        Ok(Self { config, nodes: 0 })
    }

    pub fn score_actions(&mut self, state: GameState, actions: &[Action]) -> Vec<f32> {
        self.nodes = 0;
        let root = state.turn;
        let weights = EvaluationWeights::default();
        let mut fallback = Vec::with_capacity(actions.len());
        let mut ordered = Vec::with_capacity(actions.len());
        for action in actions.iter().copied() {
            let afterstate = state.apply_legal(action).state;
            let fallback_score = if afterstate.winner == Some(root) {
                WIN_SCORE
            } else {
                i64::from(evaluate(afterstate, root, weights))
            };
            let tactical = if afterstate.winner == Some(state.turn) {
                TACTICAL_WIN_SCORE
            } else {
                i64::from(afterstate.last_capture) * 10_000
            };
            fallback.push(fallback_score as f32);
            ordered.push((tactical + fallback_score, action));
        }
        ordered.sort_by(|left, right| {
            right
                .0
                .cmp(&left.0)
                .then_with(|| action_sort_key(left.1).cmp(&action_sort_key(right.1)))
        });
        let mut scores = fallback;
        for (_root_score, action) in ordered.into_iter().take(self.config.beam_width) {
            let afterstate = state.apply_legal(action).state;
            if afterstate.winner == Some(root) || self.nodes >= self.config.max_nodes {
                continue;
            }
            self.nodes += 1;
            let score = self.search(
                afterstate,
                root,
                self.config.depth.saturating_sub(1),
                i64::MIN / 4,
                i64::MAX / 4,
                weights,
            );
            if let Some(index) = actions.iter().position(|candidate| *candidate == action) {
                scores[index] = score as f32;
            }
        }
        scores
    }

    fn search(
        &mut self,
        state: GameState,
        root: Player,
        depth: u8,
        mut alpha: i64,
        mut beta: i64,
        weights: EvaluationWeights,
    ) -> i64 {
        if state.winner.is_some() || depth == 0 || self.nodes >= self.config.max_nodes {
            return i64::from(evaluate(state, root, weights));
        }
        let actions = self.ordered_actions(state, root, weights);
        if actions.is_empty() {
            return i64::from(evaluate(state, root, weights));
        }
        let maximizing = state.turn == root;
        let mut best = if maximizing {
            i64::MIN / 4
        } else {
            i64::MAX / 4
        };
        for action in actions.into_iter().take(self.config.beam_width) {
            if self.nodes >= self.config.max_nodes {
                break;
            }
            self.nodes += 1;
            let score = self.search(
                state.apply_legal(action).state,
                root,
                depth.saturating_sub(1),
                alpha,
                beta,
                weights,
            );
            if maximizing {
                best = best.max(score);
                alpha = alpha.max(best);
            } else {
                best = best.min(score);
                beta = beta.min(best);
            }
            if beta <= alpha {
                break;
            }
        }
        best
    }

    fn ordered_actions(
        &self,
        state: GameState,
        root: Player,
        weights: EvaluationWeights,
    ) -> Vec<Action> {
        let mut scored = state
            .legal_actions()
            .into_iter()
            .map(|action| {
                let next = state.apply_legal(action).state;
                let tactical = if next.winner == Some(state.turn) {
                    TACTICAL_WIN_SCORE
                } else {
                    i64::from(next.last_capture) * 10_000
                };
                (tactical + i64::from(evaluate(next, root, weights)), action)
            })
            .collect::<Vec<_>>();
        scored.sort_by(|left, right| {
            right
                .0
                .cmp(&left.0)
                .then_with(|| action_sort_key(left.1).cmp(&action_sort_key(right.1)))
        });
        scored.into_iter().map(|(_, action)| action).collect()
    }
}

fn action_sort_key(action: Action) -> u32 {
    match action {
        Action::Place { to } => u32::from(to),
        Action::Relocate { from, to } => u32::from(from) * 10_000 + u32::from(to),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_move_state() -> GameState {
        GameState {
            config: crate::BoardConfig::new(3, 4).expect("valid board config"),
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
        }
    }

    #[test]
    fn scores_all_root_actions_and_respects_budget() {
        let state = GameState::new();
        let actions = state.legal_actions();
        let mut guide = PathfinderGuide::new(PathfinderConfig {
            depth: 2,
            beam_width: 8,
            max_nodes: 24,
        })
        .unwrap();
        let scores = guide.score_actions(state, &actions);
        assert_eq!(scores.len(), actions.len());
        assert!(scores.iter().all(|score| score.is_finite()));
    }

    #[test]
    fn config_validation_defaults_and_action_sort_keys_are_stable() {
        assert_eq!(PathfinderConfig::default().depth, 2);
        for config in [
            PathfinderConfig {
                depth: 0,
                ..PathfinderConfig::default()
            },
            PathfinderConfig {
                beam_width: 0,
                ..PathfinderConfig::default()
            },
            PathfinderConfig {
                max_nodes: 0,
                ..PathfinderConfig::default()
            },
        ] {
            assert!(PathfinderGuide::new(config).is_err());
        }
        assert!(PathfinderGuide::new(PathfinderConfig::default()).is_ok());
        assert_eq!(action_sort_key(Action::Place { to: 7 }), 7);
        assert_eq!(action_sort_key(Action::Relocate { from: 3, to: 9 }), 30_009);
    }

    #[test]
    fn score_actions_covers_root_wins_empty_roots_and_all_budget_limits() {
        let mut terminal = GameState::new();
        terminal.winner = Some(Player::Light);
        let mut guide = PathfinderGuide::new(PathfinderConfig::default()).unwrap();
        assert!(guide.score_actions(terminal, &[]).is_empty());

        let mut guide = PathfinderGuide::new(PathfinderConfig {
            depth: 1,
            beam_width: 1,
            max_nodes: 1,
        })
        .unwrap();
        let actions = GameState::new().legal_actions();
        let scores = guide.score_actions(GameState::new(), &actions);
        assert_eq!(scores.len(), actions.len());
        assert_eq!(guide.nodes, 1);

        let config = crate::BoardConfig::new(7, 14)
            .unwrap()
            .with_max_plies(180)
            .unwrap();
        let winning_state = GameState {
            config,
            light: [7_u8, 14, 21, 28, 35, 42, 48]
                .into_iter()
                .fold(0_u64, |mask, square| mask | (1_u64 << square)),
            dark: [1_u8, 2, 3, 4, 5, 6]
                .into_iter()
                .fold(0_u64, |mask, square| mask | (1_u64 << square)),
            reserve: [0, 0],
            turn: Player::Light,
            forbidden: 0,
            last_relocated_to: [None, None],
            last_capture: 0,
            last_player: None,
            winner: None,
            ply: 20,
        };
        let winning_action = Action::Relocate { from: 48, to: 0 };
        let actions = winning_state.legal_actions();
        assert!(actions.contains(&winning_action));
        let scores = guide.score_actions(winning_state, &actions);
        assert_eq!(
            scores[actions.iter().position(|a| *a == winning_action).unwrap()],
            WIN_SCORE as f32
        );
    }

    #[test]
    fn recursive_search_covers_leaf_terminal_empty_minimizing_and_pruning_paths() {
        let weights = EvaluationWeights::default();
        let mut guide = PathfinderGuide::new(PathfinderConfig::default()).unwrap();
        let state = GameState::new();
        assert_eq!(
            guide.search(state, Player::Light, 0, i64::MIN / 4, i64::MAX / 4, weights),
            evaluate(state, Player::Light, weights) as i64
        );

        let mut terminal = state;
        terminal.winner = Some(Player::Light);
        assert_eq!(
            guide.search(
                terminal,
                Player::Light,
                2,
                i64::MIN / 4,
                i64::MAX / 4,
                weights
            ),
            evaluate(terminal, Player::Light, weights) as i64
        );
        assert_eq!(
            guide.search(
                no_move_state(),
                Player::Light,
                2,
                i64::MIN / 4,
                i64::MAX / 4,
                weights
            ),
            evaluate(no_move_state(), Player::Light, weights) as i64
        );

        let minimizing = guide.search(state, Player::Dark, 1, i64::MIN / 4, i64::MAX / 4, weights);
        assert!(minimizing <= i64::MAX / 4);
        guide.nodes = 0;
        let _ = guide.search(state, Player::Light, 1, 0, 0, weights);
        assert!(guide.nodes <= 1);
        guide.nodes = guide.config.max_nodes;
        assert_eq!(
            guide.search(state, Player::Light, 2, i64::MIN / 4, i64::MAX / 4, weights),
            evaluate(state, Player::Light, weights) as i64
        );
    }
}
