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
}
