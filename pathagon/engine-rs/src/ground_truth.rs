//! Strict exact/unknown endgame oracle results.
//!
//! This module is deliberately separate from the bounded tactical proof API.
//! A bounded search may be useful to choose a move, but an exhausted or
//! horizon-limited result is not a draw and must never be promoted as one.

use std::collections::HashMap;

use crate::endgame::EndgameRepetitionKey;
use crate::{Action, GameState};
use serde::{Deserialize, Serialize};

/// Four-valued result used by the golden-data pipeline.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum GroundTruthOutcome {
    Loss,
    Draw,
    Win,
    Unknown,
}

impl GroundTruthOutcome {
    pub const fn is_known(self) -> bool {
        !matches!(self, Self::Unknown)
    }

    pub const fn negate(self) -> Self {
        match self {
            Self::Loss => Self::Win,
            Self::Draw => Self::Draw,
            Self::Win => Self::Loss,
            Self::Unknown => Self::Unknown,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroundTruthConfig {
    /// Optional proof horizon. `None` means continue until a rule terminal,
    /// repetition draw, or the configured node budget.
    pub horizon: Option<u16>,
    pub max_nodes: u64,
    /// `None` gives the reusable historyless/infinite-play semantics. A value
    /// can be supplied when solving the live finite-ply rules contract.
    pub max_plies: Option<u16>,
}

impl Default for GroundTruthConfig {
    fn default() -> Self {
        Self {
            horizon: None,
            max_nodes: 100_000,
            max_plies: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroundTruthStats {
    pub nodes: u64,
    pub cache_hits: u64,
    pub table_entries: usize,
    pub exhausted: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroundTruthValue {
    pub outcome: GroundTruthOutcome,
    pub distance: Option<u16>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GroundTruthAction {
    pub action: Action,
    /// Result from the parent side-to-move perspective.
    pub outcome: GroundTruthOutcome,
    pub distance: Option<u16>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GroundTruthAnalysis {
    pub outcome: GroundTruthOutcome,
    pub distance: Option<u16>,
    pub actions: Vec<GroundTruthAction>,
    /// This is false when any legal action remains unknown. A known winning
    /// action is still useful, but must not be presented as the complete
    /// optimal-action set until every action has a proven result.
    pub optimal_actions_complete: bool,
    pub optimal_actions: Vec<Action>,
    pub stats: GroundTruthStats,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TableEntry {
    value: GroundTruthValue,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct TableKey {
    state: GameState,
    horizon: Option<u16>,
    history: Vec<(EndgameRepetitionKey, u8)>,
}

struct Solver {
    config: GroundTruthConfig,
    table: HashMap<TableKey, TableEntry>,
    nodes: u64,
    cache_hits: u64,
    exhausted: bool,
}

impl Solver {
    fn new(config: GroundTruthConfig) -> Self {
        Self {
            config,
            table: HashMap::new(),
            nodes: 0,
            cache_hits: 0,
            exhausted: false,
        }
    }

    fn stats(&self) -> GroundTruthStats {
        GroundTruthStats {
            nodes: self.nodes,
            cache_hits: self.cache_hits,
            table_entries: self.table.len(),
            exhausted: self.exhausted,
        }
    }

    fn history_signature(
        history: &HashMap<EndgameRepetitionKey, u8>,
    ) -> Vec<(EndgameRepetitionKey, u8)> {
        let mut signature = history
            .iter()
            .filter(|(_, count)| **count > 0)
            .map(|(key, count)| (*key, *count))
            .collect::<Vec<_>>();
        signature.sort_by_key(|(key, _)| *key);
        signature
    }

    fn next_history(
        history: &HashMap<EndgameRepetitionKey, u8>,
        state: GameState,
    ) -> HashMap<EndgameRepetitionKey, u8> {
        let mut next = history.clone();
        *next.entry(EndgameRepetitionKey::from(state)).or_default() += 1;
        next
    }

    fn key(
        state: GameState,
        horizon: Option<u16>,
        history: &HashMap<EndgameRepetitionKey, u8>,
    ) -> TableKey {
        TableKey {
            state,
            horizon,
            history: Self::history_signature(history),
        }
    }

    fn terminal_value(
        &self,
        state: GameState,
        history: &HashMap<EndgameRepetitionKey, u8>,
        horizon: Option<u16>,
    ) -> Option<GroundTruthValue> {
        if let Some(winner) = state.winner {
            return Some(GroundTruthValue {
                outcome: if winner == state.turn {
                    GroundTruthOutcome::Win
                } else {
                    GroundTruthOutcome::Loss
                },
                distance: Some(0),
            });
        }
        if history
            .get(&EndgameRepetitionKey::from(state))
            .copied()
            .unwrap_or(0)
            >= 3
        {
            return Some(GroundTruthValue {
                outcome: GroundTruthOutcome::Draw,
                distance: None,
            });
        }
        if self
            .config
            .max_plies
            .is_some_and(|limit| state.ply >= limit)
        {
            return Some(GroundTruthValue {
                outcome: GroundTruthOutcome::Draw,
                distance: None,
            });
        }
        if horizon == Some(0) {
            return Some(GroundTruthValue {
                outcome: GroundTruthOutcome::Unknown,
                distance: None,
            });
        }
        None
    }

    fn solve(
        &mut self,
        state: GameState,
        history: &HashMap<EndgameRepetitionKey, u8>,
        horizon: Option<u16>,
    ) -> GroundTruthValue {
        let key = Self::key(state, horizon, history);
        if let Some(entry) = self.table.get(&key).copied() {
            self.cache_hits += 1;
            return entry.value;
        }

        if let Some(value) = self.terminal_value(state, history, horizon) {
            self.table.insert(key, TableEntry { value });
            return value;
        }
        if self.nodes >= self.config.max_nodes {
            self.exhausted = true;
            return GroundTruthValue {
                outcome: GroundTruthOutcome::Unknown,
                distance: None,
            };
        }
        self.nodes += 1;

        let actions = state.legal_actions();
        if actions.is_empty() {
            let value = GroundTruthValue {
                outcome: GroundTruthOutcome::Draw,
                distance: None,
            };
            self.table.insert(key, TableEntry { value });
            return value;
        }

        let next_horizon = horizon.map(|remaining| remaining.saturating_sub(1));
        let mut labels = Vec::with_capacity(actions.len());
        for action in actions {
            let child = state.apply_legal(action).state;
            let child_value = self.solve(child, &Self::next_history(history, child), next_horizon);
            let label = GroundTruthValue {
                outcome: child_value.outcome.negate(),
                distance: child_value
                    .distance
                    .map(|distance| distance.saturating_add(1)),
            };
            labels.push(label);
            // One proven losing child is enough to prove this side wins. The
            // public analysis later evaluates every action for completeness.
            if label.outcome == GroundTruthOutcome::Win {
                break;
            }
        }
        let value = aggregate_values(&labels, labels.len() == state.legal_action_count());
        if value.outcome.is_known() {
            self.table.insert(key, TableEntry { value });
        }
        value
    }
}

fn aggregate_values(values: &[GroundTruthValue], complete: bool) -> GroundTruthValue {
    if values
        .iter()
        .any(|value| value.outcome == GroundTruthOutcome::Win)
    {
        let distance = values
            .iter()
            .filter(|value| value.outcome == GroundTruthOutcome::Win)
            .filter_map(|value| value.distance)
            .min();
        return GroundTruthValue {
            outcome: GroundTruthOutcome::Win,
            distance,
        };
    }
    if !complete || values.iter().any(|value| !value.outcome.is_known()) {
        return GroundTruthValue {
            outcome: GroundTruthOutcome::Unknown,
            distance: None,
        };
    }
    if values
        .iter()
        .all(|value| value.outcome == GroundTruthOutcome::Loss)
    {
        return GroundTruthValue {
            outcome: GroundTruthOutcome::Loss,
            distance: values.iter().filter_map(|value| value.distance).max(),
        };
    }
    GroundTruthValue {
        outcome: GroundTruthOutcome::Draw,
        distance: None,
    }
}

/// Analyze a fresh position with strict exact/unknown semantics.
pub fn analyze(state: GameState, config: GroundTruthConfig) -> GroundTruthAnalysis {
    let mut solver = Solver::new(config);
    let mut history = HashMap::new();
    *history
        .entry(EndgameRepetitionKey::from(state))
        .or_default() += 1;
    let root_horizon = config.horizon;
    let _ = solver.solve(state, &history, root_horizon);

    let actions = state.legal_actions();
    let action_count = actions.len();
    let mut labels = Vec::with_capacity(actions.len());
    let next_horizon = root_horizon.map(|remaining| remaining.saturating_sub(1));
    for action in actions.iter().copied() {
        let child = state.apply_legal(action).state;
        let child_value = solver.solve(child, &Solver::next_history(&history, child), next_horizon);
        labels.push(GroundTruthAction {
            action,
            outcome: child_value.outcome.negate(),
            distance: child_value
                .distance
                .map(|distance| distance.saturating_add(1)),
        });
    }

    let values = labels
        .iter()
        .map(|label| GroundTruthValue {
            outcome: label.outcome,
            distance: label.distance,
        })
        .collect::<Vec<_>>();
    let root = if action_count == 0 {
        solver.solve(state, &history, root_horizon)
    } else {
        aggregate_values(&values, true)
    };
    let optimal_actions_complete =
        root.outcome.is_known() && labels.iter().all(|label| label.outcome.is_known());
    let optimal_actions = if !root.outcome.is_known() {
        Vec::new()
    } else {
        labels
            .iter()
            .filter(|label| label.outcome == root.outcome)
            .map(|label| label.action)
            .collect()
    };

    GroundTruthAnalysis {
        outcome: root.outcome,
        distance: root.distance,
        actions: labels,
        optimal_actions_complete,
        optimal_actions,
        stats: solver.stats(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BoardConfig, Player};

    fn non_terminal_state() -> GameState {
        GameState {
            config: BoardConfig::new(3, 3).expect("valid config"),
            light: 1 << 6,
            dark: 1 << 2,
            reserve: [0, 0],
            turn: Player::Light,
            forbidden: 0,
            last_relocated_to: [None, None],
            last_capture: 0,
            last_player: None,
            winner: None,
            ply: 4,
        }
    }

    #[test]
    fn horizon_cutoff_is_unknown_not_draw() {
        let analysis = analyze(
            non_terminal_state(),
            GroundTruthConfig {
                horizon: Some(0),
                max_nodes: 1_000,
                max_plies: None,
            },
        );
        assert_eq!(analysis.outcome, GroundTruthOutcome::Unknown);
        assert!(!analysis.optimal_actions_complete);
        assert!(analysis
            .actions
            .iter()
            .all(|action| { action.outcome == GroundTruthOutcome::Unknown }));
    }

    #[test]
    fn node_cutoff_is_unknown_not_draw() {
        let analysis = analyze(
            non_terminal_state(),
            GroundTruthConfig {
                horizon: None,
                max_nodes: 0,
                max_plies: None,
            },
        );
        assert_eq!(analysis.outcome, GroundTruthOutcome::Unknown);
        assert!(analysis.stats.exhausted);
    }

    #[test]
    fn finite_ply_boundary_is_an_exact_draw() {
        let mut state = non_terminal_state();
        state.ply = 4;
        let analysis = analyze(
            state,
            GroundTruthConfig {
                horizon: None,
                max_nodes: 1,
                max_plies: Some(4),
            },
        );
        assert_eq!(analysis.outcome, GroundTruthOutcome::Draw);
        assert!(analysis.optimal_actions_complete);
    }

    #[test]
    fn outcomes_and_aggregation_preserve_unknown_semantics() {
        for (outcome, known, negated) in [
            (GroundTruthOutcome::Loss, true, GroundTruthOutcome::Win),
            (GroundTruthOutcome::Draw, true, GroundTruthOutcome::Draw),
            (GroundTruthOutcome::Win, true, GroundTruthOutcome::Loss),
            (
                GroundTruthOutcome::Unknown,
                false,
                GroundTruthOutcome::Unknown,
            ),
        ] {
            assert_eq!(outcome.is_known(), known);
            assert_eq!(outcome.negate(), negated);
        }
        assert_eq!(
            aggregate_values(
                &[GroundTruthValue {
                    outcome: GroundTruthOutcome::Win,
                    distance: Some(3)
                }],
                false,
            ),
            GroundTruthValue {
                outcome: GroundTruthOutcome::Win,
                distance: Some(3)
            }
        );
        assert_eq!(
            aggregate_values(
                &[
                    GroundTruthValue {
                        outcome: GroundTruthOutcome::Win,
                        distance: None
                    },
                    GroundTruthValue {
                        outcome: GroundTruthOutcome::Win,
                        distance: Some(2)
                    },
                ],
                false,
            ),
            GroundTruthValue {
                outcome: GroundTruthOutcome::Win,
                distance: Some(2)
            }
        );
        assert_eq!(
            aggregate_values(
                &[GroundTruthValue {
                    outcome: GroundTruthOutcome::Unknown,
                    distance: None
                }],
                true,
            )
            .outcome,
            GroundTruthOutcome::Unknown
        );
        assert_eq!(
            aggregate_values(
                &[GroundTruthValue {
                    outcome: GroundTruthOutcome::Loss,
                    distance: Some(2)
                }],
                true,
            ),
            GroundTruthValue {
                outcome: GroundTruthOutcome::Loss,
                distance: Some(2)
            }
        );
        assert_eq!(
            aggregate_values(
                &[
                    GroundTruthValue {
                        outcome: GroundTruthOutcome::Loss,
                        distance: Some(2)
                    },
                    GroundTruthValue {
                        outcome: GroundTruthOutcome::Loss,
                        distance: Some(4)
                    },
                ],
                true,
            )
            .distance,
            Some(4)
        );
        assert_eq!(
            aggregate_values(
                &[GroundTruthValue {
                    outcome: GroundTruthOutcome::Draw,
                    distance: None
                }],
                true,
            )
            .outcome,
            GroundTruthOutcome::Draw
        );
    }

    #[test]
    fn solver_history_cache_and_terminal_precedence_are_explicit() {
        let state = non_terminal_state();
        let mut history = HashMap::new();
        let key = EndgameRepetitionKey::from(state);
        history.insert(key, 0);
        let signature = Solver::history_signature(&history);
        assert!(signature.is_empty());
        let next = Solver::next_history(&history, state);
        assert_eq!(next[&key], 1);
        assert_eq!(Solver::key(state, Some(2), &next).horizon, Some(2));

        let mut solver = Solver::new(GroundTruthConfig::default());
        assert!(solver.terminal_value(state, &next, None).is_none());
        let horizon = solver.terminal_value(state, &next, Some(0)).unwrap();
        assert_eq!(horizon.outcome, GroundTruthOutcome::Unknown);
        let mut repeated = next.clone();
        repeated.insert(key, 3);
        assert_eq!(
            solver
                .terminal_value(state, &repeated, None)
                .unwrap()
                .outcome,
            GroundTruthOutcome::Draw
        );
        let mut finite = solver.config;
        finite.max_plies = Some(state.ply);
        let finite_solver = Solver::new(finite);
        assert_eq!(
            finite_solver
                .terminal_value(state, &next, None)
                .unwrap()
                .outcome,
            GroundTruthOutcome::Draw
        );

        let mut won = state;
        won.winner = Some(Player::Light);
        won.turn = Player::Light;
        assert_eq!(
            solver.terminal_value(won, &next, None).unwrap().outcome,
            GroundTruthOutcome::Win
        );
        won.turn = Player::Dark;
        assert_eq!(
            solver.terminal_value(won, &next, None).unwrap().outcome,
            GroundTruthOutcome::Loss
        );

        let first = solver.solve(state, &next, Some(0));
        let entries_before = solver.table.len();
        let second = solver.solve(state, &next, Some(0));
        assert_eq!(first, second);
        assert_eq!(solver.table.len(), entries_before);
        assert_eq!(solver.cache_hits, 1);
    }

    #[test]
    fn analysis_handles_terminal_and_no_legal_action_positions() {
        let mut terminal = non_terminal_state();
        terminal.winner = Some(Player::Light);
        terminal.turn = Player::Light;
        let winning = analyze(terminal, GroundTruthConfig::default());
        assert_eq!(winning.outcome, GroundTruthOutcome::Win);
        assert!(winning.actions.is_empty());
        assert!(winning.optimal_actions.is_empty());
        terminal.turn = Player::Dark;
        assert_eq!(
            analyze(terminal, GroundTruthConfig::default()).outcome,
            GroundTruthOutcome::Loss
        );

        let no_moves = GameState {
            config: BoardConfig::new(3, 4).unwrap(),
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
        let analysis = analyze(no_moves, GroundTruthConfig::default());
        assert_eq!(analysis.outcome, GroundTruthOutcome::Draw);
        assert!(analysis.optimal_actions_complete);
        assert!(analysis.actions.is_empty());
        assert_eq!(analysis.stats.nodes, 1);
    }
}
