//! Generic shallow proof search for small-board tactical positions.
//!
//! This is intentionally separate from the heuristic search. It evaluates
//! only rule terminals, repetition draws, no-legal-action draws, and the
//! configured proof horizon. The search does not know about named tactics;
//! blocks and forks emerge from the legal AND/OR tree.

use std::collections::HashMap;

use crate::{Action, GameState, Player};

const MAX_SOLVER_BOARD_SIZE: u8 = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TacticalProofConfig {
    pub horizon: u8,
    pub max_nodes: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TacticalProofStats {
    pub nodes: u64,
    pub cache_hits: u64,
    pub table_entries: usize,
    pub exhausted: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TacticalProofAction {
    pub action: Action,
    /// Win/draw/loss from the root side-to-move perspective.
    pub outcome: i8,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TacticalProofAnalysis {
    pub outcome: i8,
    pub actions: Vec<TacticalProofAction>,
    pub optimal_actions: Vec<Action>,
    pub stats: TacticalProofStats,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TacticalProofResult {
    pub action: Option<Action>,
    pub outcome: i8,
    pub nodes: u64,
    pub exhausted: bool,
    pub completed_depth: u8,
    pub table_hits: u64,
}

/// Rule-relevant identity used by threefold repetition.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct EndgameRepetitionKey {
    pub light: u64,
    pub dark: u64,
    pub reserve: [u8; 2],
    pub turn: Player,
    pub forbidden: u64,
    pub last_relocated_to: [Option<u8>; 2],
}

impl From<GameState> for EndgameRepetitionKey {
    fn from(state: GameState) -> Self {
        Self {
            light: state.light,
            dark: state.dark,
            reserve: state.reserve,
            turn: state.turn,
            forbidden: state.forbidden,
            last_relocated_to: state.last_relocated_to,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Bound {
    Exact,
    Lower,
    Upper,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TableEntry {
    outcome: i8,
    bound: Bound,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct TableKey {
    state: GameState,
    depth_remaining: u8,
    history: Vec<(EndgameRepetitionKey, u8)>,
}

struct Solver {
    config: TacticalProofConfig,
    table: HashMap<TableKey, TableEntry>,
    nodes: u64,
    cache_hits: u64,
    exhausted: bool,
}

impl Solver {
    fn new(config: TacticalProofConfig) -> Self {
        Self {
            config,
            table: HashMap::new(),
            nodes: 0,
            cache_hits: 0,
            exhausted: false,
        }
    }

    fn stats(&self) -> TacticalProofStats {
        TacticalProofStats {
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

    fn table_key(
        &self,
        state: GameState,
        depth_remaining: u8,
        history: &HashMap<EndgameRepetitionKey, u8>,
    ) -> TableKey {
        TableKey {
            state,
            depth_remaining,
            history: Self::history_signature(history),
        }
    }

    fn next_history(
        history: &HashMap<EndgameRepetitionKey, u8>,
        state: GameState,
    ) -> HashMap<EndgameRepetitionKey, u8> {
        let mut next = history.clone();
        let key = EndgameRepetitionKey::from(state);
        *next.entry(key).or_default() += 1;
        next
    }

    fn ordered_actions(state: GameState) -> Vec<Action> {
        let mut actions = state.legal_actions();
        actions.sort_by_key(|action| {
            let wins_now = state.apply_legal(*action).state.winner == Some(state.turn);
            (!wins_now, action.order())
        });
        actions
    }

    fn solve(
        &mut self,
        state: GameState,
        history: &HashMap<EndgameRepetitionKey, u8>,
        depth_remaining: u8,
        mut alpha: i8,
        mut beta: i8,
    ) -> i8 {
        let key = self.table_key(state, depth_remaining, history);
        if let Some(entry) = self.table.get(&key).copied() {
            match entry.bound {
                Bound::Exact => {
                    self.cache_hits += 1;
                    return entry.outcome;
                }
                Bound::Lower => alpha = alpha.max(entry.outcome),
                Bound::Upper => beta = beta.min(entry.outcome),
            }
            if alpha >= beta {
                self.cache_hits += 1;
                return entry.outcome;
            }
        }

        if self.nodes >= self.config.max_nodes {
            self.exhausted = true;
            return 0;
        }
        let original_alpha = alpha;
        let original_beta = beta;
        self.nodes += 1;

        let outcome = if let Some(winner) = state.winner {
            if winner == state.turn {
                1
            } else {
                -1
            }
        } else if history
            .get(&EndgameRepetitionKey::from(state))
            .copied()
            .unwrap_or(0)
            >= 3
        {
            0
        } else if state.ply >= state.config.max_plies || depth_remaining == 0 {
            0
        } else {
            let actions = Self::ordered_actions(state);
            if actions.is_empty() {
                0
            } else {
                let mut best = -1_i8;
                for action in actions {
                    let child = state.apply_legal(action).state;
                    let child_outcome = self.solve(
                        child,
                        &Self::next_history(history, child),
                        depth_remaining - 1,
                        -beta,
                        -alpha,
                    );
                    let candidate = -child_outcome;
                    best = best.max(candidate);
                    alpha = alpha.max(candidate);
                    if alpha >= beta || self.exhausted {
                        break;
                    }
                }
                best
            }
        };

        if !self.exhausted {
            let bound = if outcome <= original_alpha {
                Bound::Upper
            } else if outcome >= original_beta {
                Bound::Lower
            } else {
                Bound::Exact
            };
            self.table.insert(key, TableEntry { outcome, bound });
        }
        outcome
    }
}

pub fn analyze(
    state: GameState,
    config: TacticalProofConfig,
) -> Result<TacticalProofAnalysis, String> {
    analyze_with_history(state, config, &[])
}

pub fn analyze_with_history(
    state: GameState,
    config: TacticalProofConfig,
    history: &[(EndgameRepetitionKey, u8)],
) -> Result<TacticalProofAnalysis, String> {
    if state.config.board_size > MAX_SOLVER_BOARD_SIZE {
        return Err(format!(
            "tactical proof search is limited to boards up to {MAX_SOLVER_BOARD_SIZE}x{MAX_SOLVER_BOARD_SIZE}"
        ));
    }
    let mut solver = Solver::new(config);
    let mut counts = history.iter().copied().collect::<HashMap<_, _>>();
    let root_key = EndgameRepetitionKey::from(state);
    *counts.entry(root_key).or_default() += 1;
    let outcome = solver.solve(state, &counts, config.horizon, -1, 1);
    if state.winner.is_some()
        || counts.get(&root_key).copied().unwrap_or(0) >= 3
        || state.ply >= state.config.max_plies
    {
        return Ok(TacticalProofAnalysis {
            outcome,
            actions: Vec::new(),
            optimal_actions: Vec::new(),
            stats: solver.stats(),
        });
    }
    let mut actions = Vec::new();
    for action in state.legal_actions() {
        let child = state.apply_legal(action).state;
        let child_outcome = solver.solve(
            child,
            &Solver::next_history(&counts, child),
            config.horizon.saturating_sub(1),
            -1,
            1,
        );
        actions.push(TacticalProofAction {
            action,
            outcome: -child_outcome,
        });
    }
    let optimal_actions = actions
        .iter()
        .filter(|item| item.outcome == outcome)
        .map(|item| item.action)
        .collect();
    Ok(TacticalProofAnalysis {
        outcome,
        actions,
        optimal_actions,
        stats: solver.stats(),
    })
}

pub fn search_best_action(state: GameState, config: TacticalProofConfig) -> TacticalProofResult {
    match analyze(state, config) {
        Ok(analysis) => TacticalProofResult {
            action: analysis.optimal_actions.first().copied(),
            outcome: analysis.outcome,
            nodes: analysis.stats.nodes,
            exhausted: analysis.stats.exhausted,
            completed_depth: if analysis.stats.exhausted {
                0
            } else {
                config.horizon
            },
            table_hits: analysis.stats.cache_hits,
        },
        Err(_) => TacticalProofResult {
            action: None,
            outcome: 0,
            nodes: 0,
            exhausted: true,
            completed_depth: 0,
            table_hits: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mask(squares: &[u8]) -> u64 {
        squares
            .iter()
            .fold(0, |mask, square| mask | (1_u64 << square))
    }

    fn fixture(light: &[u8], dark: &[u8]) -> GameState {
        let config = crate::BoardConfig::new(4, 5)
            .expect("valid board config")
            .with_max_plies(64)
            .expect("valid ply limit");
        GameState {
            config,
            light: mask(light),
            dark: mask(dark),
            reserve: [0, 0],
            turn: Player::Light,
            forbidden: 0,
            last_relocated_to: [None, None],
            last_capture: 0,
            last_player: None,
            winner: None,
            ply: 20,
        }
    }

    fn config() -> TacticalProofConfig {
        TacticalProofConfig {
            horizon: 3,
            max_nodes: 100_000,
        }
    }

    #[test]
    fn generic_solver_labels_immediate_block_and_fork_fixtures() {
        let cases = [
            (
                fixture(&[4, 8, 12, 2, 10], &[1, 3, 6, 9, 14]),
                1,
                vec![
                    Action::Relocate { from: 2, to: 0 },
                    Action::Relocate { from: 10, to: 0 },
                ],
            ),
            (
                fixture(&[5, 7, 9, 11, 15], &[1, 2, 3, 6, 10]),
                0,
                vec![
                    Action::Relocate { from: 5, to: 0 },
                    Action::Relocate { from: 7, to: 0 },
                    Action::Relocate { from: 9, to: 0 },
                    Action::Relocate { from: 11, to: 0 },
                    Action::Relocate { from: 15, to: 0 },
                ],
            ),
            (
                fixture(&[4, 5, 8, 10, 15], &[2, 3, 6, 9, 14]),
                1,
                vec![
                    Action::Relocate { from: 10, to: 12 },
                    Action::Relocate { from: 15, to: 12 },
                ],
            ),
        ];
        for (state, expected_outcome, expected_actions) in cases {
            let analysis = analyze(state, config()).expect("solve tactical fixture");
            assert_eq!(analysis.outcome, expected_outcome);
            assert_eq!(analysis.optimal_actions, expected_actions);
            assert!(analysis.stats.cache_hits > 0);
        }
    }

    #[test]
    fn solver_respects_threefold_history_before_search() {
        let state = GameState::with_board_size(3);
        let history = [(EndgameRepetitionKey::from(state), 2)];
        let analysis = analyze_with_history(state, config(), &history).expect("solve with history");
        assert_eq!(analysis.outcome, 0);
        assert_eq!(analysis.stats.nodes, 1);
    }

    #[test]
    fn solver_rejects_large_boards() {
        let state = GameState::new();
        assert!(analyze(state, config()).is_err());
    }
}
