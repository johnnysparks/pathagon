use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::contract::RootQTargets;
use crate::corpus::StrategyBook;
#[cfg(feature = "inference")]
use crate::inference::{OnnxGnnPolicyValueModel, OnnxQAdvModel, PolicyValue, PolicyValueModel};
use crate::learned::LearnedBook;
#[cfg(feature = "inference")]
use crate::pathfinder::{PathfinderConfig, PathfinderGuide};
#[cfg(feature = "inference")]
use crate::puct::{
    search as puct_search,
    search_with_root_output_and_seeds_and_actions as puct_search_with_root_output_and_seeds_and_actions,
    PuctConfig,
};
use crate::search::{
    lunatic_action, search_best_action, search_best_action_with_deadline,
    search_best_action_with_root_probe, search_best_action_with_tactical_filter,
    search_best_action_with_tactical_filter_deadline, search_best_action_with_tactical_guard,
    search_best_action_with_tactical_proof, search_best_action_with_tt_order, SearchConfig,
};
#[cfg(feature = "inference")]
use crate::search::{
    ordered_root_actions_with_tactical_guard, search_best_action_with_root_order_and_root_limit,
    search_best_action_with_root_order_and_root_limit_deadline, tactical_root_safe_actions,
};
use crate::transition_policy::TransitionPolicyModel;
use crate::{bit_squares, Action, BoardConfig, GameState, Player};

fn contextual_phase_index(state: GameState) -> usize {
    let occupied = (state.light | state.dark).count_ones();
    let reserves = u32::from(state.reserve[0]) + u32::from(state.reserve[1]);
    if occupied < 8 {
        0 // opening
    } else if reserves == 0 {
        2 // movement
    } else if occupied >= 20 {
        3 // late-game
    } else {
        1 // placement
    }
}

#[derive(Clone)]
pub enum Agent {
    Random {
        id: String,
    },
    Lunatic {
        id: String,
    },
    Search {
        id: String,
        config: SearchConfig,
        book: Option<Arc<StrategyBook>>,
        deadline_ms: Option<u32>,
    },
    /// Pathfinder with a bounded exact root scout used only for experiments.
    SearchProbe {
        id: String,
        config: SearchConfig,
        probe_depth: u8,
        probe_nodes: u64,
        probe_actions: usize,
    },
    /// Pathfinder with table/killer move ordering enabled for all roots.
    SearchTtOrder {
        id: String,
        config: SearchConfig,
    },
    /// Pathfinder with the bounded immediate-threat root ordering guard.
    SearchTacticalGuard {
        id: String,
        config: SearchConfig,
    },
    /// Pathfinder restricted to roots without an immediate losing reply.
    SearchTacticalFilter {
        id: String,
        config: SearchConfig,
        deadline_ms: Option<u32>,
    },
    /// Pathfinder with a phase-conditioned evaluator. The four configs are
    /// selected from the current board state without changing legal actions.
    Contextual {
        id: String,
        configs: [SearchConfig; 4],
        deadline_ms: Option<u32>,
    },
    /// Pathfinder with phase-conditioned evaluators selected by the player to
    /// move. This is a research variant for testing color/turn asymmetry.
    ContextualByPlayer {
        id: String,
        light_configs: [SearchConfig; 4],
        dark_configs: [SearchConfig; 4],
        deadline_ms: Option<u32>,
    },
    /// Pathfinder with a compact action-transition policy used only to order
    /// the tactical-safe root. The native search remains authoritative.
    TransitionPolicy {
        id: String,
        config: SearchConfig,
        model: Arc<TransitionPolicyModel>,
        deadline_ms: Option<u32>,
    },
    /// Pathfinder with a selective, bounded rule-grounded tactical proof.
    SearchTacticalProof {
        id: String,
        config: SearchConfig,
        proof_horizon: u8,
        proof_nodes: u64,
    },
    #[cfg(feature = "inference")]
    GnnSorter {
        id: String,
        config: SearchConfig,
        top_k: usize,
        sort_all_actions: bool,
        root_limit: usize,
        min_margin: f32,
        max_heuristic_gap: i32,
        model: Arc<OnnxGnnPolicyValueModel>,
    },
    /// Pathfinder with a board-aware policy/value model ordering the complete
    /// tactical-safe root. The model never changes the legal action set.
    #[cfg(feature = "inference")]
    BoardPolicySorter {
        id: String,
        config: SearchConfig,
        model: Arc<OnnxGnnPolicyValueModel>,
        deadline_ms: Option<u32>,
    },
    /// Pathfinder with a board-aware Q/advantage head ordering the complete
    /// tactical-safe root.
    #[cfg(feature = "inference")]
    BoardQAdvSorter {
        id: String,
        config: SearchConfig,
        model: Arc<OnnxQAdvModel>,
        deadline_ms: Option<u32>,
    },
    #[cfg(feature = "inference")]
    QAdvSorter {
        id: String,
        config: SearchConfig,
        top_k: usize,
        sort_all_actions: bool,
        root_limit: usize,
        min_margin: f32,
        max_heuristic_gap: i32,
        model: Arc<OnnxQAdvModel>,
    },
    Learned {
        id: String,
        config: SearchConfig,
        book: Arc<LearnedBook>,
        minimum_visits: u32,
    },
    #[cfg(feature = "inference")]
    Gnn {
        id: String,
        config: PuctConfig,
        model: Arc<OnnxGnnPolicyValueModel>,
    },
    #[cfg(feature = "inference")]
    GnnGuided {
        id: String,
        config: GnnPlayConfig,
        model: Arc<OnnxGnnPolicyValueModel>,
    },
    #[cfg(feature = "inference")]
    GnnQAdv {
        id: String,
        config: QAdvPlayConfig,
        model: Arc<OnnxQAdvModel>,
    },
}

#[cfg(feature = "inference")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GnnPlayConfig {
    pub puct: PuctConfig,
    pub temperature_moves: u16,
    pub policy_temperature: f32,
    pub opening_moves: u16,
    pub opening_temperature: f32,
    pub opening_randomness: f32,
    pub pathfinder_guidance: f32,
    pub placement_guidance: f32,
    pub pathfinder_temperature: f32,
    pub pathfinder_depth: u8,
    pub pathfinder_beam: usize,
    pub pathfinder_nodes: u64,
}

#[cfg(feature = "inference")]
impl Default for GnnPlayConfig {
    fn default() -> Self {
        Self {
            puct: PuctConfig::default(),
            temperature_moves: 8,
            policy_temperature: 1.0,
            opening_moves: 0,
            opening_temperature: 1.0,
            opening_randomness: 0.0,
            pathfinder_guidance: 0.0,
            placement_guidance: 0.0,
            pathfinder_temperature: 1.0,
            pathfinder_depth: 2,
            pathfinder_beam: 8,
            pathfinder_nodes: 1_000,
        }
    }
}

#[cfg(feature = "inference")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QAdvPlayConfig {
    pub guided: GnnPlayConfig,
    pub qadv_weight: f32,
    pub tactical_simulations: u32,
    pub tactical_capture_threshold: u8,
    /// Optional bounded proof extension for tactical states. The proof search
    /// remains disabled when this is `None` or `Some(0)`.
    pub tactical_proof_horizon: Option<u8>,
    pub tactical_proof_nodes: u64,
}

#[cfg(feature = "inference")]
impl Default for QAdvPlayConfig {
    fn default() -> Self {
        Self {
            guided: GnnPlayConfig::default(),
            qadv_weight: 1.0,
            tactical_simulations: 0,
            tactical_capture_threshold: 1,
            tactical_proof_horizon: None,
            tactical_proof_nodes: 50_000,
        }
    }
}

impl Agent {
    pub fn random(id: impl Into<String>) -> Self {
        Self::Random { id: id.into() }
    }

    pub fn search(id: impl Into<String>, config: SearchConfig) -> Self {
        Self::Search {
            id: id.into(),
            config,
            book: None,
            deadline_ms: None,
        }
    }

    pub fn search_with_deadline(
        id: impl Into<String>,
        config: SearchConfig,
        deadline_ms: u32,
    ) -> Self {
        assert!(
            deadline_ms > 0,
            "Pathfinder search deadline must be positive"
        );
        Self::Search {
            id: id.into(),
            config,
            book: None,
            deadline_ms: Some(deadline_ms),
        }
    }

    pub fn search_probe(
        id: impl Into<String>,
        config: SearchConfig,
        probe_depth: u8,
        probe_nodes: u64,
        probe_actions: usize,
    ) -> Self {
        assert!(
            probe_depth > 0,
            "Pathfinder root probe depth must be positive"
        );
        assert!(
            probe_nodes > 0,
            "Pathfinder root probe nodes must be positive"
        );
        assert!(
            probe_actions > 0,
            "Pathfinder root probe actions must be positive"
        );
        Self::SearchProbe {
            id: id.into(),
            config,
            probe_depth,
            probe_nodes,
            probe_actions,
        }
    }

    pub fn search_tt_order(id: impl Into<String>, config: SearchConfig) -> Self {
        Self::SearchTtOrder {
            id: id.into(),
            config,
        }
    }

    pub fn search_tactical_guard(id: impl Into<String>, config: SearchConfig) -> Self {
        Self::SearchTacticalGuard {
            id: id.into(),
            config,
        }
    }

    pub fn search_tactical_filter(id: impl Into<String>, config: SearchConfig) -> Self {
        Self::SearchTacticalFilter {
            id: id.into(),
            config,
            deadline_ms: None,
        }
    }

    pub fn search_tactical_filter_with_deadline(
        id: impl Into<String>,
        config: SearchConfig,
        deadline_ms: u32,
    ) -> Self {
        assert!(deadline_ms > 0, "tactical filter deadline must be positive");
        Self::SearchTacticalFilter {
            id: id.into(),
            config,
            deadline_ms: Some(deadline_ms),
        }
    }

    pub fn contextual(id: impl Into<String>, configs: [SearchConfig; 4]) -> Self {
        Self::Contextual {
            id: id.into(),
            configs,
            deadline_ms: None,
        }
    }

    pub fn contextual_with_deadline(
        id: impl Into<String>,
        configs: [SearchConfig; 4],
        deadline_ms: u32,
    ) -> Self {
        assert!(
            deadline_ms > 0,
            "contextual search deadline must be positive"
        );
        Self::Contextual {
            id: id.into(),
            configs,
            deadline_ms: Some(deadline_ms),
        }
    }

    pub fn contextual_by_player_with_deadline(
        id: impl Into<String>,
        light_configs: [SearchConfig; 4],
        dark_configs: [SearchConfig; 4],
        deadline_ms: u32,
    ) -> Self {
        assert!(
            deadline_ms > 0,
            "contextual-by-player search deadline must be positive"
        );
        Self::ContextualByPlayer {
            id: id.into(),
            light_configs,
            dark_configs,
            deadline_ms: Some(deadline_ms),
        }
    }

    pub fn transition_policy_with_deadline(
        id: impl Into<String>,
        config: SearchConfig,
        model: Arc<TransitionPolicyModel>,
        deadline_ms: u32,
    ) -> Self {
        assert!(
            deadline_ms > 0,
            "transition policy deadline must be positive"
        );
        Self::TransitionPolicy {
            id: id.into(),
            config,
            model,
            deadline_ms: Some(deadline_ms),
        }
    }

    pub fn search_tactical_proof(
        id: impl Into<String>,
        config: SearchConfig,
        proof_horizon: u8,
        proof_nodes: u64,
    ) -> Self {
        assert!(proof_horizon > 0, "tactical proof horizon must be positive");
        assert!(
            proof_nodes > 0,
            "tactical proof node budget must be positive"
        );
        Self::SearchTacticalProof {
            id: id.into(),
            config,
            proof_horizon,
            proof_nodes,
        }
    }

    pub fn lunatic(id: impl Into<String>) -> Self {
        Self::Lunatic { id: id.into() }
    }

    pub fn with_book(self, book: Arc<StrategyBook>) -> Self {
        match self {
            Self::Search {
                id,
                config,
                deadline_ms,
                ..
            } => Self::Search {
                id,
                config,
                book: Some(book),
                deadline_ms,
            },
            random => random,
        }
    }

    pub fn learned(
        id: impl Into<String>,
        config: SearchConfig,
        book: Arc<LearnedBook>,
        minimum_visits: u32,
    ) -> Self {
        Self::Learned {
            id: id.into(),
            config,
            book,
            minimum_visits,
        }
    }

    #[cfg(feature = "inference")]
    pub fn gnn_sorter(
        id: impl Into<String>,
        config: SearchConfig,
        top_k: usize,
        model: Arc<OnnxGnnPolicyValueModel>,
    ) -> Self {
        Self::gnn_sorter_with_pool(id, config, top_k, false, 0, 0.0, 0, model)
    }

    #[cfg(feature = "inference")]
    pub fn gnn_sorter_with_pool(
        id: impl Into<String>,
        config: SearchConfig,
        top_k: usize,
        sort_all_actions: bool,
        root_limit: usize,
        min_margin: f32,
        max_heuristic_gap: i32,
        model: Arc<OnnxGnnPolicyValueModel>,
    ) -> Self {
        assert!(top_k > 0, "Pathfinder ONNX sorter top-k must be positive");
        Self::GnnSorter {
            id: id.into(),
            config,
            top_k,
            sort_all_actions,
            root_limit,
            min_margin,
            max_heuristic_gap,
            model,
        }
    }

    #[cfg(feature = "inference")]
    pub fn board_policy_sorter_with_deadline(
        id: impl Into<String>,
        config: SearchConfig,
        model: Arc<OnnxGnnPolicyValueModel>,
        deadline_ms: u32,
    ) -> Self {
        assert!(
            deadline_ms > 0,
            "board policy sorter deadline must be positive"
        );
        Self::BoardPolicySorter {
            id: id.into(),
            config,
            model,
            deadline_ms: Some(deadline_ms),
        }
    }

    #[cfg(feature = "inference")]
    pub fn board_qadv_sorter_with_deadline(
        id: impl Into<String>,
        config: SearchConfig,
        model: Arc<OnnxQAdvModel>,
        deadline_ms: u32,
    ) -> Self {
        assert!(
            deadline_ms > 0,
            "board QAdv sorter deadline must be positive"
        );
        Self::BoardQAdvSorter {
            id: id.into(),
            config,
            model,
            deadline_ms: Some(deadline_ms),
        }
    }

    #[cfg(feature = "inference")]
    pub fn qadv_sorter(
        id: impl Into<String>,
        config: SearchConfig,
        top_k: usize,
        model: Arc<OnnxQAdvModel>,
    ) -> Self {
        Self::qadv_sorter_with_pool(id, config, top_k, false, 0, 0.0, 0, model)
    }

    #[cfg(feature = "inference")]
    pub fn qadv_sorter_with_pool(
        id: impl Into<String>,
        config: SearchConfig,
        top_k: usize,
        sort_all_actions: bool,
        root_limit: usize,
        min_margin: f32,
        max_heuristic_gap: i32,
        model: Arc<OnnxQAdvModel>,
    ) -> Self {
        assert!(top_k > 0, "Pathfinder QAdv sorter top-k must be positive");
        Self::QAdvSorter {
            id: id.into(),
            config,
            top_k,
            sort_all_actions,
            root_limit,
            min_margin,
            max_heuristic_gap,
            model,
        }
    }

    #[cfg(feature = "inference")]
    pub fn gnn(
        id: impl Into<String>,
        config: PuctConfig,
        model: Arc<OnnxGnnPolicyValueModel>,
    ) -> Self {
        Self::Gnn {
            id: id.into(),
            config,
            model,
        }
    }

    pub fn id(&self) -> &str {
        match self {
            Self::Random { id }
            | Self::Lunatic { id }
            | Self::Search { id, .. }
            | Self::SearchProbe { id, .. }
            | Self::SearchTtOrder { id, .. }
            | Self::SearchTacticalGuard { id, .. }
            | Self::SearchTacticalFilter { id, .. }
            | Self::Contextual { id, .. }
            | Self::ContextualByPlayer { id, .. }
            | Self::TransitionPolicy { id, .. }
            | Self::SearchTacticalProof { id, .. }
            | Self::Learned { id, .. } => id,
            #[cfg(feature = "inference")]
            Self::Gnn { id, .. }
            | Self::GnnSorter { id, .. }
            | Self::BoardPolicySorter { id, .. }
            | Self::BoardQAdvSorter { id, .. }
            | Self::QAdvSorter { id, .. } => id,
            #[cfg(feature = "inference")]
            Self::GnnGuided { id, .. } => id,
            #[cfg(feature = "inference")]
            Self::GnnQAdv { id, .. } => id,
        }
    }

    #[cfg(feature = "inference")]
    pub fn gnn_guided(
        id: impl Into<String>,
        config: GnnPlayConfig,
        model: Arc<OnnxGnnPolicyValueModel>,
    ) -> Self {
        Self::GnnGuided {
            id: id.into(),
            config,
            model,
        }
    }

    #[cfg(feature = "inference")]
    pub fn qadv(id: impl Into<String>, config: QAdvPlayConfig, model: Arc<OnnxQAdvModel>) -> Self {
        Self::GnnQAdv {
            id: id.into(),
            config,
            model,
        }
    }

    fn choose(
        &self,
        state: GameState,
        random: &mut Mulberry32,
        history: &HashSet<RepetitionKey>,
        repetition_count: u8,
    ) -> Decision {
        match self {
            Self::Random { .. } => {
                let actions = state.legal_actions();
                Decision {
                    action: random.choose(&actions),
                    score: 0,
                    nodes: u64::from(!actions.is_empty()),
                    completed_depth: 0,
                    table_hits: 0,
                    book_hit: false,
                    root_q: None,
                }
            }
            Self::SearchProbe {
                config,
                probe_depth,
                probe_nodes,
                probe_actions,
                ..
            } => {
                let result = search_best_action_with_root_probe(
                    state,
                    *config,
                    *probe_depth,
                    *probe_nodes,
                    *probe_actions,
                );
                Decision {
                    action: result.action,
                    score: result.score,
                    nodes: result.nodes,
                    completed_depth: result.completed_depth,
                    table_hits: result.table_hits,
                    book_hit: false,
                    root_q: None,
                }
            }
            Self::SearchTtOrder { config, .. } => {
                let result = search_best_action_with_tt_order(state, *config);
                Decision {
                    action: result.action,
                    score: result.score,
                    nodes: result.nodes,
                    completed_depth: result.completed_depth,
                    table_hits: result.table_hits,
                    book_hit: false,
                    root_q: None,
                }
            }
            Self::SearchTacticalGuard { config, .. } => {
                let result = search_best_action_with_tactical_guard(state, *config);
                Decision {
                    action: result.action,
                    score: result.score,
                    nodes: result.nodes,
                    completed_depth: result.completed_depth,
                    table_hits: result.table_hits,
                    book_hit: false,
                    root_q: None,
                }
            }
            Self::SearchTacticalFilter {
                config,
                deadline_ms,
                ..
            } => {
                let result = deadline_ms.map_or_else(
                    || search_best_action_with_tactical_filter(state, *config),
                    |deadline_ms| {
                        search_best_action_with_tactical_filter_deadline(
                            state,
                            *config,
                            deadline_ms,
                        )
                    },
                );
                Decision {
                    action: result.action,
                    score: result.score,
                    nodes: result.nodes,
                    completed_depth: result.completed_depth,
                    table_hits: result.table_hits,
                    book_hit: false,
                    root_q: None,
                }
            }
            Self::Contextual {
                configs,
                deadline_ms,
                ..
            } => {
                let config = configs[contextual_phase_index(state)];
                let result = deadline_ms.map_or_else(
                    || search_best_action_with_tactical_filter(state, config),
                    |deadline_ms| {
                        search_best_action_with_tactical_filter_deadline(state, config, deadline_ms)
                    },
                );
                Decision {
                    action: result.action,
                    score: result.score,
                    nodes: result.nodes,
                    completed_depth: result.completed_depth,
                    table_hits: result.table_hits,
                    book_hit: false,
                    root_q: None,
                }
            }
            Self::ContextualByPlayer {
                light_configs,
                dark_configs,
                deadline_ms,
                ..
            } => {
                let configs = if state.turn == Player::Light {
                    light_configs
                } else {
                    dark_configs
                };
                let config = configs[contextual_phase_index(state)];
                let result = deadline_ms.map_or_else(
                    || search_best_action_with_tactical_filter(state, config),
                    |deadline_ms| {
                        search_best_action_with_tactical_filter_deadline(state, config, deadline_ms)
                    },
                );
                Decision {
                    action: result.action,
                    score: result.score,
                    nodes: result.nodes,
                    completed_depth: result.completed_depth,
                    table_hits: result.table_hits,
                    book_hit: false,
                    root_q: None,
                }
            }
            Self::TransitionPolicy {
                config,
                model,
                deadline_ms,
                ..
            } => choose_transition_policy(state, *config, model.as_ref(), *deadline_ms),
            Self::SearchTacticalProof {
                config,
                proof_horizon,
                proof_nodes,
                ..
            } => {
                let proof_history = endgame_proof_history(&history, state, repetition_count);
                let result = search_best_action_with_tactical_proof(
                    state,
                    *config,
                    *proof_horizon,
                    *proof_nodes,
                    &proof_history,
                );
                Decision {
                    action: result.action,
                    score: result.score,
                    nodes: result.nodes,
                    completed_depth: result.completed_depth,
                    table_hits: result.table_hits,
                    book_hit: false,
                    root_q: None,
                }
            }
            Self::Lunatic { .. } => {
                let result = lunatic_action(state);
                Decision {
                    action: result.action,
                    score: result.score,
                    nodes: result.nodes,
                    completed_depth: result.completed_depth,
                    table_hits: result.table_hits,
                    book_hit: false,
                    root_q: None,
                }
            }
            Self::Search {
                id,
                config,
                book,
                deadline_ms,
            } => {
                if let Some(choice) = book
                    .as_ref()
                    .and_then(|book| book.choose(id, state, config.depth))
                {
                    return Decision {
                        action: Some(choice.action),
                        score: choice.score,
                        nodes: 0,
                        completed_depth: choice.completed_depth,
                        table_hits: 0,
                        book_hit: true,
                        root_q: None,
                    };
                }
                let result = deadline_ms.map_or_else(
                    || search_best_action(state, *config),
                    |deadline_ms| search_best_action_with_deadline(state, *config, deadline_ms),
                );
                Decision {
                    action: result.action,
                    score: result.score,
                    nodes: result.nodes,
                    completed_depth: result.completed_depth,
                    table_hits: result.table_hits,
                    book_hit: false,
                    root_q: None,
                }
            }
            Self::Learned {
                config,
                book,
                minimum_visits,
                ..
            } => {
                if let Some(choice) = book.choose(state, *minimum_visits) {
                    return Decision {
                        action: Some(choice.action),
                        score: choice.points_rate_per_mille() as i32,
                        nodes: 0,
                        completed_depth: 0,
                        table_hits: 0,
                        book_hit: true,
                        root_q: None,
                    };
                }
                let result = search_best_action(state, *config);
                Decision {
                    action: result.action,
                    score: result.score,
                    nodes: result.nodes,
                    completed_depth: result.completed_depth,
                    table_hits: result.table_hits,
                    book_hit: false,
                    root_q: None,
                }
            }
            #[cfg(feature = "inference")]
            Self::GnnSorter {
                config,
                top_k,
                sort_all_actions,
                root_limit,
                min_margin,
                max_heuristic_gap,
                model,
                ..
            } => choose_gnn_sorter(
                state,
                *config,
                *top_k,
                *sort_all_actions,
                *root_limit,
                *min_margin,
                *max_heuristic_gap,
                model.as_ref(),
            ),
            #[cfg(feature = "inference")]
            Self::BoardPolicySorter {
                config,
                model,
                deadline_ms,
                ..
            } => choose_board_policy_sorter(state, *config, model.as_ref(), *deadline_ms),
            #[cfg(feature = "inference")]
            Self::BoardQAdvSorter {
                config,
                model,
                deadline_ms,
                ..
            } => choose_board_qadv_sorter(state, *config, model.as_ref(), *deadline_ms),
            #[cfg(feature = "inference")]
            Self::QAdvSorter {
                config,
                top_k,
                sort_all_actions,
                root_limit,
                min_margin,
                max_heuristic_gap,
                model,
                ..
            } => choose_qadv_sorter(
                state,
                *config,
                *top_k,
                *sort_all_actions,
                *root_limit,
                *min_margin,
                *max_heuristic_gap,
                model.as_ref(),
            ),
            #[cfg(feature = "inference")]
            Self::Gnn { config, model, .. } => {
                let result = puct_search(model.as_ref(), state, *config)
                    .unwrap_or_else(|error| panic!("native GNN PUCT failed: {error}"));
                Decision {
                    action: result.action,
                    score: (result.value * 1_000.0) as i32,
                    nodes: result
                        .evaluations
                        .iter()
                        .map(|evaluation| u64::from(evaluation.visits))
                        .sum(),
                    completed_depth: 0,
                    table_hits: 0,
                    book_hit: false,
                    root_q: result.root_q_targets().ok(),
                }
            }
            #[cfg(feature = "inference")]
            Self::GnnGuided { config, model, .. } => {
                choose_gnn_guided(state, random, history, *config, model.as_ref())
            }
            #[cfg(feature = "inference")]
            Self::GnnQAdv { config, model, .. } => choose_qadv_guided(
                state,
                random,
                history,
                repetition_count,
                *config,
                model.as_ref(),
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatchOptions {
    pub seed: u32,
    pub max_plies: u16,
    pub opening_random_plies: u16,
    pub board_size: u8,
    pub reserve_per_player: u8,
}

impl Default for MatchOptions {
    fn default() -> Self {
        Self {
            seed: 20_260_823,
            max_plies: 180,
            opening_random_plies: 2,
            board_size: 7,
            reserve_per_player: 14,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoveRecord {
    pub ply: u16,
    pub player: Player,
    pub action: Action,
    pub captured: u64,
    pub score: i32,
    pub nodes: u64,
    pub completed_depth: u8,
    pub table_hits: u64,
    pub book_hit: bool,
    pub root_q: Option<RootQTargets>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TerminationReason {
    Path,
    ThreefoldRepetition,
    MaxPlies,
    NoLegalAction,
}

impl TerminationReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Path => "path",
            Self::ThreefoldRepetition => "threefold-repetition",
            Self::MaxPlies => "max-plies",
            Self::NoLegalAction => "no-legal-action",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GameRecord {
    pub seed: u32,
    pub max_plies: u16,
    pub board_size: u8,
    pub reserve_per_player: u8,
    pub light_agent: String,
    pub dark_agent: String,
    pub light_specification: String,
    pub dark_specification: String,
    pub winner: Option<Player>,
    pub reason: TerminationReason,
    pub moves: Vec<MoveRecord>,
}

impl GameRecord {
    pub fn to_json(&self) -> String {
        let winner = self.winner.map_or("null".to_owned(), |player| {
            format!("\"{}\"", player.as_str())
        });
        let moves = self.moves.iter().map(|record| {
            let action = match record.action {
                Action::Place { to } => format!("{{\"kind\":\"place\",\"to\":{to}}}"),
                Action::Relocate { from, to } => format!("{{\"kind\":\"relocate\",\"from\":{from},\"to\":{to}}}"),
            };
            let captured = bit_squares(record.captured).iter().map(u8::to_string).collect::<Vec<_>>().join(",");
            let root_q = record.root_q.as_ref().map_or_else(String::new, |targets| {
                targets.validate().expect("generated root-Q targets are valid");
                let values = serde_json::to_string(&targets.action_values).expect("serialize root-Q action values");
                let visits = serde_json::to_string(&targets.action_visits).expect("serialize root-Q action visits");
                format!(",\"actionValues\":{values},\"actionVisits\":{visits},\"actionValueSource\":\"{}\"", crate::contract::ROOT_Q_SOURCE)
            });
            format!(
                "{{\"ply\":{},\"player\":\"{}\",\"action\":{},\"captured\":[{}],\"score\":{},\"nodes\":{},\"completedDepth\":{},\"tableHits\":{},\"bookHit\":{}{} }}",
                record.ply,
                record.player.as_str(),
                action,
                captured,
                record.score,
                record.nodes,
                record.completed_depth,
                record.table_hits,
                record.book_hit,
                root_q,
            )
        }).collect::<Vec<_>>().join(",");
        format!(
            "{{\"contractVersion\":1,\"seed\":{},\"config\":{{\"rulesVersion\":\"pathagon-rules-v1\",\"boardSize\":{},\"reservePerPlayer\":{},\"maxPlies\":{},\"repetitionLimit\":3}},\"engine\":{{\"id\":\"rust-bitboard\",\"runtime\":\"rust\",\"version\":\"1.0.0\",\"rulesVersion\":\"pathagon-rules-v1\"}},\"agents\":{{\"light\":\"{}\",\"dark\":\"{}\"}},\"agentSpecifications\":{{\"light\":{},\"dark\":{}}},\"winner\":{},\"result\":\"{}\",\"reason\":\"{}\",\"plies\":{},\"moves\":[{}]}}",
            self.seed,
            self.board_size,
            self.reserve_per_player,
            self.max_plies,
            json_escape(&self.light_agent),
            json_escape(&self.dark_agent),
            self.light_specification,
            self.dark_specification,
            winner,
            if self.winner.is_some() { "win" } else { "draw" },
            self.reason.as_str(),
            self.moves.len(),
            moves,
        )
    }
}

fn agent_spec_json(agent: &Agent) -> String {
    let (kind, name, depth, node_budget, beam, weights, tactical_proof_horizon) = match agent {
        Agent::Random { .. } => (
            "random",
            "Coin Flip",
            0,
            0,
            0,
            crate::search::EvaluationWeights::default(),
            None,
        ),
        Agent::Lunatic { .. } => (
            "heuristic",
            "Lunatic",
            1,
            0,
            0,
            crate::search::EvaluationWeights::default(),
            None,
        ),
        Agent::Search { config, .. } => (
            "search",
            "Rust Search",
            u32::from(config.depth),
            config.max_nodes,
            config.beam_width as u32,
            config.weights,
            config.tactical_proof_horizon,
        ),
        Agent::SearchProbe { config, .. } => (
            "search",
            "Rust Pathfinder · root probe",
            u32::from(config.depth),
            config.max_nodes,
            config.beam_width as u32,
            config.weights,
            config.tactical_proof_horizon,
        ),
        Agent::SearchTtOrder { config, .. } => (
            "search",
            "Rust Pathfinder · TT ordering",
            u32::from(config.depth),
            config.max_nodes,
            config.beam_width as u32,
            config.weights,
            config.tactical_proof_horizon,
        ),
        Agent::SearchTacticalGuard { config, .. } => (
            "search",
            "Rust Pathfinder · tactical root guard",
            u32::from(config.depth),
            config.max_nodes,
            config.beam_width as u32,
            config.weights,
            config.tactical_proof_horizon,
        ),
        Agent::SearchTacticalFilter { config, .. } => (
            "search",
            "Rust Pathfinder · tactical root filter",
            u32::from(config.depth),
            config.max_nodes,
            config.beam_width as u32,
            config.weights,
            config.tactical_proof_horizon,
        ),
        Agent::Contextual { configs, .. } => (
            "search",
            "Rust Pathfinder · contextual evaluator",
            u32::from(configs[0].depth),
            configs[0].max_nodes,
            configs[0].beam_width as u32,
            configs[0].weights,
            configs[0].tactical_proof_horizon,
        ),
        Agent::ContextualByPlayer { light_configs, .. } => (
            "search",
            "Rust Pathfinder · contextual evaluator by player",
            u32::from(light_configs[0].depth),
            light_configs[0].max_nodes,
            light_configs[0].beam_width as u32,
            light_configs[0].weights,
            light_configs[0].tactical_proof_horizon,
        ),
        Agent::TransitionPolicy { config, .. } => (
            "search",
            "Pathfinder · action-transition policy",
            u32::from(config.depth),
            config.max_nodes,
            config.beam_width as u32,
            config.weights,
            config.tactical_proof_horizon,
        ),
        Agent::SearchTacticalProof { config, .. } => (
            "search",
            "Rust Pathfinder · proof-guided tactical search",
            u32::from(config.depth),
            config.max_nodes,
            config.beam_width as u32,
            config.weights,
            config.tactical_proof_horizon,
        ),
        Agent::Learned { config, .. } => (
            "learned",
            "Learned",
            u32::from(config.depth),
            config.max_nodes,
            config.beam_width as u32,
            config.weights,
            config.tactical_proof_horizon,
        ),
        #[cfg(feature = "inference")]
        Agent::GnnSorter { config, .. } => (
            "search",
            "Pathfinder · ONNX root sorter",
            u32::from(config.depth),
            config.max_nodes,
            config.beam_width as u32,
            config.weights,
            config.tactical_proof_horizon,
        ),
        #[cfg(feature = "inference")]
        Agent::BoardPolicySorter { config, .. } => (
            "search",
            "Pathfinder · board-aware policy/value sorter",
            u32::from(config.depth),
            config.max_nodes,
            config.beam_width as u32,
            config.weights,
            config.tactical_proof_horizon,
        ),
        #[cfg(feature = "inference")]
        Agent::BoardQAdvSorter { config, .. } => (
            "search",
            "Pathfinder · board-aware Q/advantage sorter",
            u32::from(config.depth),
            config.max_nodes,
            config.beam_width as u32,
            config.weights,
            config.tactical_proof_horizon,
        ),
        #[cfg(feature = "inference")]
        Agent::QAdvSorter { config, .. } => (
            "search",
            "Pathfinder · ONNX QAdv root sorter",
            u32::from(config.depth),
            config.max_nodes,
            config.beam_width as u32,
            config.weights,
            config.tactical_proof_horizon,
        ),
        #[cfg(feature = "inference")]
        Agent::Gnn { config, .. } => (
            "neural",
            "The Q-Arbiter · Rust GNN policy",
            0,
            u64::from(config.simulations),
            0,
            crate::search::EvaluationWeights::default(),
            None,
        ),
        #[cfg(feature = "inference")]
        Agent::GnnGuided { config, .. } => (
            "neural",
            "The Q-Arbiter · Rust Pathfinder-guided GNN",
            0,
            u64::from(config.puct.simulations),
            config.pathfinder_beam as u32,
            crate::search::EvaluationWeights::default(),
            None,
        ),
        #[cfg(feature = "inference")]
        Agent::GnnQAdv { config, .. } => (
            "qadv",
            "The Q-Arbiter · Rust Q/Advantage action head",
            0,
            0,
            config.guided.pathfinder_beam as u32,
            crate::search::EvaluationWeights::default(),
            None,
        ),
    };
    let mut specification = serde_json::json!({
        "id": agent.id(),
        "name": name,
        "version": "1.0.0",
        "kind": kind,
        "engineId": "rust-bitboard",
        "manifest": {
            "manifestVersion": 1,
            "runtime": "rust",
            "rulesVersion": "pathagon-rules-v1",
            "evaluatorWeights": {
                "path": weights.path,
                "material": weights.material,
                "capture": weights.capture,
                "structure": weights.structure,
                "threat": weights.threat,
                "edge": weights.edge,
            },
            "depth": depth,
            "nodeBudget": node_budget,
            "beam": beam,
            "modelHash": serde_json::Value::Null,
        },
    });
    let mut parameters = serde_json::Map::new();
    if let Agent::Search {
        deadline_ms: Some(deadline_ms),
        ..
    } = agent
    {
        parameters.insert("deadlineMs".to_owned(), serde_json::json!(deadline_ms));
    }
    #[cfg(feature = "inference")]
    if let Agent::GnnQAdv { config, .. } = agent {
        parameters.insert(
            "qadvTreeSeeds".to_owned(),
            serde_json::json!(config.guided.puct.use_action_value_seeds),
        );
        parameters.insert(
            "tacticalProofHorizon".to_owned(),
            serde_json::json!(config.tactical_proof_horizon),
        );
        parameters.insert(
            "tacticalProofNodes".to_owned(),
            serde_json::json!(config.tactical_proof_nodes),
        );
        parameters.insert(
            "tacticalSimulations".to_owned(),
            serde_json::json!(config.tactical_simulations),
        );
        parameters.insert(
            "tacticalCaptureThreshold".to_owned(),
            serde_json::json!(config.tactical_capture_threshold),
        );
    }
    if let Agent::SearchProbe {
        probe_depth,
        probe_nodes,
        probe_actions,
        ..
    } = agent
    {
        parameters.insert("rootProbeDepth".to_owned(), serde_json::json!(probe_depth));
        parameters.insert("rootProbeNodes".to_owned(), serde_json::json!(probe_nodes));
        parameters.insert(
            "rootProbeActions".to_owned(),
            serde_json::json!(probe_actions),
        );
    }
    if let Agent::SearchTacticalProof {
        proof_horizon,
        proof_nodes,
        ..
    } = agent
    {
        parameters.insert("proofHorizon".to_owned(), serde_json::json!(proof_horizon));
        parameters.insert("proofNodes".to_owned(), serde_json::json!(proof_nodes));
    }
    if let Agent::SearchTacticalFilter {
        deadline_ms: Some(deadline_ms),
        ..
    } = agent
    {
        parameters.insert("deadlineMs".to_owned(), serde_json::json!(deadline_ms));
    }
    if let Agent::Contextual {
        configs,
        deadline_ms,
        ..
    } = agent
    {
        let phase_names = ["opening", "placement", "movement", "late-game"];
        let contextual_weights = phase_names
            .iter()
            .zip(configs.iter())
            .map(|(phase, config)| {
                (
                    (*phase).to_owned(),
                    serde_json::json!({
                        "path": config.weights.path,
                        "material": config.weights.material,
                        "capture": config.weights.capture,
                        "structure": config.weights.structure,
                        "threat": config.weights.threat,
                        "edge": config.weights.edge,
                    }),
                )
            })
            .collect::<serde_json::Map<_, _>>();
        parameters.insert(
            "contextualWeights".to_owned(),
            serde_json::Value::Object(contextual_weights),
        );
        if let Some(deadline_ms) = deadline_ms {
            parameters.insert("deadlineMs".to_owned(), serde_json::json!(deadline_ms));
        }
    }
    if let Agent::ContextualByPlayer {
        light_configs,
        dark_configs,
        deadline_ms,
        ..
    } = agent
    {
        let phase_names = ["opening", "placement", "movement", "late-game"];
        let vectors = |configs: &[SearchConfig; 4]| {
            phase_names
                .iter()
                .zip(configs.iter())
                .map(|(phase, config)| {
                    (
                        (*phase).to_owned(),
                        serde_json::json!({
                            "path": config.weights.path,
                            "material": config.weights.material,
                            "capture": config.weights.capture,
                            "structure": config.weights.structure,
                            "threat": config.weights.threat,
                            "edge": config.weights.edge,
                        }),
                    )
                })
                .collect::<serde_json::Map<_, _>>()
        };
        parameters.insert(
            "contextualWeightsByPlayer".to_owned(),
            serde_json::json!({
                "light": vectors(light_configs),
                "dark": vectors(dark_configs),
            }),
        );
        if let Some(deadline_ms) = deadline_ms {
            parameters.insert("deadlineMs".to_owned(), serde_json::json!(deadline_ms));
        }
    }
    if let Agent::TransitionPolicy {
        model, deadline_ms, ..
    } = agent
    {
        parameters.insert(
            "transitionPolicyModel".to_owned(),
            serde_json::json!({
                "schemaVersion": model.schema_version,
                "model": model.model.as_str(),
                "featureOrder": &model.feature_order,
            }),
        );
        parameters.insert(
            "transitionPolicyRootOrdering".to_owned(),
            serde_json::json!("tactical-safe-full-root"),
        );
        if let Some(deadline_ms) = deadline_ms {
            parameters.insert("deadlineMs".to_owned(), serde_json::json!(deadline_ms));
        }
    }
    #[cfg(feature = "inference")]
    if let Agent::GnnSorter {
        top_k,
        sort_all_actions,
        root_limit,
        min_margin,
        max_heuristic_gap,
        ..
    } = agent
    {
        parameters.insert("sorterTopK".to_owned(), serde_json::json!(top_k));
        parameters.insert("sorter".to_owned(), serde_json::json!("onnx-policy"));
        parameters.insert(
            "sorterPool".to_owned(),
            serde_json::json!(if *sort_all_actions {
                "all-legal"
            } else {
                "pathfinder-beam"
            }),
        );
        parameters.insert("sorterRootLimit".to_owned(), serde_json::json!(root_limit));
        parameters.insert("sorterMinMargin".to_owned(), serde_json::json!(min_margin));
        parameters.insert(
            "sorterMaxHeuristicGap".to_owned(),
            serde_json::json!(max_heuristic_gap),
        );
    }
    #[cfg(feature = "inference")]
    if let Agent::BoardPolicySorter { deadline_ms, .. } = agent {
        parameters.insert(
            "boardPolicyValueModel".to_owned(),
            serde_json::json!("residual-mean-message-passing-v1"),
        );
        parameters.insert(
            "rootOrdering".to_owned(),
            serde_json::json!("tactical-safe-full-root"),
        );
        if let Some(deadline_ms) = deadline_ms {
            parameters.insert("deadlineMs".to_owned(), serde_json::json!(deadline_ms));
        }
    }
    #[cfg(feature = "inference")]
    if let Agent::BoardQAdvSorter { deadline_ms, .. } = agent {
        parameters.insert(
            "boardPolicyValueModel".to_owned(),
            serde_json::json!("residual-mean-message-passing-qadv-v1"),
        );
        parameters.insert(
            "rootOrdering".to_owned(),
            serde_json::json!("tactical-safe-full-root-q-values"),
        );
        if let Some(deadline_ms) = deadline_ms {
            parameters.insert("deadlineMs".to_owned(), serde_json::json!(deadline_ms));
        }
    }
    #[cfg(feature = "inference")]
    if let Agent::QAdvSorter {
        top_k,
        sort_all_actions,
        root_limit,
        min_margin,
        max_heuristic_gap,
        ..
    } = agent
    {
        parameters.insert("sorterTopK".to_owned(), serde_json::json!(top_k));
        parameters.insert("sorter".to_owned(), serde_json::json!("onnx-qadv"));
        parameters.insert(
            "sorterPool".to_owned(),
            serde_json::json!(if *sort_all_actions {
                "all-legal"
            } else {
                "pathfinder-beam"
            }),
        );
        parameters.insert("sorterRootLimit".to_owned(), serde_json::json!(root_limit));
        parameters.insert("sorterMinMargin".to_owned(), serde_json::json!(min_margin));
        parameters.insert(
            "sorterMaxHeuristicGap".to_owned(),
            serde_json::json!(max_heuristic_gap),
        );
    }
    if let Some(horizon) = tactical_proof_horizon {
        parameters.insert(
            "tacticalProofHorizon".to_owned(),
            serde_json::json!(horizon),
        );
    }
    if !parameters.is_empty() {
        specification["parameters"] = serde_json::Value::Object(parameters);
    }
    serde_json::to_string(&specification).expect("serialize Rust agent specification")
}

pub fn play_game(light: &Agent, dark: &Agent, options: MatchOptions) -> GameRecord {
    let mut random = Mulberry32::new(options.seed);
    let config = BoardConfig::new(options.board_size, options.reserve_per_player)
        .and_then(|config| config.with_max_plies(options.max_plies))
        .expect("valid match configuration");
    let mut state = GameState::with_config(config);
    let mut moves = Vec::new();
    let mut repetitions = HashMap::<RepetitionKey, u8>::new();

    while state.winner.is_none() && state.ply < options.max_plies {
        let (repetition_count, history) = {
            let repeated = repetitions.entry(RepetitionKey::from(state)).or_default();
            *repeated += 1;
            (
                *repeated,
                repetitions.keys().copied().collect::<HashSet<_>>(),
            )
        };
        if repetition_count >= 3 {
            return record(
                light,
                dark,
                options,
                None,
                TerminationReason::ThreefoldRepetition,
                moves,
            );
        }
        let actions = state.legal_actions();
        if actions.is_empty() {
            return record(
                light,
                dark,
                options,
                None,
                TerminationReason::NoLegalAction,
                moves,
            );
        }
        let player = state.turn;
        let decision = if state.ply < options.opening_random_plies {
            Decision {
                action: random.choose(&actions),
                score: 0,
                nodes: 1,
                completed_depth: 0,
                table_hits: 0,
                book_hit: false,
                root_q: None,
            }
        } else if player == Player::Light {
            light.choose(state, &mut random, &history, repetition_count)
        } else {
            dark.choose(state, &mut random, &history, repetition_count)
        };
        let Some(action) = decision.action else {
            return record(
                light,
                dark,
                options,
                None,
                TerminationReason::NoLegalAction,
                moves,
            );
        };
        if !actions.contains(&action) {
            return record(
                light,
                dark,
                options,
                None,
                TerminationReason::NoLegalAction,
                moves,
            );
        }
        let transition = state.apply_legal(action);
        state = transition.state;
        moves.push(MoveRecord {
            ply: state.ply,
            player,
            action,
            captured: transition.captured,
            score: decision.score,
            nodes: decision.nodes,
            completed_depth: decision.completed_depth,
            table_hits: decision.table_hits,
            book_hit: decision.book_hit,
            root_q: decision.root_q,
        });
    }
    if let Some(winner) = state.winner {
        record(
            light,
            dark,
            options,
            Some(winner),
            TerminationReason::Path,
            moves,
        )
    } else {
        record(
            light,
            dark,
            options,
            None,
            TerminationReason::MaxPlies,
            moves,
        )
    }
}

#[derive(Clone, Debug)]
struct Decision {
    action: Option<Action>,
    score: i32,
    nodes: u64,
    completed_depth: u8,
    table_hits: u64,
    book_hit: bool,
    root_q: Option<RootQTargets>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct RepetitionKey {
    light: u64,
    dark: u64,
    reserve: [u8; 2],
    turn: Player,
    forbidden: u64,
    last_relocated_to: [Option<u8>; 2],
}

impl From<GameState> for RepetitionKey {
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

fn endgame_proof_history(
    history: &HashSet<RepetitionKey>,
    state: GameState,
    repetition_count: u8,
) -> Vec<(crate::endgame::EndgameRepetitionKey, u8)> {
    let root_key = RepetitionKey::from(state);
    history
        .iter()
        .filter_map(|key| {
            let count = if *key == root_key {
                repetition_count.saturating_sub(1)
            } else {
                1
            };
            (count > 0).then_some((key, count))
        })
        .map(|(key, count)| {
            (
                crate::endgame::EndgameRepetitionKey {
                    light: key.light,
                    dark: key.dark,
                    reserve: key.reserve,
                    turn: key.turn,
                    forbidden: key.forbidden,
                    last_relocated_to: key.last_relocated_to,
                },
                count,
            )
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
pub struct Mulberry32(u32);

impl Mulberry32 {
    pub const fn new(seed: u32) -> Self {
        Self(seed)
    }

    pub fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x6D2B_79F5);
        let mut result = self.0;
        result = (result ^ (result >> 15)).wrapping_mul(result | 1);
        result ^= result.wrapping_add((result ^ (result >> 7)).wrapping_mul(result | 61));
        result ^ (result >> 14)
    }

    pub fn choose<T: Copy>(&mut self, values: &[T]) -> Option<T> {
        if values.is_empty() {
            return None;
        }
        let index = ((self.next_u32() as u64 * values.len() as u64) >> 32) as usize;
        values.get(index).copied()
    }

    pub fn weighted_choose<T: Copy>(&mut self, values: &[T], weights: &[f32]) -> Option<T> {
        if values.is_empty() || values.len() != weights.len() {
            return None;
        }
        let total = weights
            .iter()
            .copied()
            .filter(|weight| weight.is_finite() && *weight > 0.0)
            .sum::<f32>();
        if total <= 0.0 {
            return self.choose(values);
        }
        let mut remaining = (self.next_u32() as f32 / 4_294_967_296.0) * total;
        for (value, weight) in values.iter().copied().zip(weights.iter().copied()) {
            if weight <= 0.0 || !weight.is_finite() {
                continue;
            }
            if remaining < weight {
                return Some(value);
            }
            remaining -= weight;
        }
        values.last().copied()
    }
}

fn choose_transition_policy(
    state: GameState,
    config: SearchConfig,
    model: &TransitionPolicyModel,
    deadline_ms: Option<u32>,
) -> Decision {
    let result = model.search(state, config, deadline_ms);
    Decision {
        action: result.action,
        score: result.score,
        nodes: result.nodes,
        completed_depth: result.completed_depth,
        table_hits: result.table_hits,
        book_hit: false,
        root_q: None,
    }
}

#[cfg(feature = "inference")]
fn choose_gnn_sorter(
    state: GameState,
    config: SearchConfig,
    top_k: usize,
    sort_all_actions: bool,
    root_limit: usize,
    min_margin: f32,
    max_heuristic_gap: i32,
    model: &OnnxGnnPolicyValueModel,
) -> Decision {
    let fallback = ordered_root_actions_with_tactical_guard(state, state.turn, config.weights);
    if fallback.is_empty() {
        return Decision {
            action: None,
            score: 0,
            nodes: 0,
            completed_depth: 0,
            table_hits: 0,
            book_hit: false,
            root_q: None,
        };
    }
    let pool_len = top_k.min(config.beam_width.max(1)).min(fallback.len());
    let sort_pool_len = if sort_all_actions {
        fallback.len()
    } else {
        pool_len
    };
    let sort_pool = &fallback[..sort_pool_len];
    let output = model
        .evaluate_with_actions(state, sort_pool)
        .unwrap_or_else(|error| panic!("native Pathfinder ONNX sorter failed: {error}"));
    let mut ranked = sort_pool
        .iter()
        .copied()
        .zip(output.policy_logits.into_iter().take(sort_pool_len))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.order().cmp(&right.0.order()))
    });
    let confidence_ok = min_margin <= 0.0
        || ranked
            .first()
            .zip(ranked.iter().find(|(action, _)| *action == sort_pool[0]))
            .is_some_and(|(best, original)| best.1 - original.1 >= min_margin);
    let heuristic_gap_ok = max_heuristic_gap <= 0
        || ranked.first().is_some_and(|(best, _)| {
            let original_score = crate::search::evaluate(
                state.apply_legal(sort_pool[0]).state,
                state.turn,
                config.weights,
            );
            let best_score =
                crate::search::evaluate(state.apply_legal(*best).state, state.turn, config.weights);
            (original_score - best_score).abs() <= max_heuristic_gap
        });
    let should_reorder = confidence_ok && heuristic_gap_ok;
    let mut root_order = if should_reorder {
        ranked
            .into_iter()
            .take(pool_len)
            .map(|(action, _logit)| action)
            .collect::<Vec<_>>()
    } else {
        sort_pool.iter().copied().take(pool_len).collect()
    };
    for action in fallback {
        if !root_order.contains(&action) {
            root_order.push(action);
        }
    }
    // Keep the incumbent search horizon unchanged for the root-sorter
    // treatment. The optional tactical leaf extension remains available to
    // future ablations, but an isolated 120-game screen did not justify
    // enabling it by default.
    let result = search_best_action_with_root_order_and_root_limit(
        state,
        config,
        &root_order,
        false,
        Some(if root_limit == 0 {
            config.beam_width.saturating_mul(2)
        } else {
            root_limit
        }),
    );
    Decision {
        action: result.action,
        score: result.score,
        nodes: result.nodes,
        completed_depth: result.completed_depth,
        table_hits: result.table_hits,
        book_hit: false,
        root_q: None,
    }
}

#[cfg(feature = "inference")]
fn choose_board_policy_sorter(
    state: GameState,
    config: SearchConfig,
    model: &OnnxGnnPolicyValueModel,
    deadline_ms: Option<u32>,
) -> Decision {
    if state.config.board_size != 7 {
        let result = deadline_ms.map_or_else(
            || search_best_action_with_tactical_filter(state, config),
            |deadline| search_best_action_with_tactical_filter_deadline(state, config, deadline),
        );
        return Decision {
            action: result.action,
            score: result.score,
            nodes: result.nodes,
            completed_depth: result.completed_depth,
            table_hits: result.table_hits,
            book_hit: false,
            root_q: None,
        };
    }
    let fallback = tactical_root_safe_actions(state, state.turn, config.weights);
    if fallback.is_empty() {
        return Decision {
            action: None,
            score: 0,
            nodes: 0,
            completed_depth: 0,
            table_hits: 0,
            book_hit: false,
            root_q: None,
        };
    }
    let output = model
        .evaluate_with_actions(state, &fallback)
        .unwrap_or_else(|error| panic!("native board policy/value sorter failed: {error}"));
    let mut ranked = fallback
        .iter()
        .copied()
        .zip(output.policy_logits.into_iter().take(fallback.len()))
        .map(|(action, logit)| {
            (
                action,
                state.apply_legal(action).state.winner == Some(state.turn),
                logit,
            )
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then_with(|| right.2.total_cmp(&left.2))
            .then_with(|| left.0.order().cmp(&right.0.order()))
    });
    let root_order = ranked
        .iter()
        .map(|(action, _, _)| *action)
        .collect::<Vec<_>>();
    let result = deadline_ms.map_or_else(
        || {
            search_best_action_with_root_order_and_root_limit(
                state,
                config,
                &root_order,
                false,
                Some(fallback.len()),
            )
        },
        |deadline| {
            search_best_action_with_root_order_and_root_limit_deadline(
                state,
                config,
                &root_order,
                false,
                Some(fallback.len()),
                deadline,
            )
        },
    );
    Decision {
        action: result.action,
        score: result.score,
        nodes: result.nodes,
        completed_depth: result.completed_depth,
        table_hits: result.table_hits,
        book_hit: false,
        root_q: None,
    }
}

#[cfg(feature = "inference")]
fn choose_board_qadv_sorter(
    state: GameState,
    config: SearchConfig,
    model: &OnnxQAdvModel,
    deadline_ms: Option<u32>,
) -> Decision {
    if state.config.board_size != 7 {
        let result = deadline_ms.map_or_else(
            || search_best_action_with_tactical_filter(state, config),
            |deadline| search_best_action_with_tactical_filter_deadline(state, config, deadline),
        );
        return Decision {
            action: result.action,
            score: result.score,
            nodes: result.nodes,
            completed_depth: result.completed_depth,
            table_hits: result.table_hits,
            book_hit: false,
            root_q: None,
        };
    }
    let fallback = tactical_root_safe_actions(state, state.turn, config.weights);
    if fallback.is_empty() {
        return Decision {
            action: None,
            score: 0,
            nodes: 0,
            completed_depth: 0,
            table_hits: 0,
            book_hit: false,
            root_q: None,
        };
    }
    let output = model
        .evaluate_qadv_with_actions(state, &fallback)
        .unwrap_or_else(|error| panic!("native board QAdv sorter failed: {error}"));
    let mut ranked = fallback
        .iter()
        .copied()
        .zip(output.q_values.into_iter().take(fallback.len()))
        .map(|(action, q_value)| {
            (
                action,
                state.apply_legal(action).state.winner == Some(state.turn),
                q_value,
            )
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .cmp(&left.1)
            .then_with(|| right.2.total_cmp(&left.2))
            .then_with(|| left.0.order().cmp(&right.0.order()))
    });
    let root_order = ranked
        .iter()
        .map(|(action, _, _)| *action)
        .collect::<Vec<_>>();
    let result = deadline_ms.map_or_else(
        || {
            search_best_action_with_root_order_and_root_limit(
                state,
                config,
                &root_order,
                false,
                Some(fallback.len()),
            )
        },
        |deadline| {
            search_best_action_with_root_order_and_root_limit_deadline(
                state,
                config,
                &root_order,
                false,
                Some(fallback.len()),
                deadline,
            )
        },
    );
    Decision {
        action: result.action,
        score: result.score,
        nodes: result.nodes,
        completed_depth: result.completed_depth,
        table_hits: result.table_hits,
        book_hit: false,
        root_q: None,
    }
}

#[cfg(feature = "inference")]
fn choose_qadv_sorter(
    state: GameState,
    config: SearchConfig,
    top_k: usize,
    sort_all_actions: bool,
    root_limit: usize,
    min_margin: f32,
    max_heuristic_gap: i32,
    model: &OnnxQAdvModel,
) -> Decision {
    let fallback = ordered_root_actions_with_tactical_guard(state, state.turn, config.weights);
    if fallback.is_empty() {
        return Decision {
            action: None,
            score: 0,
            nodes: 0,
            completed_depth: 0,
            table_hits: 0,
            book_hit: false,
            root_q: None,
        };
    }
    let pool_len = top_k.min(config.beam_width.max(1)).min(fallback.len());
    let sort_pool_len = if sort_all_actions {
        fallback.len()
    } else {
        pool_len
    };
    let sort_pool = &fallback[..sort_pool_len];
    let output = model
        .evaluate_qadv_with_actions(state, sort_pool)
        .unwrap_or_else(|error| panic!("native Pathfinder QAdv sorter failed: {error}"));
    let mut ranked = sort_pool
        .iter()
        .copied()
        .zip(output.q_values.into_iter().take(sort_pool_len))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.order().cmp(&right.0.order()))
    });
    let confidence_ok = min_margin <= 0.0
        || ranked
            .first()
            .zip(ranked.iter().find(|(action, _)| *action == sort_pool[0]))
            .is_some_and(|(best, original)| best.1 - original.1 >= min_margin);
    let heuristic_gap_ok = max_heuristic_gap <= 0
        || ranked.first().is_some_and(|(best, _)| {
            let original_score = crate::search::evaluate(
                state.apply_legal(sort_pool[0]).state,
                state.turn,
                config.weights,
            );
            let best_score =
                crate::search::evaluate(state.apply_legal(*best).state, state.turn, config.weights);
            (original_score - best_score).abs() <= max_heuristic_gap
        });
    let should_reorder = confidence_ok && heuristic_gap_ok;
    let mut root_order = if should_reorder {
        ranked
            .into_iter()
            .take(pool_len)
            .map(|(action, _q)| action)
            .collect::<Vec<_>>()
    } else {
        sort_pool.iter().copied().take(pool_len).collect()
    };
    for action in fallback {
        if !root_order.contains(&action) {
            root_order.push(action);
        }
    }
    let result = search_best_action_with_root_order_and_root_limit(
        state,
        config,
        &root_order,
        false,
        Some(if root_limit == 0 {
            config.beam_width.saturating_mul(2)
        } else {
            root_limit
        }),
    );
    Decision {
        action: result.action,
        score: result.score,
        nodes: result.nodes,
        completed_depth: result.completed_depth,
        table_hits: result.table_hits,
        book_hit: false,
        root_q: None,
    }
}

#[cfg(feature = "inference")]
fn choose_gnn_guided(
    state: GameState,
    random: &mut Mulberry32,
    history: &HashSet<RepetitionKey>,
    config: GnnPlayConfig,
    model: &OnnxGnnPolicyValueModel,
) -> Decision {
    let in_opening = state.ply < config.opening_moves;
    let effective_temperature = if in_opening {
        config.opening_temperature
    } else {
        config.policy_temperature
    };
    let result = puct_search(model, state, config.puct)
        .unwrap_or_else(|error| panic!("native guided GNN PUCT failed: {error}"));
    let actions = result
        .evaluations
        .iter()
        .map(|evaluation| evaluation.action)
        .collect::<Vec<_>>();
    let mut probabilities = visit_probabilities(&result, effective_temperature);
    probabilities = avoid_repeated_successors(state, &actions, &probabilities, history);
    let guidance_weight = if state.reserve[state.turn.index()] > 0 {
        config.placement_guidance
    } else {
        config.pathfinder_guidance
    };
    if guidance_weight > 0.0 {
        let mut guide = PathfinderGuide::new(PathfinderConfig {
            depth: config.pathfinder_depth,
            beam_width: config.pathfinder_beam,
            max_nodes: config.pathfinder_nodes,
        })
        .unwrap_or_else(|error| panic!("invalid Pathfinder guidance config: {error}"));
        let scores = guide.score_actions(state, &actions);
        let guide_probabilities = softmax_scores(&scores, config.pathfinder_temperature);
        for index in 0..probabilities.len() {
            probabilities[index] = (1.0 - guidance_weight) * probabilities[index]
                + guidance_weight * guide_probabilities[index];
        }
    }
    if in_opening && config.opening_randomness > 0.0 && !probabilities.is_empty() {
        let uniform = 1.0 / probabilities.len() as f32;
        for probability in &mut probabilities {
            *probability = (1.0 - config.opening_randomness) * *probability
                + config.opening_randomness * uniform;
        }
    }
    probabilities = avoid_repeated_successors(state, &actions, &probabilities, history);
    let action = if state.ply < config.temperature_moves {
        random.weighted_choose(&actions, &probabilities)
    } else {
        actions
            .iter()
            .enumerate()
            .max_by(|(left_index, _), (right_index, _)| {
                probabilities[*left_index]
                    .total_cmp(&probabilities[*right_index])
                    .then_with(|| {
                        actions[*right_index]
                            .order()
                            .cmp(&actions[*left_index].order())
                    })
            })
            .map(|(_, action)| *action)
    };
    Decision {
        action,
        score: (result.value * 1_000.0) as i32,
        nodes: result
            .evaluations
            .iter()
            .map(|evaluation| u64::from(evaluation.visits))
            .sum(),
        completed_depth: 0,
        table_hits: 0,
        book_hit: false,
        root_q: result.root_q_targets().ok(),
    }
}

#[cfg(feature = "inference")]
fn choose_qadv_guided(
    state: GameState,
    random: &mut Mulberry32,
    history: &HashSet<RepetitionKey>,
    repetition_count: u8,
    config: QAdvPlayConfig,
    model: &OnnxQAdvModel,
) -> Decision {
    let actions = state.legal_actions();
    if actions.is_empty() {
        return Decision {
            action: None,
            score: 0,
            nodes: 0,
            completed_depth: 0,
            table_hits: 0,
            book_hit: false,
            root_q: None,
        };
    }
    let immediate_wins = actions
        .iter()
        .copied()
        .filter(|action| state.apply_legal(*action).state.winner == Some(state.turn))
        .min_by_key(|action| action.order());
    let simulations = if config.tactical_simulations > config.guided.puct.simulations
        && is_tactical_state(state, config.tactical_capture_threshold)
    {
        config.tactical_simulations
    } else {
        config.guided.puct.simulations
    };
    let puct_config = PuctConfig {
        simulations,
        ..config.guided.puct
    };
    let output = model
        .evaluate_qadv_with_actions(state, &actions)
        .unwrap_or_else(|error| panic!("native QAdv evaluation failed: {error}"));
    let q_values = output.q_values;
    let tactical_proof_action = config
        .tactical_proof_horizon
        .filter(|horizon| *horizon > 0)
        .and_then(|horizon| {
            choose_tactical_proof_action(
                state,
                &actions,
                &q_values,
                &history,
                repetition_count,
                horizon,
                config.tactical_proof_nodes,
                config.tactical_capture_threshold,
            )
        });
    let root_output = PolicyValue {
        policy_logits: output.policy_logits,
        value: output.value,
    };
    let result = puct_search_with_root_output_and_seeds_and_actions(
        model,
        state,
        puct_config,
        Some(root_output),
        Some(q_values.clone()),
        Some(actions),
    )
    .unwrap_or_else(|error| panic!("native QAdv PUCT failed: {error}"));
    let actions = result
        .evaluations
        .iter()
        .map(|evaluation| evaluation.action)
        .collect::<Vec<_>>();
    let in_opening = state.ply < config.guided.opening_moves;
    let effective_temperature = if in_opening {
        config.guided.opening_temperature
    } else {
        config.guided.policy_temperature
    };
    let q_values = &q_values[..actions.len()];
    let q_probability = q_softmax(q_values, effective_temperature);
    let visit_probability = visit_probabilities(&result, effective_temperature);
    let q_weight = config.qadv_weight.clamp(0.0, 1.0);
    let mut probabilities = q_probability
        .iter()
        .zip(visit_probability.iter())
        .map(|(q, visits)| q_weight * q + (1.0 - q_weight) * visits)
        .collect::<Vec<_>>();
    probabilities = avoid_repeated_successors(state, &actions, &probabilities, history);
    let guidance_weight = if state.reserve[state.turn.index()] > 0 {
        config.guided.placement_guidance
    } else {
        config.guided.pathfinder_guidance
    };
    if guidance_weight > 0.0 {
        let mut guide = PathfinderGuide::new(PathfinderConfig {
            depth: config.guided.pathfinder_depth,
            beam_width: config.guided.pathfinder_beam,
            max_nodes: config.guided.pathfinder_nodes,
        })
        .unwrap_or_else(|error| panic!("invalid Pathfinder guidance config: {error}"));
        let path_scores = guide.score_actions(state, &actions);
        let path_probability = softmax_scores(&path_scores, config.guided.pathfinder_temperature);
        for index in 0..probabilities.len() {
            probabilities[index] = (1.0 - guidance_weight) * probabilities[index]
                + guidance_weight * path_probability[index];
        }
    }
    let uniform = 1.0 / actions.len() as f32;
    if in_opening && config.guided.opening_randomness > 0.0 {
        for probability in &mut probabilities {
            *probability = (1.0 - config.guided.opening_randomness) * *probability
                + config.guided.opening_randomness * uniform;
        }
    }
    probabilities = avoid_repeated_successors(state, &actions, &probabilities, history);
    let action = if let Some(immediate_win) = immediate_wins {
        Some(immediate_win)
    } else if let Some(proof_action) = tactical_proof_action {
        Some(proof_action)
    } else if state.ply < config.guided.temperature_moves {
        random.weighted_choose(&actions, &probabilities)
    } else {
        actions
            .iter()
            .enumerate()
            .max_by(|(left_index, _), (right_index, _)| {
                probabilities[*left_index]
                    .total_cmp(&probabilities[*right_index])
                    .then_with(|| {
                        actions[*right_index]
                            .order()
                            .cmp(&actions[*left_index].order())
                    })
            })
            .map(|(_, action)| *action)
    };
    Decision {
        action,
        score: if immediate_wins.is_some() {
            1_000_000_000
        } else {
            (result.value * 1_000.0) as i32
        },
        nodes: result
            .evaluations
            .iter()
            .map(|evaluation| u64::from(evaluation.visits))
            .sum(),
        completed_depth: 0,
        table_hits: 0,
        book_hit: false,
        root_q: result.root_q_targets().ok(),
    }
}

#[cfg(feature = "inference")]
fn is_tactical_state(state: GameState, capture_threshold: u8) -> bool {
    let own_tactic = state.legal_actions().iter().copied().any(|action| {
        let transition = state.apply_legal(action);
        transition.state.winner == Some(state.turn)
            || transition.captured.count_ones() >= u32::from(capture_threshold)
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

#[cfg(feature = "inference")]
fn choose_tactical_proof_action(
    state: GameState,
    actions: &[Action],
    q_values: &[f32],
    history: &HashSet<RepetitionKey>,
    repetition_count: u8,
    horizon: u8,
    max_nodes: u64,
    capture_threshold: u8,
) -> Option<Action> {
    if state.config.board_size > 7 || max_nodes == 0 || !is_tactical_state(state, capture_threshold)
    {
        return None;
    }
    let q_by_action = actions
        .iter()
        .enumerate()
        .map(|(index, action)| (*action, q_values.get(index).copied().unwrap_or(0.0)))
        .collect::<HashMap<_, _>>();
    let mut root_order = actions.to_vec();
    root_order.sort_by(|left, right| {
        let left_q = q_by_action.get(left).copied().unwrap_or(0.0);
        let right_q = q_by_action.get(right).copied().unwrap_or(0.0);
        right_q
            .total_cmp(&left_q)
            .then_with(|| left.order().cmp(&right.order()))
    });
    let root_key = RepetitionKey::from(state);
    let proof_history = history
        .iter()
        .filter_map(|key| {
            let count = if *key == root_key {
                repetition_count.saturating_sub(1)
            } else {
                1
            };
            (count > 0).then_some((key, count))
        })
        .map(|(key, count)| {
            (
                crate::endgame::EndgameRepetitionKey {
                    light: key.light,
                    dark: key.dark,
                    reserve: key.reserve,
                    turn: key.turn,
                    forbidden: key.forbidden,
                    last_relocated_to: key.last_relocated_to,
                },
                count,
            )
        })
        .collect::<Vec<_>>();
    let analysis = crate::endgame::analyze_with_history_and_root_order(
        state,
        crate::endgame::TacticalProofConfig { horizon, max_nodes },
        &proof_history,
        &root_order,
    )
    .ok()?;
    if analysis.stats.exhausted
        || analysis
            .actions
            .iter()
            .all(|item| item.outcome == analysis.outcome)
    {
        return None;
    }
    analysis
        .optimal_actions
        .iter()
        .copied()
        .max_by(|left, right| {
            let left_q = q_by_action.get(left).copied().unwrap_or(0.0);
            let right_q = q_by_action.get(right).copied().unwrap_or(0.0);
            left_q
                .total_cmp(&right_q)
                .then_with(|| right.order().cmp(&left.order()))
        })
}

#[cfg(feature = "inference")]
fn visit_probabilities(result: &crate::puct::PuctResult, temperature: f32) -> Vec<f32> {
    if result.evaluations.is_empty() {
        return Vec::new();
    }
    if temperature <= 0.0 {
        let best = result
            .evaluations
            .iter()
            .enumerate()
            .max_by_key(|(_, evaluation)| {
                (
                    evaluation.visits,
                    std::cmp::Reverse(evaluation.action.order()),
                )
            })
            .map(|(index, _)| index)
            .unwrap_or(0);
        return (0..result.evaluations.len())
            .map(|index| f32::from(index == best))
            .collect();
    }
    let mut values = result
        .evaluations
        .iter()
        .map(|evaluation| (evaluation.visits as f32).powf(1.0 / temperature))
        .collect::<Vec<_>>();
    let total = values.iter().sum::<f32>();
    if total > 0.0 {
        values.iter_mut().for_each(|value| *value /= total);
    } else {
        let uniform = 1.0 / values.len() as f32;
        values.iter_mut().for_each(|value| *value = uniform);
    }
    values
}

#[cfg(feature = "inference")]
fn q_softmax(values: &[f32], temperature: f32) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    if temperature <= 0.0 {
        let best = values
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .unwrap_or(0);
        return (0..values.len())
            .map(|index| f32::from(index == best))
            .collect();
    }
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut weights = values
        .iter()
        .map(|value| ((*value - maximum) / temperature).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f32>();
    if total > 0.0 && total.is_finite() {
        weights.iter_mut().for_each(|weight| *weight /= total);
    } else {
        let uniform = 1.0 / weights.len() as f32;
        weights.iter_mut().for_each(|weight| *weight = uniform);
    }
    weights
}

#[cfg(feature = "inference")]
fn softmax_scores(scores: &[f32], temperature: f32) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    if temperature <= 0.0 {
        let best = scores
            .iter()
            .enumerate()
            .max_by(|(left_index, left), (right_index, right)| {
                left.total_cmp(right)
                    .then_with(|| right_index.cmp(left_index))
            })
            .map(|(index, _)| index)
            .unwrap_or(0);
        return (0..scores.len())
            .map(|index| f32::from(index == best))
            .collect();
    }
    let scale = (3_500.0 * temperature).max(1.0);
    let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut weights = scores
        .iter()
        .map(|score| ((*score - maximum) / scale).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f32>();
    if total > 0.0 && total.is_finite() {
        weights.iter_mut().for_each(|weight| *weight /= total);
    }
    weights
}

#[cfg(feature = "inference")]
fn avoid_repeated_successors(
    state: GameState,
    actions: &[Action],
    probabilities: &[f32],
    history: &HashSet<RepetitionKey>,
) -> Vec<f32> {
    let safe = actions
        .iter()
        .enumerate()
        .filter_map(|(index, action)| {
            (!history.contains(&RepetitionKey::from(state.apply_legal(*action).state)))
                .then_some(index)
        })
        .collect::<Vec<_>>();
    if safe.is_empty() || safe.len() == actions.len() {
        return probabilities.to_vec();
    }
    let total = safe.iter().map(|index| probabilities[*index]).sum::<f32>();
    let mut filtered = vec![0.0; actions.len()];
    if total > 0.0 {
        for index in safe {
            filtered[index] = probabilities[index] / total;
        }
    } else {
        let uniform = 1.0 / safe.len() as f32;
        for index in safe {
            filtered[index] = uniform;
        }
    }
    filtered
}

fn record(
    light: &Agent,
    dark: &Agent,
    options: MatchOptions,
    winner: Option<Player>,
    reason: TerminationReason,
    moves: Vec<MoveRecord>,
) -> GameRecord {
    GameRecord {
        seed: options.seed,
        max_plies: options.max_plies,
        board_size: options.board_size,
        reserve_per_player: options.reserve_per_player,
        light_agent: light.id().to_owned(),
        dark_agent: dark.id().to_owned(),
        light_specification: agent_spec_json(light),
        dark_specification: agent_spec_json(dark),
        winner,
        reason,
        moves,
    }
}

fn json_escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::EvaluationWeights;

    #[test]
    fn seeded_random_games_are_reproducible() {
        let light = Agent::random("light-random");
        let dark = Agent::random("dark-random");
        let options = MatchOptions {
            seed: 42,
            max_plies: 60,
            opening_random_plies: 2,
            ..MatchOptions::default()
        };
        assert_eq!(
            play_game(&light, &dark, options),
            play_game(&light, &dark, options)
        );
    }

    #[test]
    fn lunatic_games_are_reproducible_and_nonempty() {
        let light = Agent::lunatic("lunatic-v0.1.0");
        let dark = Agent::random("dark-random");
        let options = MatchOptions {
            seed: 4242,
            max_plies: 60,
            opening_random_plies: 2,
            ..MatchOptions::default()
        };
        let record = play_game(&light, &dark, options);
        assert!(!record.moves.is_empty());
        assert_eq!(record, play_game(&light, &dark, options));
        assert_eq!(record.moves.first().unwrap().ply, 1);
    }

    #[test]
    fn generated_records_include_manifest_backed_agent_identity() {
        let light = Agent::search_tactical_filter(
            "pathfinder-v0.4.0-tactical-filter",
            SearchConfig::default(),
        );
        let dark = Agent::random("coin-flip-seeded");
        let record = play_game(
            &light,
            &dark,
            MatchOptions {
                seed: 7,
                max_plies: 8,
                opening_random_plies: 0,
                board_size: 4,
                reserve_per_player: 8,
            },
        );
        let replay = crate::contract::ReplayRecord::from_json(&record.to_json())
            .expect("generated replay follows contract");
        assert_eq!(replay.agent_specifications.light.manifest.runtime, "rust");
        assert_eq!(
            replay.agent_specifications.light.manifest.node_budget,
            2_000
        );
    }

    #[test]
    fn contextual_agent_selects_phase_and_records_all_weight_vectors() {
        let weights = EvaluationWeights::default();
        let agent = Agent::contextual_with_deadline(
            "contextual-test",
            [
                SearchConfig {
                    depth: 1,
                    max_nodes: 32,
                    beam_width: 2,
                    weights,
                    tactical_proof_horizon: None,
                },
                SearchConfig {
                    depth: 1,
                    max_nodes: 32,
                    beam_width: 2,
                    weights,
                    tactical_proof_horizon: None,
                },
                SearchConfig {
                    depth: 1,
                    max_nodes: 32,
                    beam_width: 2,
                    weights,
                    tactical_proof_horizon: None,
                },
                SearchConfig {
                    depth: 1,
                    max_nodes: 32,
                    beam_width: 2,
                    weights,
                    tactical_proof_horizon: None,
                },
            ],
            50,
        );
        let opponent = Agent::random("contextual-random");
        let record = play_game(
            &agent,
            &opponent,
            MatchOptions {
                seed: 91,
                max_plies: 8,
                opening_random_plies: 0,
                board_size: 4,
                reserve_per_player: 8,
            },
        );
        let replay = crate::contract::ReplayRecord::from_json(&record.to_json())
            .expect("contextual replay follows contract");
        let parameters = replay
            .agent_specifications
            .light
            .parameters
            .expect("contextual parameters");
        assert_eq!(parameters["deadlineMs"], serde_json::json!(50));
        assert_eq!(
            parameters["contextualWeights"]["opening"]["path"],
            serde_json::json!(240)
        );
        assert_eq!(
            parameters["contextualWeights"]["late-game"]["capture"],
            serde_json::json!(700)
        );
    }

    #[test]
    fn contextual_phase_boundaries_match_training_labels() {
        let mut state = GameState::with_board_size(7);
        assert_eq!(contextual_phase_index(state), 0);
        state.light = (1_u64 << 0) | (1_u64 << 1) | (1_u64 << 2) | (1_u64 << 3);
        state.dark = (1_u64 << 7) | (1_u64 << 8) | (1_u64 << 9) | (1_u64 << 10);
        assert_eq!(contextual_phase_index(state), 1);
        state.reserve = [0, 0];
        assert_eq!(contextual_phase_index(state), 2);
        state.light |= (1_u64 << 4)
            | (1_u64 << 5)
            | (1_u64 << 6)
            | (1_u64 << 11)
            | (1_u64 << 12)
            | (1_u64 << 13)
            | (1_u64 << 14)
            | (1_u64 << 15)
            | (1_u64 << 16)
            | (1_u64 << 17)
            | (1_u64 << 18)
            | (1_u64 << 19);
        state.reserve = [1, 1];
        assert_eq!(contextual_phase_index(state), 3);
    }

    #[test]
    fn root_q_targets_round_trip_through_archive_contract() {
        let light = Agent::random("light-random");
        let dark = Agent::random("dark-random");
        let mut record = play_game(
            &light,
            &dark,
            MatchOptions {
                seed: 17,
                max_plies: 8,
                opening_random_plies: 0,
                board_size: 4,
                reserve_per_player: 8,
            },
        );
        record.moves[0].root_q =
            Some(RootQTargets::new(vec![-0.25, 0.75], vec![2, 10]).expect("valid root-Q targets"));

        let replay = crate::contract::ReplayRecord::from_json(&record.to_json())
            .expect("root-Q archive follows contract");
        assert_eq!(replay.moves[0].action_values, Some(vec![-0.25, 0.75]));
        assert_eq!(replay.moves[0].action_visits, Some(vec![2, 10]));
        assert_eq!(
            replay.moves[0].action_value_source.as_deref(),
            Some(crate::contract::ROOT_Q_SOURCE)
        );
    }

    #[test]
    fn mulberry_matches_javascript_reference_values() {
        let mut random = Mulberry32::new(42);
        assert_eq!(random.next_u32(), 2_581_720_956);
        assert_eq!(random.next_u32(), 1_925_393_290);
        assert_eq!(random.next_u32(), 3_661_312_704);
    }

    #[cfg(feature = "inference")]
    #[test]
    fn qadv_ordered_proof_overrides_a_bad_q_on_a_seven_by_seven_win() {
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
        let actions = state.legal_actions();
        let winning = Action::Relocate { from: 48, to: 0 };
        assert!(actions.contains(&winning));
        // Make the evaluator rank the forced win last. The proof layer must
        // still return it because QAdv only orders the rule-grounded search.
        let q_values = actions
            .iter()
            .map(|action| if *action == winning { -1.0 } else { 1.0 })
            .collect::<Vec<_>>();
        let selected = choose_tactical_proof_action(
            state,
            &actions,
            &q_values,
            &HashSet::new(),
            1,
            1,
            5_000,
            1,
        );
        assert_eq!(selected, Some(winning));
    }
}
