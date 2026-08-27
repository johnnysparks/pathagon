use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::contract::RootQTargets;
use crate::corpus::StrategyBook;
#[cfg(feature = "inference")]
use crate::inference::{OnnxGnnPolicyValueModel, OnnxQAdvModel, PolicyValue};
use crate::learned::LearnedBook;
#[cfg(feature = "inference")]
use crate::pathfinder::{PathfinderConfig, PathfinderGuide};
#[cfg(feature = "inference")]
use crate::puct::{
    search as puct_search,
    search_with_root_output_and_seeds_and_actions as puct_search_with_root_output_and_seeds_and_actions,
    PuctConfig,
};
use crate::search::{lunatic_action, search_best_action, SearchConfig};
use crate::{bit_squares, Action, BoardConfig, GameState, Player};

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
}

#[cfg(feature = "inference")]
impl Default for QAdvPlayConfig {
    fn default() -> Self {
        Self {
            guided: GnnPlayConfig::default(),
            qadv_weight: 1.0,
            tactical_simulations: 0,
            tactical_capture_threshold: 1,
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
        }
    }

    pub fn lunatic(id: impl Into<String>) -> Self {
        Self::Lunatic { id: id.into() }
    }

    pub fn with_book(self, book: Arc<StrategyBook>) -> Self {
        match self {
            Self::Search { id, config, .. } => Self::Search {
                id,
                config,
                book: Some(book),
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
            | Self::Learned { id, .. } => id,
            #[cfg(feature = "inference")]
            Self::Gnn { id, .. } => id,
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
            Self::Search { id, config, book } => {
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
            Self::GnnQAdv { config, model, .. } => {
                choose_qadv_guided(state, random, history, *config, model.as_ref())
            }
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
    if let Some(horizon) = tactical_proof_horizon {
        specification["parameters"] = serde_json::json!({
            "tacticalProofHorizon": horizon,
        });
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
        let repeated = repetitions.entry(RepetitionKey::from(state)).or_default();
        *repeated += 1;
        if *repeated >= 3 {
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
            light.choose(state, &mut random, &repetitions.keys().copied().collect())
        } else {
            dark.choose(state, &mut random, &repetitions.keys().copied().collect())
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
    state.legal_actions().iter().copied().any(|action| {
        let transition = state.apply_legal(action);
        transition.state.winner == Some(state.turn)
            || transition.captured.count_ones() >= u32::from(capture_threshold)
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
        let light = Agent::search("rust-pathfinder-v0.1.0", SearchConfig::default());
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
            90_000
        );
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
}
