use std::collections::HashMap;
use std::sync::Arc;

use crate::corpus::StrategyBook;
use crate::learned::LearnedBook;
use crate::search::{lunatic_action, search_best_action, SearchConfig};
use crate::{bit_squares, Action, BoardConfig, GameState, Player};

#[derive(Clone, Debug)]
pub enum Agent {
    Random { id: String },
    Lunatic { id: String },
    Search { id: String, config: SearchConfig, book: Option<Arc<StrategyBook>> },
    Learned { id: String, config: SearchConfig, book: Arc<LearnedBook>, minimum_visits: u32 },
}

impl Agent {
    pub fn random(id: impl Into<String>) -> Self {
        Self::Random { id: id.into() }
    }

    pub fn search(id: impl Into<String>, config: SearchConfig) -> Self {
        Self::Search { id: id.into(), config, book: None }
    }

    pub fn lunatic(id: impl Into<String>) -> Self {
        Self::Lunatic { id: id.into() }
    }

    pub fn with_book(self, book: Arc<StrategyBook>) -> Self {
        match self {
            Self::Search { id, config, .. } => Self::Search { id, config, book: Some(book) },
            random => random,
        }
    }

    pub fn learned(id: impl Into<String>, config: SearchConfig, book: Arc<LearnedBook>, minimum_visits: u32) -> Self {
        Self::Learned { id: id.into(), config, book, minimum_visits }
    }

    pub fn id(&self) -> &str {
        match self {
            Self::Random { id } | Self::Lunatic { id } | Self::Search { id, .. } | Self::Learned { id, .. } => id,
        }
    }

    fn choose(&self, state: GameState, random: &mut Mulberry32) -> Decision {
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
                }
            }
            Self::Search { id, config, book } => {
                if let Some(choice) = book.as_ref().and_then(|book| book.choose(id, state, config.depth)) {
                    return Decision {
                        action: Some(choice.action),
                        score: choice.score,
                        nodes: 0,
                        completed_depth: choice.completed_depth,
                        table_hits: 0,
                        book_hit: true,
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
                }
            }
            Self::Learned { config, book, minimum_visits, .. } => {
                if let Some(choice) = book.choose(state, *minimum_visits) {
                    return Decision {
                        action: Some(choice.action),
                        score: choice.points_rate_per_mille() as i32,
                        nodes: 0,
                        completed_depth: 0,
                        table_hits: 0,
                        book_hit: true,
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
                }
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
        Self { seed: 20_260_823, max_plies: 180, opening_random_plies: 2, board_size: 7, reserve_per_player: 14 }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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

#[derive(Clone, Debug, Eq, PartialEq)]
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
        let winner = self.winner.map_or("null".to_owned(), |player| format!("\"{}\"", player.as_str()));
        let moves = self.moves.iter().map(|record| {
            let action = match record.action {
                Action::Place { to } => format!("{{\"kind\":\"place\",\"to\":{to}}}"),
                Action::Relocate { from, to } => format!("{{\"kind\":\"relocate\",\"from\":{from},\"to\":{to}}}"),
            };
            let captured = bit_squares(record.captured).iter().map(u8::to_string).collect::<Vec<_>>().join(",");
            format!(
                "{{\"ply\":{},\"player\":\"{}\",\"action\":{},\"captured\":[{}],\"score\":{},\"nodes\":{},\"completedDepth\":{},\"tableHits\":{},\"bookHit\":{}}}",
                record.ply,
                record.player.as_str(),
                action,
                captured,
                record.score,
                record.nodes,
                record.completed_depth,
                record.table_hits,
                record.book_hit,
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
    let (kind, name, depth, node_budget, beam, weights) = match agent {
        Agent::Random { .. } => ("random", "Coin Flip", 0, 0, 0, crate::search::EvaluationWeights::default()),
        Agent::Lunatic { .. } => ("heuristic", "Lunatic", 1, 0, 0, crate::search::EvaluationWeights::default()),
        Agent::Search { config, .. } => ("search", "Rust Search", u32::from(config.depth), config.max_nodes, config.beam_width as u32, config.weights),
        Agent::Learned { config, .. } => ("learned", "Learned", u32::from(config.depth), config.max_nodes, config.beam_width as u32, config.weights),
    };
    format!(
        "{{\"id\":\"{}\",\"name\":\"{}\",\"version\":\"1.0.0\",\"kind\":\"{}\",\"engineId\":\"rust-bitboard\",\"manifest\":{{\"manifestVersion\":1,\"runtime\":\"rust\",\"rulesVersion\":\"pathagon-rules-v1\",\"evaluatorWeights\":{{\"path\":{},\"material\":{},\"capture\":{},\"structure\":{},\"threat\":{},\"edge\":{}}},\"depth\":{},\"nodeBudget\":{},\"beam\":{},\"modelHash\":null}}}}",
        json_escape(agent.id()), name, kind, weights.path, weights.material, weights.capture, weights.structure, weights.threat, weights.edge, depth, node_budget, beam,
    )
}

pub fn play_game(light: &Agent, dark: &Agent, options: MatchOptions) -> GameRecord {
    let mut random = Mulberry32::new(options.seed);
    let config = BoardConfig::new(options.board_size, options.reserve_per_player).expect("valid match configuration");
    let mut state = GameState::with_config(config);
    let mut moves = Vec::new();
    let mut repetitions = HashMap::<RepetitionKey, u8>::new();

    while state.winner.is_none() && state.ply < options.max_plies {
        let repeated = repetitions.entry(RepetitionKey::from(state)).or_default();
        *repeated += 1;
        if *repeated >= 3 {
            return record(light, dark, options, None, TerminationReason::ThreefoldRepetition, moves);
        }
        let actions = state.legal_actions();
        if actions.is_empty() {
            return record(light, dark, options, None, TerminationReason::NoLegalAction, moves);
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
            }
        } else if player == Player::Light {
            light.choose(state, &mut random)
        } else {
            dark.choose(state, &mut random)
        };
        let Some(action) = decision.action else {
            return record(light, dark, options, None, TerminationReason::NoLegalAction, moves);
        };
        if !actions.contains(&action) {
            return record(light, dark, options, None, TerminationReason::NoLegalAction, moves);
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
        });
    }
    if let Some(winner) = state.winner {
        record(light, dark, options, Some(winner), TerminationReason::Path, moves)
    } else {
        record(light, dark, options, None, TerminationReason::MaxPlies, moves)
    }
}

#[derive(Clone, Copy, Debug)]
struct Decision {
    action: Option<Action>,
    score: i32,
    nodes: u64,
    completed_depth: u8,
    table_hits: u64,
    book_hit: bool,
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
        let options = MatchOptions { seed: 42, max_plies: 60, opening_random_plies: 2, ..MatchOptions::default() };
        assert_eq!(play_game(&light, &dark, options), play_game(&light, &dark, options));
    }

    #[test]
    fn lunatic_games_are_reproducible_and_nonempty() {
        let light = Agent::lunatic("lunatic-v0.1.0");
        let dark = Agent::random("dark-random");
        let options = MatchOptions { seed: 4242, max_plies: 60, opening_random_plies: 2, ..MatchOptions::default() };
        let record = play_game(&light, &dark, options);
        assert!(!record.moves.is_empty());
        assert_eq!(record, play_game(&light, &dark, options));
        assert_eq!(record.moves.first().unwrap().ply, 1);
    }

    #[test]
    fn generated_records_include_manifest_backed_agent_identity() {
        let light = Agent::search("rust-pathfinder-v0.1.0", SearchConfig::default());
        let dark = Agent::random("coin-flip-seeded");
        let record = play_game(&light, &dark, MatchOptions { seed: 7, max_plies: 8, opening_random_plies: 0, board_size: 4, reserve_per_player: 8 });
        let replay = crate::contract::ReplayRecord::from_json(&record.to_json()).expect("generated replay follows contract");
        assert_eq!(replay.agent_specifications.light.manifest.runtime, "rust");
        assert_eq!(replay.agent_specifications.light.manifest.node_budget, 90_000);
    }

    #[test]
    fn mulberry_matches_javascript_reference_values() {
        let mut random = Mulberry32::new(42);
        assert_eq!(random.next_u32(), 2_581_720_956);
        assert_eq!(random.next_u32(), 1_925_393_290);
        assert_eq!(random.next_u32(), 3_661_312_704);
    }
}
