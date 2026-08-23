use std::collections::HashMap;

use crate::search::{search_best_action, SearchConfig};
use crate::{bit_squares, Action, GameState, Player};

#[derive(Clone, Debug)]
pub enum Agent {
    Random { id: String },
    Search { id: String, config: SearchConfig },
}

impl Agent {
    pub fn random(id: impl Into<String>) -> Self {
        Self::Random { id: id.into() }
    }

    pub fn search(id: impl Into<String>, config: SearchConfig) -> Self {
        Self::Search { id: id.into(), config }
    }

    pub fn id(&self) -> &str {
        match self {
            Self::Random { id } | Self::Search { id, .. } => id,
        }
    }

    fn choose(&self, state: GameState, random: &mut Mulberry32) -> Decision {
        match self {
            Self::Random { .. } => {
                let actions = state.legal_actions();
                Decision {
                    action: random.choose(&actions),
                    nodes: u64::from(!actions.is_empty()),
                    completed_depth: 0,
                    table_hits: 0,
                }
            }
            Self::Search { config, .. } => {
                let result = search_best_action(state, *config);
                Decision {
                    action: result.action,
                    nodes: result.nodes,
                    completed_depth: result.completed_depth,
                    table_hits: result.table_hits,
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
}

impl Default for MatchOptions {
    fn default() -> Self {
        Self { seed: 20_260_823, max_plies: 180, opening_random_plies: 2 }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MoveRecord {
    pub ply: u16,
    pub player: Player,
    pub action: Action,
    pub captured: u64,
    pub nodes: u64,
    pub completed_depth: u8,
    pub table_hits: u64,
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
    pub light_agent: String,
    pub dark_agent: String,
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
                "{{\"ply\":{},\"player\":\"{}\",\"action\":{},\"captured\":[{}],\"nodes\":{},\"completedDepth\":{},\"tableHits\":{}}}",
                record.ply,
                record.player.as_str(),
                action,
                captured,
                record.nodes,
                record.completed_depth,
                record.table_hits,
            )
        }).collect::<Vec<_>>().join(",");
        format!(
            "{{\"schemaVersion\":2,\"seed\":{},\"agents\":{{\"light\":\"{}\",\"dark\":\"{}\"}},\"winner\":{},\"result\":\"{}\",\"reason\":\"{}\",\"plies\":{},\"moves\":[{}]}}",
            self.seed,
            json_escape(&self.light_agent),
            json_escape(&self.dark_agent),
            winner,
            if self.winner.is_some() { "win" } else { "draw" },
            self.reason.as_str(),
            self.moves.len(),
            moves,
        )
    }
}

pub fn play_game(light: &Agent, dark: &Agent, options: MatchOptions) -> GameRecord {
    let mut random = Mulberry32::new(options.seed);
    let mut state = GameState::new();
    let mut moves = Vec::new();
    let mut repetitions = HashMap::<RepetitionKey, u8>::new();

    while state.winner.is_none() && state.ply < options.max_plies {
        let repeated = repetitions.entry(RepetitionKey::from(state)).or_default();
        *repeated += 1;
        if *repeated >= 3 {
            return record(light, dark, options.seed, None, TerminationReason::ThreefoldRepetition, moves);
        }
        let actions = state.legal_actions();
        if actions.is_empty() {
            return record(light, dark, options.seed, None, TerminationReason::NoLegalAction, moves);
        }
        let player = state.turn;
        let decision = if state.ply < options.opening_random_plies {
            Decision { action: random.choose(&actions), nodes: 1, completed_depth: 0, table_hits: 0 }
        } else if player == Player::Light {
            light.choose(state, &mut random)
        } else {
            dark.choose(state, &mut random)
        };
        let Some(action) = decision.action else {
            return record(light, dark, options.seed, None, TerminationReason::NoLegalAction, moves);
        };
        if !actions.contains(&action) {
            return record(light, dark, options.seed, None, TerminationReason::NoLegalAction, moves);
        }
        let transition = state.apply_legal(action);
        state = transition.state;
        moves.push(MoveRecord {
            ply: state.ply,
            player,
            action,
            captured: transition.captured,
            nodes: decision.nodes,
            completed_depth: decision.completed_depth,
            table_hits: decision.table_hits,
        });
    }
    if let Some(winner) = state.winner {
        record(light, dark, options.seed, Some(winner), TerminationReason::Path, moves)
    } else {
        record(light, dark, options.seed, None, TerminationReason::MaxPlies, moves)
    }
}

#[derive(Clone, Copy, Debug)]
struct Decision {
    action: Option<Action>,
    nodes: u64,
    completed_depth: u8,
    table_hits: u64,
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
    seed: u32,
    winner: Option<Player>,
    reason: TerminationReason,
    moves: Vec<MoveRecord>,
) -> GameRecord {
    GameRecord {
        seed,
        light_agent: light.id().to_owned(),
        dark_agent: dark.id().to_owned(),
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
        let options = MatchOptions { seed: 42, max_plies: 60, opening_random_plies: 2 };
        assert_eq!(play_game(&light, &dark, options), play_game(&light, &dark, options));
    }

    #[test]
    fn mulberry_matches_javascript_reference_values() {
        let mut random = Mulberry32::new(42);
        assert_eq!(random.next_u32(), 2_581_720_956);
        assert_eq!(random.next_u32(), 1_925_393_290);
        assert_eq!(random.next_u32(), 3_661_312_704);
    }
}
