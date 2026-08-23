//! Compact, replayable game histories and a persistent position/action book.
//!
//! Games need only the seed, agents, outcome, and two base64url characters per
//! action. Captures and every intermediate board are derived by replaying the
//! rules engine. The position book is deliberately separate: it caches the
//! expensive search answer and observed outcome for exact positions.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::Path;

use crate::selfplay::{GameRecord, TerminationReason};
use crate::{Action, GameState, Player, CELL_COUNT};

const ALPHABET: &[u8; 64] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompactGame {
    pub seed: u32,
    pub light_agent: String,
    pub dark_agent: String,
    pub winner: Option<Player>,
    pub reason: TerminationReason,
    pub actions: Vec<Action>,
}

impl CompactGame {
    pub fn replay(&self) -> Result<GameState, String> {
        let mut state = GameState::new();
        for action in &self.actions {
            state = state.apply(*action)?.state;
        }
        if state.winner != self.winner {
            return Err("stored outcome does not match replayed outcome".to_owned());
        }
        Ok(state)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BookChoice {
    pub action: Action,
    pub score: i32,
    pub completed_depth: u8,
    pub prior_nodes: u64,
    pub visits: u32,
}

#[derive(Clone, Debug, Default)]
pub struct StrategyBook {
    entries: HashMap<BookKey, BookEntry>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct BookKey {
    state: GameState,
    agent: String,
    action: Action,
}

#[derive(Clone, Copy, Debug, Default)]
struct BookEntry {
    score: i32,
    completed_depth: u8,
    prior_nodes: u64,
    visits: u32,
    wins: u32,
    losses: u32,
    draws: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CorpusSummary {
    pub games: usize,
    pub positions: usize,
    pub added_games: usize,
}

impl StrategyBook {
    pub fn load(path: &Path) -> io::Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let source = fs::read_to_string(path)?;
        let mut book = Self::default();
        for (line_number, line) in source.lines().enumerate() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() != 10 {
                return invalid_data(line_number, "expected 10 tab-separated fields");
            }
            let state = decode_state(fields[0]).map_err(|error| {
                io::Error::new(io::ErrorKind::InvalidData, format!("line {}: {error}", line_number + 1))
            })?;
            let action = decode_action(fields[2]).map_err(|error| {
                io::Error::new(io::ErrorKind::InvalidData, format!("line {}: {error}", line_number + 1))
            })?;
            let entry = BookEntry {
                completed_depth: parse_field(fields[3], line_number)?,
                score: parse_field(fields[4], line_number)?,
                prior_nodes: parse_field(fields[5], line_number)?,
                visits: parse_field(fields[6], line_number)?,
                wins: parse_field(fields[7], line_number)?,
                losses: parse_field(fields[8], line_number)?,
                draws: parse_field(fields[9], line_number)?,
            };
            book.entries.insert(BookKey { state, agent: fields[1].to_owned(), action }, entry);
        }
        Ok(book)
    }

    pub fn choose(&self, agent: &str, state: GameState, minimum_depth: u8) -> Option<BookChoice> {
        let legal = state.legal_actions();
        self.entries
            .iter()
            .filter(|(key, entry)| {
                key.agent == agent
                    && key.state == state
                    && entry.completed_depth >= minimum_depth
                    && legal.contains(&key.action)
            })
            .map(|(key, entry)| (key.action, *entry))
            .max_by(|left, right| {
                let left_net = left.1.wins as i64 - left.1.losses as i64;
                let right_net = right.1.wins as i64 - right.1.losses as i64;
                left.1.completed_depth.cmp(&right.1.completed_depth)
                    .then_with(|| (left_net * right.1.visits as i64).cmp(&(right_net * left.1.visits as i64)))
                    .then_with(|| left.1.visits.cmp(&right.1.visits))
                    .then_with(|| right.0.order().cmp(&left.0.order()))
            })
            .map(|(action, entry)| BookChoice {
                action,
                score: entry.score,
                completed_depth: entry.completed_depth,
                prior_nodes: entry.prior_nodes,
                visits: entry.visits,
            })
    }

    pub fn record_game(&mut self, game: &GameRecord) -> Result<(), String> {
        let mut state = GameState::new();
        for movement in &game.moves {
            if movement.completed_depth > 0 {
                let agent = if movement.player == Player::Light {
                    &game.light_agent
                } else {
                    &game.dark_agent
                };
                let entry = self.entries.entry(BookKey {
                    state,
                    agent: agent.clone(),
                    action: movement.action,
                }).or_default();
                entry.visits = entry.visits.saturating_add(1);
                match game.winner {
                    Some(winner) if winner == movement.player => entry.wins = entry.wins.saturating_add(1),
                    Some(_) => entry.losses = entry.losses.saturating_add(1),
                    None => entry.draws = entry.draws.saturating_add(1),
                }
                if movement.completed_depth > entry.completed_depth
                    || (movement.completed_depth == entry.completed_depth && movement.nodes > entry.prior_nodes)
                {
                    entry.completed_depth = movement.completed_depth;
                    entry.score = movement.score;
                    entry.prior_nodes = movement.nodes;
                }
            }
            state = state.apply(movement.action)?.state;
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    fn write(&self, path: &Path) -> io::Result<()> {
        let mut lines = self.entries.iter().map(|(key, entry)| {
            format!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                encode_state(key.state),
                safe_field(&key.agent),
                encode_action(key.action),
                entry.completed_depth,
                entry.score,
                entry.prior_nodes,
                entry.visits,
                entry.wins,
                entry.losses,
                entry.draws,
            )
        }).collect::<Vec<_>>();
        lines.sort();
        let mut output = String::from("# state\tagent\taction\tdepth\tscore\tprior_nodes\tvisits\twins\tlosses\tdraws\n");
        output.push_str(&lines.join("\n"));
        output.push('\n');
        fs::write(path, output)
    }
}

pub fn write_corpus(directory: &Path, records: &[GameRecord]) -> io::Result<CorpusSummary> {
    fs::create_dir_all(directory)?;
    let games_path = directory.join("games.tsv");
    let positions_path = directory.join("positions.tsv");
    let mut lines = HashSet::<String>::new();
    if games_path.exists() {
        for line in fs::read_to_string(&games_path)?.lines() {
            if !line.is_empty() && !line.starts_with('#') {
                lines.insert(line.to_owned());
            }
        }
    }
    let before = lines.len();
    for record in records {
        lines.insert(compact_game_line(record));
    }
    let mut sorted = lines.into_iter().collect::<Vec<_>>();
    sorted.sort();
    let mut game_output = String::from("# p1\tseed64\tlight\tdark\twinner\treason\t2-char-actions\n");
    game_output.push_str(&sorted.join("\n"));
    game_output.push('\n');
    fs::write(&games_path, game_output)?;

    let mut book = StrategyBook::load(&positions_path)?;
    for record in records {
        book.record_game(record).map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    }
    book.write(&positions_path)?;
    let manifest = format!(
        "{{\"schemaVersion\":1,\"actionEncoding\":\"base64url-12bit\",\"games\":{},\"positions\":{}}}\n",
        sorted.len(),
        book.len(),
    );
    fs::write(directory.join("manifest.json"), manifest)?;
    Ok(CorpusSummary { games: sorted.len(), positions: book.len(), added_games: sorted.len() - before })
}

pub fn compact_game_line(record: &GameRecord) -> String {
    let actions = record.moves.iter().map(|movement| encode_action(movement.action)).collect::<String>();
    format!(
        "p1\t{}\t{}\t{}\t{}\t{}\t{}",
        encode_radix(record.seed as u64),
        safe_field(&record.light_agent),
        safe_field(&record.dark_agent),
        player_code(record.winner),
        reason_code(record.reason),
        actions,
    )
}

pub fn parse_compact_game(line: &str) -> Result<CompactGame, String> {
    let fields: Vec<&str> = line.split('\t').collect();
    if fields.len() != 7 || fields[0] != "p1" {
        return Err("invalid compact game header".to_owned());
    }
    let seed = decode_radix(fields[1])?.try_into().map_err(|_| "seed exceeds u32".to_owned())?;
    let winner = match fields[4] {
        "L" => Some(Player::Light),
        "D" => Some(Player::Dark),
        "-" => None,
        _ => return Err("invalid winner code".to_owned()),
    };
    let reason = match fields[5] {
        "P" => TerminationReason::Path,
        "R" => TerminationReason::ThreefoldRepetition,
        "M" => TerminationReason::MaxPlies,
        "N" => TerminationReason::NoLegalAction,
        _ => return Err("invalid termination code".to_owned()),
    };
    if fields[6].len() % 2 != 0 {
        return Err("action stream must have an even byte count".to_owned());
    }
    let actions = fields[6].as_bytes().chunks_exact(2).map(|pair| {
        let token = std::str::from_utf8(pair).map_err(|_| "invalid action bytes".to_owned())?;
        decode_action(token)
    }).collect::<Result<Vec<_>, _>>()?;
    Ok(CompactGame {
        seed,
        light_agent: fields[2].to_owned(),
        dark_agent: fields[3].to_owned(),
        winner,
        reason,
        actions,
    })
}

pub fn encode_action(action: Action) -> String {
    let code = match action {
        Action::Place { to } => to as u16,
        Action::Relocate { from, to } => CELL_COUNT as u16 + from as u16 * CELL_COUNT as u16 + to as u16,
    };
    let bytes = [ALPHABET[(code >> 6) as usize], ALPHABET[(code & 63) as usize]];
    String::from_utf8(bytes.to_vec()).expect("base64url alphabet is UTF-8")
}

pub fn decode_action(token: &str) -> Result<Action, String> {
    let bytes = token.as_bytes();
    if bytes.len() != 2 {
        return Err("action token must be exactly two bytes".to_owned());
    }
    let code = (alphabet_index(bytes[0])? as u16) << 6 | alphabet_index(bytes[1])? as u16;
    if code < CELL_COUNT as u16 {
        return Ok(Action::Place { to: code as u8 });
    }
    let relocation = code - CELL_COUNT as u16;
    let from = relocation / CELL_COUNT as u16;
    let to = relocation % CELL_COUNT as u16;
    if from >= CELL_COUNT as u16 {
        return Err("relocation token is outside the board".to_owned());
    }
    Ok(Action::Relocate { from: from as u8, to: to as u8 })
}

pub fn encode_state(state: GameState) -> String {
    format!(
        "{}.{}.{}.{}.{}.{}.{}.{}.{}.{}.{}",
        encode_radix(state.light),
        encode_radix(state.dark),
        encode_radix(state.reserve[0] as u64),
        encode_radix(state.reserve[1] as u64),
        player_code(Some(state.turn)),
        encode_radix(state.forbidden),
        option_square(state.last_relocated_to[0]),
        option_square(state.last_relocated_to[1]),
        encode_radix(state.last_capture as u64),
        player_code(state.last_player),
        encode_radix(state.ply as u64),
    )
}

pub fn decode_state(text: &str) -> Result<GameState, String> {
    let fields: Vec<&str> = text.split('.').collect();
    if fields.len() != 11 {
        return Err("invalid state key".to_owned());
    }
    Ok(GameState {
        light: decode_radix(fields[0])?,
        dark: decode_radix(fields[1])?,
        reserve: [small_radix(fields[2])?, small_radix(fields[3])?],
        turn: required_player(fields[4])?,
        forbidden: decode_radix(fields[5])?,
        last_relocated_to: [optional_square(fields[6])?, optional_square(fields[7])?],
        last_capture: small_radix(fields[8])?,
        last_player: optional_player(fields[9])?,
        winner: None,
        ply: decode_radix(fields[10])?.try_into().map_err(|_| "ply exceeds u16".to_owned())?,
    })
}

fn encode_radix(value: u64) -> String {
    if value == 0 {
        return "0".to_owned();
    }
    let mut value = value;
    let mut reversed = Vec::new();
    while value > 0 {
        reversed.push(ALPHABET[(value & 63) as usize]);
        value >>= 6;
    }
    reversed.reverse();
    String::from_utf8(reversed).expect("base64url alphabet is UTF-8")
}

fn decode_radix(text: &str) -> Result<u64, String> {
    if text.is_empty() {
        return Err("empty radix value".to_owned());
    }
    text.bytes().try_fold(0_u64, |value, byte| {
        value.checked_mul(64)
            .and_then(|value| value.checked_add(alphabet_index(byte).ok()? as u64))
            .ok_or_else(|| "radix value overflow".to_owned())
    })
}

fn alphabet_index(byte: u8) -> Result<u8, String> {
    ALPHABET.iter().position(|candidate| *candidate == byte)
        .map(|index| index as u8)
        .ok_or_else(|| format!("invalid base64url byte: {byte}"))
}

fn safe_field(text: &str) -> String {
    text.replace(['\t', '\n', '\r'], "_")
}

fn player_code(player: Option<Player>) -> &'static str {
    match player {
        Some(Player::Light) => "L",
        Some(Player::Dark) => "D",
        None => "-",
    }
}

fn reason_code(reason: TerminationReason) -> &'static str {
    match reason {
        TerminationReason::Path => "P",
        TerminationReason::ThreefoldRepetition => "R",
        TerminationReason::MaxPlies => "M",
        TerminationReason::NoLegalAction => "N",
    }
}

fn option_square(square: Option<u8>) -> String {
    square.map_or_else(|| "-".to_owned(), |value| encode_radix(value as u64))
}

fn optional_square(text: &str) -> Result<Option<u8>, String> {
    if text == "-" { Ok(None) } else { Ok(Some(small_radix(text)?)) }
}

fn required_player(text: &str) -> Result<Player, String> {
    optional_player(text)?.ok_or_else(|| "missing required player".to_owned())
}

fn optional_player(text: &str) -> Result<Option<Player>, String> {
    match text {
        "L" => Ok(Some(Player::Light)),
        "D" => Ok(Some(Player::Dark)),
        "-" => Ok(None),
        _ => Err("invalid player code".to_owned()),
    }
}

fn small_radix(text: &str) -> Result<u8, String> {
    decode_radix(text)?.try_into().map_err(|_| "small radix value exceeds u8".to_owned())
}

fn parse_field<T: std::str::FromStr>(text: &str, line_number: usize) -> io::Result<T> {
    text.parse().map_err(|_| io::Error::new(
        io::ErrorKind::InvalidData,
        format!("line {}: invalid numeric field", line_number + 1),
    ))
}

fn invalid_data<T>(line_number: usize, message: &str) -> io::Result<T> {
    Err(io::Error::new(io::ErrorKind::InvalidData, format!("line {}: {message}", line_number + 1)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::selfplay::{play_game, Agent, MatchOptions};

    #[test]
    fn every_action_round_trips_through_two_bytes() {
        for to in 0..CELL_COUNT {
            let action = Action::Place { to };
            assert_eq!(decode_action(&encode_action(action)).unwrap(), action);
        }
        for from in 0..CELL_COUNT {
            for to in 0..CELL_COUNT {
                let action = Action::Relocate { from, to };
                assert_eq!(decode_action(&encode_action(action)).unwrap(), action);
            }
        }
    }

    #[test]
    fn compact_random_game_replays_exactly() {
        let light = Agent::random("light-random");
        let dark = Agent::random("dark-random");
        let record = play_game(
            &light,
            &dark,
            MatchOptions { seed: 17, max_plies: 80, opening_random_plies: 2 },
        );
        let compact = parse_compact_game(&compact_game_line(&record)).unwrap();
        assert_eq!(compact.actions, record.moves.iter().map(|movement| movement.action).collect::<Vec<_>>());
        compact.replay().unwrap();
    }
}
