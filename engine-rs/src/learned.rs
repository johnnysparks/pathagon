//! A deliberately small tabular learner for replay-derived self-play data.
//!
//! The learner memorizes exact legal state/action pairs from a compact replay
//! corpus and chooses the action with the strongest empirical result. It is
//! intentionally conservative: unseen states fall back to the normal search
//! agent, and callers can require repeated evidence before trusting a table
//! entry.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

use crate::corpus::{decode_action, decode_state, encode_action, encode_state, parse_compact_game};
use crate::{Action, GameState};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LearnedChoice {
    pub action: Action,
    pub visits: u32,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
}

impl LearnedChoice {
    pub const fn points_rate_per_mille(self) -> u32 {
        if self.visits == 0 {
            0
        } else {
            (self.wins * 2 + self.draws) * 1_000 / (self.visits * 2)
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct LearnedBook {
    entries: HashMap<BookKey, BookEntry>,
    games: u32,
    moves: u64,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct BookKey {
    state: GameState,
    action: Action,
}

#[derive(Clone, Copy, Debug, Default)]
struct BookEntry {
    visits: u32,
    wins: u32,
    losses: u32,
    draws: u32,
}

impl LearnedBook {
    pub fn from_games_file(path: &Path) -> io::Result<Self> {
        let source = fs::read_to_string(path)?;
        let mut book = Self::default();
        for (line_number, line) in source.lines().enumerate() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let game =
                parse_compact_game(line).map_err(|error| invalid_error(line_number, &error))?;
            book.record_game(&game)
                .map_err(|error| invalid_error(line_number, &error))?;
        }
        Ok(book)
    }

    pub fn load(path: &Path) -> io::Result<Self> {
        let source = fs::read_to_string(path)?;
        let mut book = Self::default();
        for (line_number, line) in source.lines().enumerate() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() != 6 {
                return invalid_data(line_number, "expected 6 tab-separated fields");
            }
            let state =
                decode_state(fields[0]).map_err(|error| invalid_error(line_number, &error))?;
            let action =
                decode_action(fields[1]).map_err(|error| invalid_error(line_number, &error))?;
            let entry = BookEntry {
                visits: parse_field(fields[2], line_number)?,
                wins: parse_field(fields[3], line_number)?,
                losses: parse_field(fields[4], line_number)?,
                draws: parse_field(fields[5], line_number)?,
            };
            book.entries.insert(BookKey { state, action }, entry);
        }
        Ok(book)
    }

    pub fn choose(&self, state: GameState, minimum_visits: u32) -> Option<LearnedChoice> {
        let legal = state.legal_actions();
        self.entries
            .iter()
            .filter(|(key, entry)| {
                key.state == state && entry.visits >= minimum_visits && legal.contains(&key.action)
            })
            .max_by(|left, right| {
                let left_points = left.1.wins as u64 * 2 + left.1.draws as u64;
                let right_points = right.1.wins as u64 * 2 + right.1.draws as u64;
                left_points
                    .saturating_mul(right.1.visits as u64)
                    .cmp(&right_points.saturating_mul(left.1.visits as u64))
                    .then_with(|| left.1.visits.cmp(&right.1.visits))
                    .then_with(|| left.1.wins.cmp(&right.1.wins))
                    .then_with(|| right.0.action.order().cmp(&left.0.action.order()))
            })
            .map(|(key, entry)| LearnedChoice {
                action: key.action,
                visits: entry.visits,
                wins: entry.wins,
                losses: entry.losses,
                draws: entry.draws,
            })
    }

    pub fn games(&self) -> u32 {
        self.games
    }

    pub fn moves(&self) -> u64 {
        self.moves
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn write(&self, path: &Path) -> io::Result<()> {
        let mut lines = self
            .entries
            .iter()
            .map(|(key, entry)| {
                format!(
                    "{}\t{}\t{}\t{}\t{}\t{}",
                    encode_state(key.state),
                    encode_action(key.action),
                    entry.visits,
                    entry.wins,
                    entry.losses,
                    entry.draws,
                )
            })
            .collect::<Vec<_>>();
        lines.sort();
        let mut output = String::from("# state\taction\tvisits\twins\tlosses\tdraws\n");
        output.push_str(&lines.join("\n"));
        output.push('\n');
        fs::write(path, output)
    }

    fn record_game(&mut self, game: &crate::corpus::CompactGame) -> Result<(), String> {
        let mut state = GameState::new();
        for &action in &game.actions {
            let entry = self.entries.entry(BookKey { state, action }).or_default();
            entry.visits = entry.visits.saturating_add(1);
            match game.winner {
                Some(winner) if winner == state.turn => entry.wins = entry.wins.saturating_add(1),
                Some(_) => entry.losses = entry.losses.saturating_add(1),
                None => entry.draws = entry.draws.saturating_add(1),
            }
            state = state.apply(action)?.state;
            self.moves = self.moves.saturating_add(1);
        }
        self.games = self.games.saturating_add(1);
        Ok(())
    }
}

fn parse_field<T: std::str::FromStr>(text: &str, line_number: usize) -> io::Result<T> {
    text.parse().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("line {}: invalid numeric field", line_number + 1),
        )
    })
}

fn invalid_data<T>(line_number: usize, message: &str) -> io::Result<T> {
    Err(invalid_error(line_number, message))
}

fn invalid_error(line_number: usize, message: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("line {}: {message}", line_number + 1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::compact_game_line;
    use crate::selfplay::{Agent, MatchOptions, play_game};

    #[test]
    fn learned_book_round_trips_a_replay_corpus() {
        let light = Agent::random("light-random");
        let dark = Agent::random("dark-random");
        let record = play_game(
            &light,
            &dark,
            MatchOptions {
                seed: 9,
                max_plies: 40,
                opening_random_plies: 0,
                ..MatchOptions::default()
            },
        );
        let line = compact_game_line(&record);
        let game = parse_compact_game(&line).unwrap();
        let mut book = LearnedBook::default();
        book.record_game(&game).unwrap();
        assert_eq!(book.games(), 1);
        assert_eq!(book.moves(), record.moves.len() as u64);
        assert_eq!(book.len(), record.moves.len());
    }
}
