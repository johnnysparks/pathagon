//! Reference-compatible, allocation-light Pathagon engine.
//!
//! Squares are row-major `0..49`. Light connects row 6 to row 0; dark
//! connects column 0 to column 6. Both bitboards fit in a single `u64`.

pub mod corpus;
pub mod search;
pub mod selfplay;
pub mod training;

use std::fmt;

pub const BOARD_SIZE: u8 = 7;
pub const CELL_COUNT: u8 = 49;
const FULL_BOARD: u64 = (1_u64 << CELL_COUNT) - 1;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Player {
    Light,
    Dark,
}

impl Player {
    pub const fn other(self) -> Self {
        match self {
            Self::Light => Self::Dark,
            Self::Dark => Self::Light,
        }
    }

    pub const fn index(self) -> usize {
        match self {
            Self::Light => 0,
            Self::Dark => 1,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Light => "light",
            Self::Dark => "dark",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Action {
    Place { to: u8 },
    Relocate { from: u8, to: u8 },
}

impl Action {
    pub const fn destination(self) -> u8 {
        match self {
            Self::Place { to } | Self::Relocate { to, .. } => to,
        }
    }

    pub const fn order(self) -> u16 {
        match self {
            Self::Place { to } => to as u16,
            Self::Relocate { from, to } => from as u16 * CELL_COUNT as u16 + to as u16,
        }
    }
}

impl fmt::Display for Action {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Place { to } => write!(formatter, "P{to}"),
            Self::Relocate { from, to } => write!(formatter, "R{from}>{to}"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GameState {
    pub light: u64,
    pub dark: u64,
    pub reserve: [u8; 2],
    pub turn: Player,
    pub forbidden: u64,
    pub last_relocated_to: [Option<u8>; 2],
    pub last_capture: u8,
    pub last_player: Option<Player>,
    pub winner: Option<Player>,
    pub ply: u16,
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

impl GameState {
    pub const fn new() -> Self {
        Self {
            light: 0,
            dark: 0,
            reserve: [14, 14],
            turn: Player::Light,
            forbidden: 0,
            last_relocated_to: [None, None],
            last_capture: 0,
            last_player: None,
            winner: None,
            ply: 0,
        }
    }

    pub const fn pieces(self, player: Player) -> u64 {
        match player {
            Player::Light => self.light,
            Player::Dark => self.dark,
        }
    }

    pub fn board_at(self, square: u8) -> Option<Player> {
        let bit = bit(square);
        if self.light & bit != 0 {
            Some(Player::Light)
        } else if self.dark & bit != 0 {
            Some(Player::Dark)
        } else {
            None
        }
    }

    pub fn legal_actions(self) -> Vec<Action> {
        if self.winner.is_some() {
            return Vec::new();
        }
        let destinations = FULL_BOARD & !(self.light | self.dark | self.forbidden);
        if self.reserve[self.turn.index()] > 0 {
            return squares(destinations)
                .map(|to| Action::Place { to })
                .collect();
        }

        let mut sources = self.pieces(self.turn);
        if let Some(square) = self.last_relocated_to[self.turn.index()] {
            sources &= !bit(square);
        }
        let destination_squares: Vec<u8> = squares(destinations).collect();
        let mut actions = Vec::with_capacity(sources.count_ones() as usize * destination_squares.len());
        for from in squares(sources) {
            for &to in &destination_squares {
                actions.push(Action::Relocate { from, to });
            }
        }
        actions
    }

    pub fn apply(self, action: Action) -> Result<Transition, &'static str> {
        if !self.legal_actions().contains(&action) {
            return Err("illegal Pathagon action");
        }
        Ok(self.apply_legal(action))
    }

    /// Apply an action already produced by `legal_actions`.
    pub fn apply_legal(mut self, action: Action) -> Transition {
        let player = self.turn;
        let opponent = player.other();
        let destination = action.destination();
        match action {
            Action::Place { .. } => {
                self.reserve[player.index()] -= 1;
                self.last_relocated_to[player.index()] = None;
            }
            Action::Relocate { from, to } => {
                self.clear_piece(player, from);
                self.last_relocated_to[player.index()] = Some(to);
            }
        }
        self.set_piece(player, destination);
        let captured = captures_from(self, destination, player);
        self.clear_mask(opponent, captured);
        self.reserve[opponent.index()] += captured.count_ones() as u8;
        self.forbidden = captured;
        self.last_capture = captured.count_ones() as u8;
        self.last_player = Some(player);
        self.winner = has_winning_path(self, player).then_some(player);
        self.turn = opponent;
        self.ply += 1;
        Transition { state: self, captured }
    }

    fn set_piece(&mut self, player: Player, square: u8) {
        match player {
            Player::Light => self.light |= bit(square),
            Player::Dark => self.dark |= bit(square),
        }
    }

    fn clear_piece(&mut self, player: Player, square: u8) {
        self.clear_mask(player, bit(square));
    }

    fn clear_mask(&mut self, player: Player, mask: u64) {
        match player {
            Player::Light => self.light &= !mask,
            Player::Dark => self.dark &= !mask,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Transition {
    pub state: GameState,
    pub captured: u64,
}

pub fn has_winning_path(state: GameState, player: Player) -> bool {
    let pieces = state.pieces(player);
    let near_edge = if player == Player::Light {
        0x7f_u64 << 42
    } else {
        column_mask(0)
    };
    let far_edge = if player == Player::Light {
        0x7f
    } else {
        column_mask(6)
    };
    let mut frontier = pieces & near_edge;
    let mut visited = frontier;
    while frontier != 0 {
        if frontier & far_edge != 0 {
            return true;
        }
        let mut adjacent = 0;
        for square in squares(frontier) {
            adjacent |= neighbor_mask(square);
        }
        frontier = adjacent & pieces & !visited;
        visited |= frontier;
    }
    false
}

pub fn parse_action(text: &str) -> Result<Action, String> {
    if let Some(destination) = text.strip_prefix('P') {
        let to = parse_square(destination)?;
        return Ok(Action::Place { to });
    }
    if let Some(relocation) = text.strip_prefix('R') {
        let (from, to) = relocation
            .split_once('>')
            .ok_or_else(|| format!("invalid relocation: {text}"))?;
        return Ok(Action::Relocate {
            from: parse_square(from)?,
            to: parse_square(to)?,
        });
    }
    Err(format!("invalid action: {text}"))
}

pub fn bit_squares(mask: u64) -> Vec<u8> {
    squares(mask).collect()
}

pub(crate) const fn bit(square: u8) -> u64 {
    1_u64 << square
}

pub(crate) fn squares(mask: u64) -> Squares {
    Squares(mask)
}

pub(crate) struct Squares(u64);

impl Iterator for Squares {
    type Item = u8;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 == 0 {
            return None;
        }
        let square = self.0.trailing_zeros() as u8;
        self.0 &= self.0 - 1;
        Some(square)
    }
}

pub(crate) fn neighbor_mask(square: u8) -> u64 {
    let row = square / BOARD_SIZE;
    let column = square % BOARD_SIZE;
    let mut mask = 0;
    if row > 0 {
        mask |= bit(square - BOARD_SIZE);
    }
    if row + 1 < BOARD_SIZE {
        mask |= bit(square + BOARD_SIZE);
    }
    if column > 0 {
        mask |= bit(square - 1);
    }
    if column + 1 < BOARD_SIZE {
        mask |= bit(square + 1);
    }
    mask
}

fn captures_from(state: GameState, origin: u8, player: Player) -> u64 {
    let opponent = player.other();
    let row = (origin / BOARD_SIZE) as i8;
    let column = (origin % BOARD_SIZE) as i8;
    let mut captured = 0;
    for (row_delta, column_delta) in [(-1_i8, 0_i8), (1, 0), (0, -1), (0, 1)] {
        let near_row = row + row_delta;
        let near_column = column + column_delta;
        let far_row = row + row_delta * 2;
        let far_column = column + column_delta * 2;
        if !(0..BOARD_SIZE as i8).contains(&far_row) || !(0..BOARD_SIZE as i8).contains(&far_column) {
            continue;
        }
        let near = (near_row * BOARD_SIZE as i8 + near_column) as u8;
        let far = (far_row * BOARD_SIZE as i8 + far_column) as u8;
        if state.board_at(near) == Some(opponent) && state.board_at(far) == Some(player) {
            captured |= bit(near);
        }
    }
    captured
}

const fn column_mask(column: u8) -> u64 {
    let mut mask = 0;
    let mut row = 0;
    while row < BOARD_SIZE {
        mask |= bit(row * BOARD_SIZE + column);
        row += 1;
    }
    mask
}

fn parse_square(text: &str) -> Result<u8, String> {
    let square: u8 = text.parse().map_err(|_| format!("invalid square: {text}"))?;
    if square >= CELL_COUNT {
        return Err(format!("square outside board: {square}"));
    }
    Ok(square)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_board_has_49_moves() {
        assert_eq!(GameState::new().legal_actions().len(), 49);
    }

    #[test]
    fn relocation_changes_the_board_and_cannot_repeat_piece() {
        let mut state = GameState::new();
        state.light = (0..14).fold(0, |mask, square| mask | bit(square));
        state.reserve[Player::Light.index()] = 0;
        let moved = state.apply(Action::Relocate { from: 0, to: 48 }).unwrap().state;
        assert!(!moved.legal_actions().iter().any(|action| matches!(action, Action::Relocate { from: 48, .. })));
        assert!(state.apply(Action::Relocate { from: 0, to: 0 }).is_err());
    }
}
