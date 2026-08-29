//! Reference-compatible, allocation-light Pathagon engine.
//!
//! Squares are row-major. Light connects the far row to the near row; dark
//! connects the left column to the right column. Boards from 3x3 through 8x8
//! fit in the bitboard representation and share the same rule implementation.

use std::collections::VecDeque;

pub mod contract;
pub mod corpus;
pub mod endgame;
#[cfg(feature = "inference")]
pub mod inference;
pub mod learned;
pub mod model;
pub mod pathfinder;
#[cfg(feature = "inference")]
pub mod puct;
pub mod qadv;
pub mod runtime;
pub mod search;
pub mod selfplay;
pub mod training;
#[cfg(feature = "wasm")]
pub mod wasm_api;

use std::fmt;

pub const MIN_BOARD_SIZE: u8 = 3;
pub const MAX_BOARD_SIZE: u8 = 8;
pub const BOARD_SIZE: u8 = 7;
pub const CELL_COUNT: u8 = BOARD_SIZE * BOARD_SIZE;
pub const MAX_CELL_COUNT: u8 = MAX_BOARD_SIZE * MAX_BOARD_SIZE;
pub const DEFAULT_MAX_PLIES: u16 = 180;

#[derive(Clone, Copy)]
struct CaptureRay {
    near: u64,
    far: u64,
}

const EMPTY_CAPTURE_RAY: CaptureRay = CaptureRay { near: 0, far: 0 };

const fn build_neighbor_masks() -> [[u64; MAX_CELL_COUNT as usize]; (MAX_BOARD_SIZE + 1) as usize] {
    let mut table = [[0_u64; MAX_CELL_COUNT as usize]; (MAX_BOARD_SIZE + 1) as usize];
    let mut board_size = 1_u8;
    while board_size <= MAX_BOARD_SIZE {
        let mut square = 0_u8;
        while square < MAX_CELL_COUNT {
            if square < board_size * board_size {
                let row = square / board_size;
                let column = square % board_size;
                let mut mask = 0_u64;
                if row > 0 {
                    mask |= 1_u64 << (square - board_size);
                }
                if row + 1 < board_size {
                    mask |= 1_u64 << (square + board_size);
                }
                if column > 0 {
                    mask |= 1_u64 << (square - 1);
                }
                if column + 1 < board_size {
                    mask |= 1_u64 << (square + 1);
                }
                table[board_size as usize][square as usize] = mask;
            }
            square += 1;
        }
        board_size += 1;
    }
    table
}

const fn build_capture_rays(
) -> [[[CaptureRay; 4]; MAX_CELL_COUNT as usize]; (MAX_BOARD_SIZE + 1) as usize] {
    let mut table =
        [[[EMPTY_CAPTURE_RAY; 4]; MAX_CELL_COUNT as usize]; (MAX_BOARD_SIZE + 1) as usize];
    let mut board_size = 1_u8;
    while board_size <= MAX_BOARD_SIZE {
        let mut square = 0_u8;
        while square < board_size * board_size {
            let row = square / board_size;
            let column = square % board_size;
            let mut direction = 0_usize;
            while direction < 4 {
                let (row_delta, column_delta) = match direction {
                    0 => (-1_i8, 0_i8),
                    1 => (1_i8, 0_i8),
                    2 => (0_i8, -1_i8),
                    _ => (0_i8, 1_i8),
                };
                let near_row = row as i8 + row_delta;
                let near_column = column as i8 + column_delta;
                let far_row = row as i8 + row_delta * 2;
                let far_column = column as i8 + column_delta * 2;
                if near_row >= 0
                    && near_row < board_size as i8
                    && near_column >= 0
                    && near_column < board_size as i8
                    && far_row >= 0
                    && far_row < board_size as i8
                    && far_column >= 0
                    && far_column < board_size as i8
                {
                    let near = (near_row as u8 * board_size + near_column as u8) as u8;
                    let far = (far_row as u8 * board_size + far_column as u8) as u8;
                    table[board_size as usize][square as usize][direction] = CaptureRay {
                        near: 1_u64 << near,
                        far: 1_u64 << far,
                    };
                }
                direction += 1;
            }
            square += 1;
        }
        board_size += 1;
    }
    table
}

static NEIGHBOR_MASKS: [[u64; MAX_CELL_COUNT as usize]; (MAX_BOARD_SIZE + 1) as usize] =
    build_neighbor_masks();
static CAPTURE_RAYS: [[[CaptureRay; 4]; MAX_CELL_COUNT as usize]; (MAX_BOARD_SIZE + 1) as usize] =
    build_capture_rays();

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BoardConfig {
    pub board_size: u8,
    pub reserve_per_player: u8,
    pub max_plies: u16,
}

impl BoardConfig {
    pub const DEFAULT: Self = Self {
        board_size: BOARD_SIZE,
        reserve_per_player: 14,
        max_plies: DEFAULT_MAX_PLIES,
    };

    pub fn new(board_size: u8, reserve_per_player: u8) -> Result<Self, String> {
        if !(MIN_BOARD_SIZE..=MAX_BOARD_SIZE).contains(&board_size) {
            return Err(format!(
                "board size outside {MIN_BOARD_SIZE}..{MAX_BOARD_SIZE}"
            ));
        }
        if reserve_per_player == 0 || reserve_per_player > 64 {
            return Err("reserve outside 1..64".to_owned());
        }
        Ok(Self {
            board_size,
            reserve_per_player,
            max_plies: DEFAULT_MAX_PLIES,
        })
    }

    pub fn with_max_plies(mut self, max_plies: u16) -> Result<Self, String> {
        if max_plies == 0 || max_plies > 4096 {
            return Err("maximum plies outside 1..4096".to_owned());
        }
        self.max_plies = max_plies;
        Ok(self)
    }

    pub fn from_contract(config: &crate::contract::GameConfig) -> Result<Self, String> {
        config.validate()?;
        Self::new(config.board_size, config.reserve_per_player)?.with_max_plies(config.max_plies)
    }

    pub const fn cells(self) -> u8 {
        self.board_size * self.board_size
    }

    pub const fn full_board(self) -> u64 {
        if self.cells() == 64 {
            u64::MAX
        } else {
            (1_u64 << self.cells()) - 1
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
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
            Self::Relocate { from, to } => from as u16 * MAX_CELL_COUNT as u16 + to as u16,
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
    pub config: BoardConfig,
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
        Self::with_config_const(BoardConfig::DEFAULT)
    }

    pub fn with_board_size(board_size: u8) -> Self {
        let reserve = board_size.saturating_mul(2);
        Self::with_config(BoardConfig::new(board_size, reserve).expect("valid board size"))
    }

    pub fn with_config(config: BoardConfig) -> Self {
        Self::with_config_const(config)
    }

    pub fn from_position(position: &crate::contract::Position) -> Result<Self, String> {
        position.validate()?;
        let config = BoardConfig::from_contract(&position.config)?;
        let mut light = 0_u64;
        let mut dark = 0_u64;
        for (square, piece) in position.board.iter().enumerate() {
            let mask = bit(square as u8);
            match piece {
                Some(crate::contract::ContractPlayer::Light) => light |= mask,
                Some(crate::contract::ContractPlayer::Dark) => dark |= mask,
                None => {}
            }
        }
        let forbidden = position
            .forbidden
            .iter()
            .fold(0_u64, |mask, square| mask | bit(*square));
        let reserve = [position.reserve.light, position.reserve.dark];
        if reserve.iter().any(|value| *value > u8::MAX as u16)
            || light & dark != 0
            || forbidden & (light | dark) != 0
            || light.count_ones() as u16 + reserve[0] != u16::from(config.reserve_per_player)
            || dark.count_ones() as u16 + reserve[1] != u16::from(config.reserve_per_player)
        {
            return Err(
                "seeded position violates board, forbidden-square, or inventory invariants"
                    .to_owned(),
            );
        }
        let state = Self {
            config,
            light,
            dark,
            reserve: [reserve[0] as u8, reserve[1] as u8],
            turn: match position.turn {
                crate::contract::ContractPlayer::Light => Player::Light,
                crate::contract::ContractPlayer::Dark => Player::Dark,
            },
            forbidden,
            last_relocated_to: [
                position.last_relocated_to.light,
                position.last_relocated_to.dark,
            ],
            last_capture: 0,
            last_player: None,
            winner: None,
            ply: position.ply,
        };
        if has_winning_path(state, Player::Light) || has_winning_path(state, Player::Dark) {
            return Err("seeded position must not contain an active winning path".to_owned());
        }
        if state.legal_actions().is_empty() {
            return Err("seeded position must have at least one legal action".to_owned());
        }
        Ok(state)
    }

    const fn with_config_const(config: BoardConfig) -> Self {
        Self {
            config,
            light: 0,
            dark: 0,
            reserve: [config.reserve_per_player, config.reserve_per_player],
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
        let destinations = self.config.full_board() & !(self.light | self.dark | self.forbidden);
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
        let mut actions =
            Vec::with_capacity(sources.count_ones() as usize * destination_squares.len());
        for from in squares(sources) {
            for &to in &destination_squares {
                actions.push(Action::Relocate { from, to });
            }
        }
        actions
    }

    /// Count legal actions without materializing the action list.
    ///
    /// This is used by evaluators that only need mobility. Keeping it beside
    /// `legal_actions` makes the count share exactly the same rule boundary
    /// without paying for thousands of short-lived `Action` vectors.
    pub fn legal_action_count(self) -> usize {
        if self.winner.is_some() {
            return 0;
        }
        let destinations = self.config.full_board() & !(self.light | self.dark | self.forbidden);
        let destination_count = destinations.count_ones() as usize;
        if self.reserve[self.turn.index()] > 0 {
            return destination_count;
        }
        let mut sources = self.pieces(self.turn);
        if let Some(square) = self.last_relocated_to[self.turn.index()] {
            sources &= !bit(square);
        }
        sources.count_ones() as usize * destination_count
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
        Transition {
            state: self,
            captured,
        }
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
        row_mask(state.config.board_size, state.config.board_size - 1)
    } else {
        column_mask(state.config.board_size, 0)
    };
    let far_edge = if player == Player::Light {
        row_mask(state.config.board_size, 0)
    } else {
        column_mask(state.config.board_size, state.config.board_size - 1)
    };
    let mut frontier = pieces & near_edge;
    let mut visited = frontier;
    while frontier != 0 {
        if frontier & far_edge != 0 {
            return true;
        }
        let mut next = 0_u64;
        for square in squares(frontier) {
            next |= neighbor_mask_for(state.config.board_size, square);
        }
        frontier = next & pieces & !visited;
        visited |= frontier;
    }
    false
}

pub fn winning_path(state: GameState, player: Player) -> Vec<u8> {
    let pieces = state.pieces(player);
    let near_edge = if player == Player::Light {
        row_mask(state.config.board_size, state.config.board_size - 1)
    } else {
        column_mask(state.config.board_size, 0)
    };
    let far_edge = if player == Player::Light {
        row_mask(state.config.board_size, 0)
    } else {
        column_mask(state.config.board_size, state.config.board_size - 1)
    };
    let starts: Vec<u8> = squares(pieces & near_edge).collect();
    let mut queue = VecDeque::from(starts.clone());
    let mut visited = pieces & near_edge;
    let mut parent = [None; MAX_CELL_COUNT as usize];
    while let Some(square) = queue.pop_front() {
        if bit(square) & far_edge != 0 {
            let mut path = vec![square];
            let mut cursor = square;
            while let Some(previous) = parent[cursor as usize] {
                path.push(previous);
                cursor = previous;
            }
            path.reverse();
            return path;
        }
        for next in squares(neighbor_mask_for(state.config.board_size, square) & pieces & !visited)
        {
            visited |= bit(next);
            parent[next as usize] = Some(square);
            queue.push_back(next);
        }
    }
    Vec::new()
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

pub(crate) fn neighbor_mask_for(board_size: u8, square: u8) -> u64 {
    NEIGHBOR_MASKS[board_size as usize][square as usize]
}

pub(crate) fn captures_from(state: GameState, origin: u8, player: Player) -> u64 {
    let opponent_pieces = state.pieces(player.other());
    let own_pieces = state.pieces(player);
    let mut captured = 0;
    for ray in CAPTURE_RAYS[state.config.board_size as usize][origin as usize] {
        if opponent_pieces & ray.near != 0 && own_pieces & ray.far != 0 {
            captured |= ray.near;
        }
    }
    captured
}

pub(crate) const fn row_mask(board_size: u8, row: u8) -> u64 {
    let mut mask = 0;
    let mut column = 0;
    while column < board_size {
        mask |= bit(row * board_size + column);
        column += 1;
    }
    mask
}

pub(crate) const fn column_mask(board_size: u8, column: u8) -> u64 {
    let mut mask = 0;
    let mut row = 0;
    while row < board_size {
        mask |= bit(row * board_size + column);
        row += 1;
    }
    mask
}

fn parse_square(text: &str) -> Result<u8, String> {
    let square: u8 = text
        .parse()
        .map_err(|_| format!("invalid square: {text}"))?;
    if square >= MAX_CELL_COUNT {
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
    fn variable_boards_have_size_appropriate_place_moves() {
        for size in 4..=7 {
            let state = GameState::with_board_size(size);
            assert_eq!(state.legal_actions().len(), (size * size) as usize);
        }
    }

    #[test]
    fn relocation_changes_the_board_and_cannot_repeat_piece() {
        let mut state = GameState::new();
        state.light = (0..14).fold(0, |mask, square| mask | bit(square));
        state.reserve[Player::Light.index()] = 0;
        let moved = state
            .apply(Action::Relocate { from: 0, to: 48 })
            .unwrap()
            .state;
        assert!(!moved
            .legal_actions()
            .iter()
            .any(|action| matches!(action, Action::Relocate { from: 48, .. })));
        assert!(state.apply(Action::Relocate { from: 0, to: 0 }).is_err());
    }

    #[test]
    fn allocation_free_action_count_matches_canonical_actions() {
        let mut state = GameState::new();
        let mut seed = 0x51f2_9ab3_u32;
        for _ in 0..96 {
            assert_eq!(state.legal_action_count(), state.legal_actions().len());
            let actions = state.legal_actions();
            let Some(action) = actions.get((seed as usize) % actions.len().max(1)).copied() else {
                break;
            };
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state = state.apply_legal(action).state;
            if state.winner.is_some() {
                break;
            }
        }
    }

    #[test]
    fn bitboard_winner_check_matches_path_reconstruction() {
        for board_size in MIN_BOARD_SIZE..=MAX_BOARD_SIZE {
            let mut state = GameState::with_board_size(board_size);
            let mut seed = u32::from(board_size).wrapping_mul(0x9e37_79b9);
            for _ in 0..128 {
                for player in [Player::Light, Player::Dark] {
                    assert_eq!(
                        has_winning_path(state, player),
                        !winning_path(state, player).is_empty(),
                        "winner mismatch on {board_size}x{board_size} for {}",
                        player.as_str()
                    );
                }
                let actions = state.legal_actions();
                let Some(action) = actions.get((seed as usize) % actions.len().max(1)).copied()
                else {
                    break;
                };
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                state = state.apply_legal(action).state;
                if state.winner.is_some() {
                    break;
                }
            }
        }
    }
}
