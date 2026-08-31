//! Runtime access to promoted, exact endgame data.
//!
//! The durable shard contains exact side-to-move W/D/L values. The optional
//! compact sidecar carries sparse per-action results, distances, and the
//! complete-action-set bit; omitted actions on an incomplete row are unknown.
//! This module intentionally keeps both pieces read-only: a search may consult
//! a promoted result, but approximate search output can never mutate the gold.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::{Action, BoardConfig, GameState, Player};
use serde_json::Value;

pub const LOSS: u8 = 0;
pub const DRAW: u8 = 1;
pub const WIN: u8 = 2;

const ACTION_ALPHABET: &[u8; 64] =
    b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_";
const ACTION_BOOK_V1_MAGIC: &[u8; 8] = b"PGACT01\0";
const ACTION_BOOK_V2_MAGIC: &[u8; 8] = b"PGACT02\0";
const ACTION_BOOK_NONE_DISTANCE: u16 = u16::MAX;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GoldenOutcome {
    Loss,
    Draw,
    Win,
}

impl GoldenOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Loss => "loss",
            Self::Draw => "draw",
            Self::Win => "win",
        }
    }

    pub const fn as_byte(self) -> u8 {
        match self {
            Self::Loss => LOSS,
            Self::Draw => DRAW,
            Self::Win => WIN,
        }
    }

    pub const fn from_byte(value: u8) -> Option<Self> {
        match value {
            LOSS => Some(Self::Loss),
            DRAW => Some(Self::Draw),
            WIN => Some(Self::Win),
            _ => None,
        }
    }
}

/// A sparse action label from a promoted sidecar. `None` is an explicitly
/// encoded unknown action result; an omitted action is also unknown whenever
/// the row's complete-action-set flag is false.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GoldenActionValue {
    pub action: Action,
    pub outcome: Option<GoldenOutcome>,
    pub distance: Option<u16>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GoldenRowValue {
    pub outcome: GoldenOutcome,
    pub distance: Option<u16>,
    pub optimal_actions_complete: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GoldenLookupStats {
    pub table_hits: u64,
    pub action_hits: u64,
}

#[derive(Debug)]
pub struct FlatGoldenTable {
    path: PathBuf,
    board_size: u8,
    reserve_per_player: u8,
    key_bytes: usize,
    row_bytes: u64,
    rows: u64,
    file: Mutex<File>,
}

/// In-memory variant used by the WASM/browser boundary. The on-disk table is
/// still the durable source, while callers can fetch its immutable bytes and
/// let the same binary-search and canonical-key logic run without a filesystem.
#[derive(Clone, Debug)]
pub struct MemoryGoldenTable {
    bytes: Vec<u8>,
    board_size: u8,
    reserve_per_player: u8,
    key_bytes: usize,
    row_bytes: usize,
}

impl FlatGoldenTable {
    pub fn open(
        path: impl AsRef<Path>,
        board_size: u8,
        reserve_per_player: u8,
    ) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let metadata = file.metadata()?;
        let key_bytes = key_bytes_for_board_size(board_size).map_err(invalid_data)?;
        let row_bytes = (key_bytes + 1) as u64;
        if metadata.len() % row_bytes != 0 {
            return Err(invalid_data(format!(
                "golden shard {} is not a multiple of {row_bytes} bytes",
                path.display()
            )));
        }
        Ok(Self {
            path,
            board_size,
            reserve_per_player,
            key_bytes,
            row_bytes,
            rows: metadata.len() / row_bytes,
            file: Mutex::new(file),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn rows(&self) -> u64 {
        self.rows
    }

    pub fn lookup(&self, state: GameState) -> Option<GoldenOutcome> {
        if !state_matches_namespace(state, self.board_size, self.reserve_per_player) {
            return None;
        }
        let key = canonical_key(state).0;
        self.lookup_key(&key)
    }

    pub fn lookup_key(&self, key: &[u8]) -> Option<GoldenOutcome> {
        if key.len() != self.key_bytes {
            return None;
        }
        let mut file = self.file.lock().ok()?;
        let mut low = 0_u64;
        let mut high = self.rows;
        let mut row = vec![0_u8; self.row_bytes as usize];
        while low < high {
            let middle = low + (high - low) / 2;
            file.seek(SeekFrom::Start(middle * self.row_bytes)).ok()?;
            file.read_exact(&mut row).ok()?;
            match row[..self.key_bytes].cmp(key) {
                std::cmp::Ordering::Less => low = middle + 1,
                std::cmp::Ordering::Greater => high = middle,
                std::cmp::Ordering::Equal => return GoldenOutcome::from_byte(row[self.key_bytes]),
            }
        }
        None
    }
}

impl MemoryGoldenTable {
    pub fn from_bytes(bytes: &[u8], board_size: u8, reserve_per_player: u8) -> io::Result<Self> {
        let key_bytes = key_bytes_for_board_size(board_size).map_err(invalid_data)?;
        let row_bytes = key_bytes + 1;
        if bytes.len() % row_bytes != 0 {
            return Err(invalid_data(format!(
                "memory golden shard is not a multiple of {row_bytes} bytes"
            )));
        }
        Ok(Self {
            bytes: bytes.to_vec(),
            board_size,
            reserve_per_player,
            key_bytes,
            row_bytes,
        })
    }

    pub fn rows(&self) -> u64 {
        (self.bytes.len() / self.row_bytes) as u64
    }

    pub fn lookup(&self, state: GameState) -> Option<GoldenOutcome> {
        if !state_matches_namespace(state, self.board_size, self.reserve_per_player) {
            return None;
        }
        self.lookup_key(&canonical_key(state).0)
    }

    pub fn lookup_key(&self, key: &[u8]) -> Option<GoldenOutcome> {
        if key.len() != self.key_bytes {
            return None;
        }
        let mut low = 0_usize;
        let mut high = self.bytes.len() / self.row_bytes;
        while low < high {
            let middle = low + (high - low) / 2;
            let start = middle * self.row_bytes;
            let row = &self.bytes[start..start + self.row_bytes];
            match row[..self.key_bytes].cmp(key) {
                std::cmp::Ordering::Less => low = middle + 1,
                std::cmp::Ordering::Greater => high = middle,
                std::cmp::Ordering::Equal => return GoldenOutcome::from_byte(row[self.key_bytes]),
            }
        }
        None
    }
}

#[derive(Clone, Debug, Default)]
pub struct GoldenActionBook {
    board_size: u8,
    actions: HashMap<Vec<u8>, Vec<Action>>,
    action_values: HashMap<Vec<u8>, Vec<GoldenActionValue>>,
    rows: HashMap<Vec<u8>, GoldenRowValue>,
}

impl GoldenActionBook {
    pub fn from_bytes(source: &[u8], board_size: u8) -> io::Result<Self> {
        Self::load_binary(source, Path::new("<memory>"), board_size)
    }

    pub fn load(path: impl AsRef<Path>, board_size: u8) -> io::Result<Self> {
        let path = path.as_ref();
        let mut source = Vec::new();
        File::open(path)?.read_to_end(&mut source)?;
        if source.starts_with(ACTION_BOOK_V1_MAGIC) || source.starts_with(ACTION_BOOK_V2_MAGIC) {
            return Self::load_binary(&source, path, board_size);
        }
        let reader = BufReader::new(Cursor::new(source));
        let key_bytes = key_bytes_for_board_size(board_size).map_err(invalid_data)?;
        let mut book = Self {
            board_size,
            actions: HashMap::new(),
            action_values: HashMap::new(),
            rows: HashMap::new(),
        };
        for (line_number, line) in reader.lines().enumerate() {
            let line = line?;
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            let row: Value = serde_json::from_str(&line).map_err(|error| {
                invalid_data(format!(
                    "{}:{}: invalid JSON: {error}",
                    path.display(),
                    line_number + 1
                ))
            })?;
            if row.get("schemaVersion").and_then(Value::as_u64) != Some(1)
                || row.get("tableFamily").and_then(Value::as_str) != Some("fresh-frontier-wdl-v1")
            {
                return Err(invalid_data(format!(
                    "{}:{}: unsupported golden action row",
                    path.display(),
                    line_number + 1
                )));
            }
            let key_hex = row.get("key").and_then(Value::as_str).ok_or_else(|| {
                invalid_data(format!(
                    "{}:{}: missing key",
                    path.display(),
                    line_number + 1
                ))
            })?;
            let key = decode_hex(key_hex).map_err(|error| {
                invalid_data(format!("{}:{}: {error}", path.display(), line_number + 1))
            })?;
            if key.len() != key_bytes {
                return Err(invalid_data(format!(
                    "{}:{}: key has {} bytes, expected {key_bytes}",
                    path.display(),
                    line_number + 1,
                    key.len()
                )));
            }
            let actions = row
                .get("provenActions")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    invalid_data(format!(
                        "{}:{}: missing provenActions",
                        path.display(),
                        line_number + 1
                    ))
                })?;
            let row_value = row
                .get("outcome")
                .and_then(Value::as_str)
                .and_then(parse_golden_outcome)
                .map(|outcome| GoldenRowValue {
                    outcome,
                    distance: row
                        .get("distance")
                        .and_then(Value::as_u64)
                        .map(|value| value as u16),
                    optimal_actions_complete: row
                        .get("optimalActionsKnown")
                        .and_then(Value::as_bool)
                        .unwrap_or(false),
                });
            let mut decoded = Vec::with_capacity(actions.len());
            for action_record in actions {
                let token = action_record
                    .get("token")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid_data("golden action is missing token".to_owned()))?;
                let action = decode_action(token, board_size).map_err(|error| {
                    invalid_data(format!("{}:{}: {error}", path.display(), line_number + 1))
                })?;
                decoded.push(action);
                book.action_values
                    .entry(key.clone())
                    .or_default()
                    .push(GoldenActionValue {
                        action,
                        outcome: action_record
                            .get("outcome")
                            .and_then(Value::as_str)
                            .and_then(parse_golden_outcome),
                        distance: action_record
                            .get("distance")
                            .and_then(Value::as_u64)
                            .map(|value| value as u16),
                    });
            }
            decoded.sort_by_key(|action| action.order());
            decoded.dedup();
            let existing = book.actions.entry(key.clone()).or_default();
            existing.extend(decoded);
            existing.sort_by_key(|action| action.order());
            existing.dedup();
            if let Some(row_value) = row_value {
                book.rows.insert(key, row_value);
            }
        }
        Ok(book)
    }

    fn load_binary(source: &[u8], path: &Path, board_size: u8) -> io::Result<Self> {
        if source.starts_with(ACTION_BOOK_V2_MAGIC) {
            return Self::load_binary_v2(source, path, board_size);
        }
        if !source.starts_with(ACTION_BOOK_V1_MAGIC) {
            return Err(invalid_data(format!(
                "{}: unsupported action-book format",
                path.display()
            )));
        }
        Self::load_binary_v1(source, path, board_size)
    }

    fn load_binary_v1(source: &[u8], path: &Path, board_size: u8) -> io::Result<Self> {
        const HEADER_BYTES: usize = 16;
        if source.len() < HEADER_BYTES {
            return Err(invalid_data(format!(
                "{}: truncated golden action-book header",
                path.display()
            )));
        }
        if source[8] != board_size || source[9] != 14 {
            return Err(invalid_data(format!(
                "{}: action-book namespace does not match {}x{}/14",
                path.display(),
                board_size,
                board_size
            )));
        }
        let key_bytes = key_bytes_for_board_size(board_size).map_err(invalid_data)?;
        if usize::from(source[10]) != key_bytes || source[11] != 0 {
            return Err(invalid_data(format!(
                "{}: unsupported action-book key width",
                path.display()
            )));
        }
        let rows = u32::from_le_bytes(source[12..16].try_into().expect("four bytes")) as usize;
        let mut offset = HEADER_BYTES;
        let mut book = Self {
            board_size,
            actions: HashMap::with_capacity(rows),
            action_values: HashMap::with_capacity(rows),
            rows: HashMap::new(),
        };
        let mut previous_key = None;
        for row_index in 0..rows {
            if offset + key_bytes + 2 > source.len() {
                return Err(invalid_data(format!(
                    "{}: truncated action-book row {row_index}",
                    path.display()
                )));
            }
            let key = source[offset..offset + key_bytes].to_vec();
            offset += key_bytes;
            if previous_key
                .as_ref()
                .is_some_and(|previous: &Vec<u8>| *previous >= key)
            {
                return Err(invalid_data(format!(
                    "{}: action-book keys are not strictly sorted at row {row_index}",
                    path.display()
                )));
            }
            let count =
                u16::from_le_bytes(source[offset..offset + 2].try_into().expect("two bytes"))
                    as usize;
            offset += 2;
            if offset + count * 2 > source.len() {
                return Err(invalid_data(format!(
                    "{}: truncated actions at row {row_index}",
                    path.display()
                )));
            }
            let mut actions = Vec::with_capacity(count);
            let mut action_values = Vec::with_capacity(count);
            for _ in 0..count {
                let code =
                    u16::from_le_bytes(source[offset..offset + 2].try_into().expect("two bytes"));
                offset += 2;
                let action = action_from_code(code, board_size)
                    .map_err(|error| invalid_data(format!("{}: {error}", path.display())))?;
                actions.push(action);
                action_values.push(GoldenActionValue {
                    action,
                    outcome: Some(GoldenOutcome::Win),
                    distance: Some(1),
                });
            }
            actions.sort_by_key(|action| action.order());
            actions.dedup();
            previous_key = Some(key.clone());
            book.actions.insert(key, actions);
            book.action_values
                .insert(previous_key.clone().expect("key inserted"), action_values);
        }
        if offset != source.len() {
            return Err(invalid_data(format!(
                "{}: trailing bytes after action-book rows",
                path.display()
            )));
        }
        Ok(book)
    }

    fn load_binary_v2(source: &[u8], path: &Path, board_size: u8) -> io::Result<Self> {
        const HEADER_BYTES: usize = 16;
        if source.len() < HEADER_BYTES {
            return Err(invalid_data(format!(
                "{}: truncated golden action-book v2 header",
                path.display()
            )));
        }
        if source[8] != board_size || source[9] != 14 {
            return Err(invalid_data(format!(
                "{}: action-book namespace does not match {}x{}/14",
                path.display(),
                board_size,
                board_size
            )));
        }
        let key_bytes = key_bytes_for_board_size(board_size).map_err(invalid_data)?;
        if usize::from(source[10]) != key_bytes || source[11] != 0 {
            return Err(invalid_data(format!(
                "{}: unsupported action-book v2 key width",
                path.display()
            )));
        }
        let rows = u32::from_le_bytes(source[12..16].try_into().expect("four bytes")) as usize;
        let mut offset = HEADER_BYTES;
        let mut book = Self {
            board_size,
            actions: HashMap::with_capacity(rows),
            action_values: HashMap::with_capacity(rows),
            rows: HashMap::with_capacity(rows),
        };
        let mut previous_key = None;
        for row_index in 0..rows {
            let row_header_bytes = key_bytes + 1 + 1 + 2 + 2;
            if offset + row_header_bytes > source.len() {
                return Err(invalid_data(format!(
                    "{}: truncated v2 action-book row {row_index}",
                    path.display()
                )));
            }
            let key = source[offset..offset + key_bytes].to_vec();
            offset += key_bytes;
            if previous_key
                .as_ref()
                .is_some_and(|previous: &Vec<u8>| *previous >= key)
            {
                return Err(invalid_data(format!(
                    "{}: action-book v2 keys are not strictly sorted at row {row_index}",
                    path.display()
                )));
            }
            let flags = source[offset];
            offset += 1;
            if flags & !1 != 0 {
                return Err(invalid_data(format!(
                    "{}: unsupported v2 action-book flags at row {row_index}",
                    path.display()
                )));
            }
            let outcome = GoldenOutcome::from_byte(source[offset]).ok_or_else(|| {
                invalid_data(format!(
                    "{}: invalid v2 row outcome at row {row_index}",
                    path.display()
                ))
            })?;
            offset += 1;
            let distance = u16::from_le_bytes(
                source[offset..offset + 2]
                    .try_into()
                    .expect("two distance bytes"),
            );
            offset += 2;
            if outcome == GoldenOutcome::Draw && distance != ACTION_BOOK_NONE_DISTANCE {
                return Err(invalid_data(format!(
                    "{}: draw v2 row has a terminal distance at row {row_index}",
                    path.display()
                )));
            }
            if outcome != GoldenOutcome::Draw && distance == ACTION_BOOK_NONE_DISTANCE {
                return Err(invalid_data(format!(
                    "{}: known v2 row lacks a terminal distance at row {row_index}",
                    path.display()
                )));
            }
            let count = u16::from_le_bytes(
                source[offset..offset + 2]
                    .try_into()
                    .expect("two action-count bytes"),
            ) as usize;
            offset += 2;
            let mut actions = Vec::with_capacity(count);
            let mut action_values = Vec::with_capacity(count);
            for action_index in 0..count {
                if offset + 5 > source.len() {
                    return Err(invalid_data(format!(
                        "{}: truncated v2 action {action_index} in row {row_index}",
                        path.display()
                    )));
                }
                let code = u16::from_le_bytes(
                    source[offset..offset + 2]
                        .try_into()
                        .expect("two action bytes"),
                );
                offset += 2;
                let action = action_from_code(code, board_size)
                    .map_err(|error| invalid_data(format!("{}: {error}", path.display())))?;
                let action_outcome = match source[offset] {
                    3 => None,
                    value => GoldenOutcome::from_byte(value),
                };
                if source[offset] != 3 && action_outcome.is_none() {
                    return Err(invalid_data(format!(
                        "{}: invalid v2 action outcome at row {row_index}",
                        path.display()
                    )));
                }
                offset += 1;
                let action_distance = u16::from_le_bytes(
                    source[offset..offset + 2]
                        .try_into()
                        .expect("two action-distance bytes"),
                );
                offset += 2;
                let action_distance =
                    (action_distance != ACTION_BOOK_NONE_DISTANCE).then_some(action_distance);
                if action_outcome.is_none() && action_distance.is_some() {
                    return Err(invalid_data(format!(
                        "{}: unknown v2 action has a distance at row {row_index}",
                        path.display()
                    )));
                }
                if action_outcome == Some(GoldenOutcome::Draw) && action_distance.is_some() {
                    return Err(invalid_data(format!(
                        "{}: draw v2 action has a terminal distance at row {row_index}",
                        path.display()
                    )));
                }
                if matches!(
                    action_outcome,
                    Some(GoldenOutcome::Loss | GoldenOutcome::Win)
                ) && action_distance.is_none()
                {
                    return Err(invalid_data(format!(
                        "{}: known v2 action lacks a terminal distance at row {row_index}",
                        path.display()
                    )));
                }
                if actions.contains(&action) {
                    return Err(invalid_data(format!(
                        "{}: duplicate v2 action at row {row_index}",
                        path.display()
                    )));
                }
                actions.push(action);
                action_values.push(GoldenActionValue {
                    action,
                    outcome: action_outcome,
                    distance: action_distance,
                });
            }
            previous_key = Some(key.clone());
            book.actions.insert(key.clone(), actions);
            book.action_values.insert(key.clone(), action_values);
            book.rows.insert(
                key,
                GoldenRowValue {
                    outcome,
                    distance: (distance != ACTION_BOOK_NONE_DISTANCE).then_some(distance),
                    optimal_actions_complete: flags & 1 != 0,
                },
            );
        }
        if offset != source.len() {
            return Err(invalid_data(format!(
                "{}: trailing bytes after v2 action-book rows",
                path.display()
            )));
        }
        Ok(book)
    }

    pub fn rows(&self) -> usize {
        self.actions.len().max(self.rows.len())
    }

    pub fn row_value(&self, state: GameState) -> Option<GoldenRowValue> {
        if state.config.board_size != self.board_size {
            return None;
        }
        let (key, _) = canonical_key(state);
        self.rows.get(&key).copied()
    }

    pub fn optimal_actions_complete(&self, state: GameState) -> Option<bool> {
        self.row_value(state)
            .map(|value| value.optimal_actions_complete)
    }

    /// Return sparse action labels in the caller's board orientation. An
    /// absent action on an incomplete row is unknown by contract.
    pub fn action_values(&self, state: GameState) -> Option<Vec<GoldenActionValue>> {
        if state.config.board_size != self.board_size {
            return None;
        }
        let (key, _) = canonical_key(state);
        let candidates = self.action_values.get(&key)?;
        let mut oriented = Vec::new();
        for symmetry in 0..8 {
            if pack_transformed(state, symmetry) != key {
                continue;
            }
            let inverse = inverse_symmetry(symmetry);
            for candidate in candidates {
                let value = GoldenActionValue {
                    action: transform_action(candidate.action, state.config.board_size, inverse),
                    outcome: candidate.outcome,
                    distance: candidate.distance,
                };
                if !oriented.contains(&value) {
                    oriented.push(value);
                }
            }
        }
        Some(oriented)
    }

    pub fn proven_action(&self, state: GameState) -> Option<Action> {
        let legal = state.legal_actions();
        if legal.is_empty() {
            return None;
        }
        self.action_values(state)?
            .into_iter()
            .filter(|value| value.outcome == Some(GoldenOutcome::Win))
            .map(|value| value.action)
            .filter(|action| legal.contains(action))
            .min_by_key(|action| action.order())
    }
}

fn parse_golden_outcome(value: &str) -> Option<GoldenOutcome> {
    match value {
        "loss" => Some(GoldenOutcome::Loss),
        "draw" => Some(GoldenOutcome::Draw),
        "win" => Some(GoldenOutcome::Win),
        _ => None,
    }
}

#[derive(Debug)]
pub struct GoldenLookup {
    pub table: FlatGoldenTable,
    pub actions: Option<GoldenActionBook>,
}

#[derive(Clone, Debug)]
pub struct MemoryGoldenLookup {
    pub table: MemoryGoldenTable,
    pub actions: Option<GoldenActionBook>,
}

/// Ordered collection of immutable on-disk golden layers. The first layer
/// containing a position owns that position; later layers are consulted only
/// when the position is absent. This lets a newer frontier ring overlay an
/// older control table without copying either artifact or weakening the
/// rollback boundary.
#[derive(Debug)]
pub struct GoldenLookupLayers {
    layers: Vec<GoldenLookup>,
}

/// In-memory equivalent of [`GoldenLookupLayers`] for WASM callers that have
/// fetched multiple versioned shards. Bytes are copied when opened, so the
/// caller may reuse its fetch buffers after construction.
#[derive(Clone, Debug)]
pub struct MemoryGoldenLookupLayers {
    layers: Vec<MemoryGoldenLookup>,
}

impl GoldenLookup {
    pub fn open(
        table_path: impl AsRef<Path>,
        sidecar_path: Option<impl AsRef<Path>>,
        board_size: u8,
        reserve_per_player: u8,
    ) -> io::Result<Self> {
        let table = FlatGoldenTable::open(table_path, board_size, reserve_per_player)?;
        let actions = sidecar_path
            .map(|path| GoldenActionBook::load(path, board_size))
            .transpose()?;
        Ok(Self { table, actions })
    }

    pub fn lookup(&self, state: GameState) -> Option<GoldenOutcome> {
        self.table.lookup(state)
    }

    pub fn proven_action(&self, state: GameState) -> Option<Action> {
        if self.lookup(state) != Some(GoldenOutcome::Win) {
            return None;
        }
        self.actions.as_ref()?.proven_action(state)
    }

    pub fn row_value(&self, state: GameState) -> Option<GoldenRowValue> {
        self.lookup(state).and_then(|_| {
            self.actions
                .as_ref()
                .and_then(|actions| actions.row_value(state))
        })
    }

    pub fn action_values(&self, state: GameState) -> Option<Vec<GoldenActionValue>> {
        self.lookup(state).and_then(|_| {
            self.actions
                .as_ref()
                .and_then(|actions| actions.action_values(state))
        })
    }
}

impl GoldenLookupLayers {
    pub fn from_lookup(lookup: GoldenLookup) -> Self {
        Self {
            layers: vec![lookup],
        }
    }

    /// Open layers in lookup priority order. A position found in an earlier
    /// layer is never replaced by a later layer.
    pub fn open(
        paths: &[(PathBuf, Option<PathBuf>)],
        board_size: u8,
        reserve_per_player: u8,
    ) -> io::Result<Self> {
        if paths.is_empty() {
            return Err(invalid_data("golden lookup requires at least one layer"));
        }
        let layers = paths
            .iter()
            .map(|(table, sidecar)| {
                GoldenLookup::open(table, sidecar.as_deref(), board_size, reserve_per_player)
            })
            .collect::<io::Result<Vec<_>>>()?;
        Ok(Self { layers })
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn rows(&self) -> u64 {
        self.layers.iter().map(|layer| layer.table.rows()).sum()
    }

    pub fn lookup(&self, state: GameState) -> Option<GoldenOutcome> {
        self.layers.iter().find_map(|layer| layer.lookup(state))
    }

    pub fn proven_action(&self, state: GameState) -> Option<Action> {
        for layer in &self.layers {
            if layer.lookup(state).is_some() {
                return layer.proven_action(state);
            }
        }
        None
    }

    pub fn row_value(&self, state: GameState) -> Option<GoldenRowValue> {
        for layer in &self.layers {
            if layer.lookup(state).is_some() {
                return layer.row_value(state);
            }
        }
        None
    }

    pub fn action_values(&self, state: GameState) -> Option<Vec<GoldenActionValue>> {
        for layer in &self.layers {
            if layer.lookup(state).is_some() {
                return layer.action_values(state);
            }
        }
        None
    }
}

impl MemoryGoldenLookup {
    pub fn open_bytes(
        table_bytes: &[u8],
        sidecar_bytes: Option<&[u8]>,
        board_size: u8,
        reserve_per_player: u8,
    ) -> io::Result<Self> {
        let table = MemoryGoldenTable::from_bytes(table_bytes, board_size, reserve_per_player)?;
        let actions = sidecar_bytes
            .filter(|bytes| !bytes.is_empty())
            .map(|bytes| GoldenActionBook::from_bytes(bytes, board_size))
            .transpose()?;
        Ok(Self { table, actions })
    }

    pub fn lookup(&self, state: GameState) -> Option<GoldenOutcome> {
        self.table.lookup(state)
    }

    pub fn proven_action(&self, state: GameState) -> Option<Action> {
        if self.lookup(state) != Some(GoldenOutcome::Win) {
            return None;
        }
        self.actions.as_ref()?.proven_action(state)
    }

    pub fn row_value(&self, state: GameState) -> Option<GoldenRowValue> {
        self.lookup(state).and_then(|_| {
            self.actions
                .as_ref()
                .and_then(|actions| actions.row_value(state))
        })
    }

    pub fn action_values(&self, state: GameState) -> Option<Vec<GoldenActionValue>> {
        self.lookup(state).and_then(|_| {
            self.actions
                .as_ref()
                .and_then(|actions| actions.action_values(state))
        })
    }
}

impl MemoryGoldenLookupLayers {
    pub fn open_bytes(
        layers: &[(&[u8], Option<&[u8]>)],
        board_size: u8,
        reserve_per_player: u8,
    ) -> io::Result<Self> {
        if layers.is_empty() {
            return Err(invalid_data("golden lookup requires at least one layer"));
        }
        let layers = layers
            .iter()
            .map(|(table, sidecar)| {
                MemoryGoldenLookup::open_bytes(table, *sidecar, board_size, reserve_per_player)
            })
            .collect::<io::Result<Vec<_>>>()?;
        Ok(Self { layers })
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn rows(&self) -> u64 {
        self.layers.iter().map(|layer| layer.table.rows()).sum()
    }

    pub fn lookup(&self, state: GameState) -> Option<GoldenOutcome> {
        self.layers.iter().find_map(|layer| layer.lookup(state))
    }

    pub fn proven_action(&self, state: GameState) -> Option<Action> {
        for layer in &self.layers {
            if layer.lookup(state).is_some() {
                return layer.proven_action(state);
            }
        }
        None
    }

    pub fn row_value(&self, state: GameState) -> Option<GoldenRowValue> {
        for layer in &self.layers {
            if layer.lookup(state).is_some() {
                return layer.row_value(state);
            }
        }
        None
    }

    pub fn action_values(&self, state: GameState) -> Option<Vec<GoldenActionValue>> {
        for layer in &self.layers {
            if layer.lookup(state).is_some() {
                return layer.action_values(state);
            }
        }
        None
    }
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn state_matches_namespace(state: GameState, board_size: u8, reserve_per_player: u8) -> bool {
    state.config.board_size == board_size
        && state.config.reserve_per_player == reserve_per_player
        && board_size <= 7
}

fn key_bytes_for_board_size(board_size: u8) -> Result<usize, String> {
    if !(3..=7).contains(&board_size) {
        return Err("golden key codec supports board sizes 3 through 7".to_owned());
    }
    let cells = usize::from(board_size) * usize::from(board_size);
    let marker_bits = (usize::BITS - cells.leading_zeros()) as usize;
    Ok((2 * cells + 1 + 2 * marker_bits).div_ceil(8))
}

fn canonical_key(state: GameState) -> (Vec<u8>, u8) {
    let mut best = pack_transformed(state, 0);
    let mut best_symmetry = 0;
    for symmetry in 1..8 {
        let candidate = pack_transformed(state, symmetry);
        if candidate < best {
            best = candidate;
            best_symmetry = symmetry;
        }
    }
    (best, best_symmetry)
}

/// Return the stable little-endian D4-canonical key used by golden tables and
/// retrograde frontier interchange files.
pub fn canonical_position_key(state: GameState) -> Vec<u8> {
    canonical_key(state).0
}

/// Apply one of the eight rules-preserving D4 transformations. This is
/// public for Rust-side frontier validation; callers should canonicalize the
/// transformed state with [`canonical_position_key`] rather than persisting
/// an orientation-specific representation.
pub fn transform_position(state: GameState, symmetry: u8) -> GameState {
    transform_state(state, symmetry)
}

/// Decode a canonical key into its canonical representative state. Reserve
/// counts are reconstructed from the fixed inventory and piece counts because
/// the namespace already fixes the reserve-per-player value.
pub fn decode_canonical_position_key(
    key: &[u8],
    board_size: u8,
    reserve_per_player: u8,
) -> Result<GameState, String> {
    let key_bytes = key_bytes_for_board_size(board_size)?;
    if key.len() != key_bytes {
        return Err(format!(
            "canonical key has {} bytes, expected {key_bytes}",
            key.len()
        ));
    }
    let cells = usize::from(board_size) * usize::from(board_size);
    let marker_bits = (usize::BITS - cells.leading_zeros()) as usize;
    let used_bits = 2 * cells + 1 + 2 * marker_bits;
    let mut padded = [0_u8; 16];
    padded[..key.len()].copy_from_slice(key);
    let packed = u128::from_le_bytes(padded);
    let used_mask = if used_bits == u128::BITS as usize {
        u128::MAX
    } else {
        (1_u128 << used_bits) - 1
    };
    if packed & !used_mask != 0 {
        return Err("canonical key sets a reserved high bit".to_owned());
    }
    let mut light = 0_u64;
    let mut dark = 0_u64;
    let mut forbidden = 0_u64;
    for square in 0..cells {
        let code = ((packed >> (2 * square)) & 0b11) as u8;
        match code {
            0 => {}
            1 => light |= 1_u64 << square,
            2 => dark |= 1_u64 << square,
            3 => forbidden |= 1_u64 << square,
            _ => unreachable!(),
        }
    }
    let turn = match ((packed >> (2 * cells)) & 1) as u8 {
        0 => Player::Light,
        1 => Player::Dark,
        _ => unreachable!(),
    };
    let decode_marker = |shift: usize| -> Result<Option<u8>, String> {
        let marker = (packed >> shift) & ((1_u128 << marker_bits) - 1);
        if marker == cells as u128 {
            Ok(None)
        } else if marker < cells as u128 {
            Ok(Some(marker as u8))
        } else {
            Err("canonical key contains an out-of-range relocation marker".to_owned())
        }
    };
    let last_relocated_to = [
        decode_marker(2 * cells + 1)?,
        decode_marker(2 * cells + 1 + marker_bits)?,
    ];
    let light_count = light.count_ones() as u8;
    let dark_count = dark.count_ones() as u8;
    if light & dark != 0 || forbidden & (light | dark) != 0 {
        return Err("canonical key overlaps pieces or forbidden squares".to_owned());
    }
    if light_count > reserve_per_player || dark_count > reserve_per_player {
        return Err("canonical key exceeds the fixed piece inventory".to_owned());
    }
    let config = BoardConfig::new(board_size, reserve_per_player)?;
    Ok(GameState {
        config,
        light,
        dark,
        reserve: [
            reserve_per_player - light_count,
            reserve_per_player - dark_count,
        ],
        turn,
        forbidden,
        last_relocated_to,
        last_capture: 0,
        last_player: None,
        winner: None,
        ply: 0,
    })
}

fn pack_transformed(state: GameState, symmetry: u8) -> Vec<u8> {
    let transformed = transform_state(state, symmetry);
    let cells = usize::from(transformed.config.cells());
    let marker_bits = (usize::BITS - cells.leading_zeros()) as usize;
    let key_bytes =
        key_bytes_for_board_size(transformed.config.board_size).expect("validated board size");
    let mut packed = 0_u128;
    for square in 0..cells {
        let mask = 1_u64 << square;
        let code = if transformed.light & mask != 0 {
            1_u128
        } else if transformed.dark & mask != 0 {
            2_u128
        } else if transformed.forbidden & mask != 0 {
            3_u128
        } else {
            0
        };
        packed |= code << (2 * square);
    }
    packed |= (transformed.turn.index() as u128) << (2 * cells);
    packed |=
        marker_code(transformed.last_relocated_to[Player::Light.index()], cells) << (2 * cells + 1);
    packed |= marker_code(transformed.last_relocated_to[Player::Dark.index()], cells)
        << (2 * cells + 1 + marker_bits);
    packed.to_le_bytes()[..key_bytes].to_vec()
}

fn marker_code(marker: Option<u8>, cells: usize) -> u128 {
    u128::from(marker.map_or(cells as u8, |value| value))
}

fn transform_state(state: GameState, symmetry: u8) -> GameState {
    let swaps_players = matches!(symmetry, 1 | 3 | 6 | 7);
    let light = transform_mask(state.light, state.config.board_size, symmetry);
    let dark = transform_mask(state.dark, state.config.board_size, symmetry);
    let forbidden = transform_mask(state.forbidden, state.config.board_size, symmetry);
    let mut markers = [
        state.last_relocated_to[0]
            .map(|square| transform_square(state.config.board_size, square, symmetry)),
        state.last_relocated_to[1]
            .map(|square| transform_square(state.config.board_size, square, symmetry)),
    ];
    let (light, dark, reserve, turn, markers) = if swaps_players {
        markers.swap(0, 1);
        (
            dark,
            light,
            [state.reserve[1], state.reserve[0]],
            state.turn.other(),
            markers,
        )
    } else {
        (light, dark, state.reserve, state.turn, markers)
    };
    GameState {
        config: state.config,
        light,
        dark,
        reserve,
        turn,
        forbidden,
        last_relocated_to: markers,
        last_capture: state.last_capture,
        last_player: state.last_player.map(|player| {
            if swaps_players {
                player.other()
            } else {
                player
            }
        }),
        winner: state.winner.map(|player| {
            if swaps_players {
                player.other()
            } else {
                player
            }
        }),
        ply: state.ply,
    }
}

fn transform_mask(mask: u64, size: u8, symmetry: u8) -> u64 {
    let mut transformed = 0_u64;
    for square in 0..(size * size) {
        if mask & (1_u64 << square) != 0 {
            transformed |= 1_u64 << transform_square(size, square, symmetry);
        }
    }
    transformed
}

fn transform_square(size: u8, square: u8, symmetry: u8) -> u8 {
    let (row, column) = (square / size, square % size);
    let last = size - 1;
    let (new_row, new_column) = match symmetry {
        0 => (row, column),
        1 => (column, last - row),
        2 => (last - row, last - column),
        3 => (last - column, row),
        4 => (last - row, column),
        5 => (row, last - column),
        6 => (column, row),
        7 => (last - column, last - row),
        _ => unreachable!("D4 symmetry is 0..8"),
    };
    new_row * size + new_column
}

fn inverse_symmetry(symmetry: u8) -> u8 {
    match symmetry {
        0 | 2 | 4 | 5 | 6 | 7 => symmetry,
        1 => 3,
        3 => 1,
        _ => unreachable!("D4 symmetry is 0..8"),
    }
}

fn transform_action(action: Action, size: u8, symmetry: u8) -> Action {
    match action {
        Action::Place { to } => Action::Place {
            to: transform_square(size, to, symmetry),
        },
        Action::Relocate { from, to } => Action::Relocate {
            from: transform_square(size, from, symmetry),
            to: transform_square(size, to, symmetry),
        },
    }
}

fn decode_action(token: &str, board_size: u8) -> Result<Action, String> {
    let bytes = token.as_bytes();
    if bytes.len() != 2 {
        return Err("golden action token must contain exactly two characters".to_owned());
    }
    let first = ACTION_ALPHABET
        .iter()
        .position(|value| *value == bytes[0])
        .ok_or_else(|| format!("invalid golden action token {token:?}"))?;
    let second = ACTION_ALPHABET
        .iter()
        .position(|value| *value == bytes[1])
        .ok_or_else(|| format!("invalid golden action token {token:?}"))?;
    let code = (first << 6) | second;
    action_from_code(code as u16, board_size)
}

fn action_from_code(code: u16, board_size: u8) -> Result<Action, String> {
    let cells = usize::from(board_size) * usize::from(board_size);
    let code = usize::from(code);
    if code < cells {
        Ok(Action::Place { to: code as u8 })
    } else {
        let relocation = code - cells;
        let from = relocation / cells;
        let to = relocation % cells;
        if from >= cells || to >= cells {
            return Err(format!("golden action code {code} is outside the board"));
        }
        Ok(Action::Relocate {
            from: from as u8,
            to: to as u8,
        })
    }
}

fn decode_hex(value: &str) -> Result<Vec<u8>, String> {
    if value.len() % 2 != 0 {
        return Err("golden key hex must have an even length".to_owned());
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    let chars = value.as_bytes();
    for chunk in chars.chunks_exact(2) {
        let high = hex_digit(chunk[0])?;
        let low = hex_digit(chunk[1])?;
        bytes.push((high << 4) | low);
    }
    Ok(bytes)
}

fn hex_digit(value: u8) -> Result<u8, String> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err("golden key contains a non-hex digit".to_owned()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BoardConfig;

    #[test]
    fn canonical_key_matches_python_codec_for_simple_position() {
        let state = GameState::with_config(BoardConfig::DEFAULT);
        let (key, symmetry) = canonical_key(state);
        assert_eq!(symmetry, 0);
        assert_eq!(key.len(), 14);
        assert_eq!(
            key,
            decode_hex("0000000000000000000000008863").expect("expected key is hex")
        );
    }

    #[test]
    fn d4_round_trip_preserves_action() {
        let config = BoardConfig::DEFAULT;
        let action = Action::Relocate { from: 2, to: 35 };
        for symmetry in 0..8 {
            let transformed = transform_action(action, config.board_size, symmetry);
            let round_trip =
                transform_action(transformed, config.board_size, inverse_symmetry(symmetry));
            assert_eq!(round_trip, action, "symmetry {symmetry}");
        }
    }

    #[test]
    fn marker_bits_and_key_size_match_persisted_formats() {
        assert_eq!(key_bytes_for_board_size(5), Ok(8));
        assert_eq!(key_bytes_for_board_size(7), Ok(14));
    }

    #[test]
    fn canonical_key_round_trips_through_state_decoder() {
        let mut state = GameState::with_config(BoardConfig::DEFAULT);
        state.light = (1_u64 << 3) | (1_u64 << 17) | (1_u64 << 31);
        state.dark = (1_u64 << 9) | (1_u64 << 40);
        state.reserve = [11, 12];
        state.turn = Player::Dark;
        state.forbidden = 1_u64 << 24;
        state.last_relocated_to = [Some(17), None];
        let key = canonical_position_key(state);
        let decoded = decode_canonical_position_key(&key, 7, 14).expect("decode canonical key");
        assert_eq!(canonical_position_key(decoded), key);
        let mut expected_reserves = state.reserve;
        expected_reserves.sort_unstable();
        let mut decoded_reserves = decoded.reserve;
        decoded_reserves.sort_unstable();
        assert_eq!(decoded_reserves, expected_reserves);
        let mut expected_counts = [state.light.count_ones(), state.dark.count_ones()];
        expected_counts.sort_unstable();
        let mut decoded_counts = [decoded.light.count_ones(), decoded.dark.count_ones()];
        decoded_counts.sort_unstable();
        assert_eq!(decoded_counts, expected_counts);
        assert_eq!(decoded.forbidden.count_ones(), state.forbidden.count_ones());
    }

    #[test]
    fn canonical_key_decoder_rejects_reserved_bits() {
        let mut key = canonical_position_key(GameState::with_config(BoardConfig::DEFAULT));
        *key.last_mut().expect("key has bytes") |= 0x80;
        assert!(decode_canonical_position_key(&key, 7, 14).is_err());
    }

    #[test]
    fn compact_action_book_v2_preserves_sparse_values_and_row_proof() {
        let mut source = Vec::new();
        source.extend_from_slice(ACTION_BOOK_V2_MAGIC);
        source.extend_from_slice(&[7, 14, 14, 0]);
        source.extend_from_slice(&1_u32.to_le_bytes());
        source.extend_from_slice(&[0; 14]);
        source.extend_from_slice(&[0, WIN]);
        source.extend_from_slice(&1_u16.to_le_bytes());
        source.extend_from_slice(&1_u16.to_le_bytes());
        source.extend_from_slice(&0_u16.to_le_bytes());
        source.push(WIN);
        source.extend_from_slice(&1_u16.to_le_bytes());

        let book = GoldenActionBook::from_bytes(&source, 7).expect("decode PGACT02");
        let key = vec![0; 14];
        assert_eq!(book.rows(), 1);
        assert_eq!(
            book.rows.get(&key),
            Some(&GoldenRowValue {
                outcome: GoldenOutcome::Win,
                distance: Some(1),
                optimal_actions_complete: false,
            })
        );
        assert_eq!(
            book.action_values.get(&key),
            Some(&vec![GoldenActionValue {
                action: Action::Place { to: 0 },
                outcome: Some(GoldenOutcome::Win),
                distance: Some(1),
            }])
        );
    }

    #[test]
    fn legacy_compact_action_book_v1_remains_readable() {
        let mut source = Vec::new();
        source.extend_from_slice(ACTION_BOOK_V1_MAGIC);
        source.extend_from_slice(&[7, 14, 14, 0]);
        source.extend_from_slice(&1_u32.to_le_bytes());
        source.extend_from_slice(&[0; 14]);
        source.extend_from_slice(&1_u16.to_le_bytes());
        source.extend_from_slice(&0_u16.to_le_bytes());

        let book = GoldenActionBook::from_bytes(&source, 7).expect("decode PGACT01");
        let key = vec![0; 14];
        assert_eq!(book.rows(), 1);
        assert_eq!(
            book.action_values[&key][0].outcome,
            Some(GoldenOutcome::Win)
        );
    }

    #[test]
    fn promoted_ring1_action_is_legal_after_symmetry_inversion() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../research/20260830-endgame-retrograde-frontier/workspace/ring-01-candidates.jsonl");
        let table_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/golden/tables/fresh-frontier-wdl-v1/7x7-r14/shard-00.bin");
        let sidecar_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/golden/sidecars/fresh-frontier-wdl-v1/7x7-r14/ring-01.bin");
        if !(root.exists() && table_path.exists() && sidecar_path.exists()) {
            return;
        }
        let raw: Value = serde_json::from_str(
            &std::fs::read_to_string(root)
                .expect("read Ring-1 candidate pilot")
                .lines()
                .next()
                .expect("candidate pilot has one row"),
        )
        .expect("candidate pilot row is JSON");
        let position = raw.get("position").expect("candidate has position");
        let config = crate::BoardConfig::DEFAULT;
        let state = GameState {
            config,
            light: position["light"].as_u64().expect("light mask"),
            dark: position["dark"].as_u64().expect("dark mask"),
            reserve: [
                position["reserve"][0].as_u64().expect("light reserve") as u8,
                position["reserve"][1].as_u64().expect("dark reserve") as u8,
            ],
            turn: if position["turn"] == "light" {
                Player::Light
            } else {
                Player::Dark
            },
            forbidden: position["forbidden"].as_u64().expect("forbidden mask"),
            last_relocated_to: [
                position["lastRelocatedTo"][0]
                    .as_u64()
                    .map(|value| value as u8),
                position["lastRelocatedTo"][1]
                    .as_u64()
                    .map(|value| value as u8),
            ],
            last_capture: 0,
            last_player: None,
            winner: None,
            ply: position["ply"].as_u64().expect("ply") as u16,
        };
        let table = FlatGoldenTable::open(&table_path, 7, 14).expect("open Ring-1 table");
        let actions = GoldenActionBook::load(&sidecar_path, 7).expect("open Ring-1 sidecar");
        assert_eq!(table.lookup(state), Some(GoldenOutcome::Win));
        assert_eq!(
            actions.row_value(state),
            Some(GoldenRowValue {
                outcome: GoldenOutcome::Win,
                distance: Some(1),
                optimal_actions_complete: false,
            })
        );
        assert!(actions.action_values(state).is_some_and(|values| {
            values
                .iter()
                .any(|value| value.outcome == Some(GoldenOutcome::Win) && value.distance == Some(1))
        }));
        let action = actions.proven_action(state).expect("recover Ring-1 action");
        assert!(
            state.legal_actions().contains(&action),
            "returned {action:?}"
        );
        assert_eq!(
            state.apply_legal(action).state.winner,
            Some(state.turn),
            "returned {action:?}"
        );
        let lookup = GoldenLookup::open(&table_path, Some(&sidecar_path), 7, 14)
            .expect("open combined Ring-1 lookup");
        let result = crate::search::search_best_action_with_golden(
            state,
            crate::search::SearchConfig::default(),
            &lookup,
        );
        assert_eq!(result.action, Some(action));
        assert_eq!(result.nodes, 0);
        assert_eq!(result.table_hits, 1);

        let memory_lookup = MemoryGoldenLookup::open_bytes(
            &std::fs::read(&table_path).expect("read Ring-1 table"),
            Some(&std::fs::read(&sidecar_path).expect("read Ring-1 sidecar")),
            7,
            14,
        )
        .expect("open in-memory Ring-1 lookup");
        assert_eq!(memory_lookup.lookup(state), Some(GoldenOutcome::Win));
        assert_eq!(memory_lookup.proven_action(state), Some(action));
        let (memory_result, memory_outcome, exact_action) =
            crate::search::search_best_action_with_golden_bytes(
                state,
                crate::search::SearchConfig::default(),
                &std::fs::read(&table_path).expect("read Ring-1 table bytes"),
                Some(&std::fs::read(&sidecar_path).expect("read Ring-1 sidecar bytes")),
            )
            .expect("search with in-memory Ring-1 lookup");
        assert_eq!(memory_outcome, Some(GoldenOutcome::Win));
        assert!(exact_action);
        assert_eq!(memory_result.action, Some(action));
    }

    #[test]
    fn promoted_ring2_singleton_preserves_exact_value_and_actions() {
        let table_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/golden/tables/fresh-frontier-wdl-v2/7x7-r14/shard-00.bin");
        let sidecar_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/golden/sidecars/fresh-frontier-wdl-v2/7x7-r14/ring-02.bin");
        if !(table_path.exists() && sidecar_path.exists()) {
            return;
        }
        let key = decode_hex("00000000004556969aa5aa5b6263").expect("Ring-2 root key");
        let state = decode_canonical_position_key(&key, 7, 14).expect("decode Ring-2 root");
        let table = FlatGoldenTable::open(&table_path, 7, 14).expect("open Ring-2 table");
        let actions = GoldenActionBook::load(&sidecar_path, 7).expect("open Ring-2 sidecar");

        assert_eq!(table.lookup(state), Some(GoldenOutcome::Loss));
        assert_eq!(
            actions.row_value(state),
            Some(GoldenRowValue {
                outcome: GoldenOutcome::Loss,
                distance: Some(2),
                optimal_actions_complete: true,
            })
        );
        let action_values = actions
            .action_values(state)
            .expect("Ring-2 row has action values");
        assert_eq!(action_values.len(), state.legal_actions().len());
        assert!(action_values.iter().all(|value| value.outcome.is_some()));
    }

    #[test]
    fn rust_promoted_ring2_pair_preserves_exact_values_and_actions() {
        let table_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/golden/tables/fresh-frontier-wdl-v3/7x7-r14/shard-00.bin");
        let sidecar_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/golden/sidecars/fresh-frontier-wdl-v3/7x7-r14/ring-02.bin");
        if !(table_path.exists() && sidecar_path.exists()) {
            return;
        }
        let table = FlatGoldenTable::open(&table_path, 7, 14).expect("open Rust Ring-2 table");
        let actions = GoldenActionBook::load(&sidecar_path, 7).expect("open Rust Ring-2 sidecar");
        for key_hex in [
            "00000000004556969aa5aa5b6263",
            "0000000000545556aa95eaaaca62",
        ] {
            let key = decode_hex(key_hex).expect("Rust Ring-2 root key");
            let state =
                decode_canonical_position_key(&key, 7, 14).expect("decode Rust Ring-2 root");
            assert_eq!(table.lookup(state), Some(GoldenOutcome::Loss));
            assert_eq!(
                actions.row_value(state),
                Some(GoldenRowValue {
                    outcome: GoldenOutcome::Loss,
                    distance: Some(2),
                    optimal_actions_complete: true,
                })
            );
            let action_values = actions
                .action_values(state)
                .expect("Rust Ring-2 row has action values");
            assert_eq!(action_values.len(), state.legal_actions().len());
            assert!(action_values.iter().all(|value| value.outcome.is_some()));
        }
    }

    #[test]
    fn rust_promoted_ring2_three_root_lookup_is_readable() {
        let table_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/golden/tables/fresh-frontier-wdl-v4/7x7-r14/shard-00.bin");
        let sidecar_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/golden/sidecars/fresh-frontier-wdl-v4/7x7-r14/ring-02.bin");
        if !(table_path.exists() && sidecar_path.exists()) {
            return;
        }
        let lookup = GoldenLookup::open(&table_path, Some(&sidecar_path), 7, 14)
            .expect("open Rust three-root lookup");
        for key_hex in [
            "00000000004556969aa5aa5b6263",
            "0000000000545556aa95eaaaca62",
            "0000000000545a556a95aaabda62",
        ] {
            let key = decode_hex(key_hex).expect("three-root key");
            let state = decode_canonical_position_key(&key, 7, 14).expect("decode three-root");
            assert_eq!(lookup.lookup(state), Some(GoldenOutcome::Loss));
            assert_eq!(
                lookup
                    .actions
                    .as_ref()
                    .expect("three-root sidecar")
                    .row_value(state)
                    .expect("three-root action row")
                    .optimal_actions_complete,
                true
            );
            assert_eq!(
                lookup
                    .actions
                    .as_ref()
                    .expect("three-root sidecar")
                    .action_values(state)
                    .expect("three-root action values")
                    .len(),
                state.legal_actions().len()
            );
        }
    }

    #[test]
    fn layered_lookup_keeps_ring1_priority_and_falls_through_to_ring2() {
        let ring1_table = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/golden/tables/fresh-frontier-wdl-v1/7x7-r14/shard-00.bin");
        let ring1_sidecar = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/golden/sidecars/fresh-frontier-wdl-v1/7x7-r14/ring-01.bin");
        let ring2_table = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/golden/tables/fresh-frontier-wdl-v4/7x7-r14/shard-00.bin");
        let ring2_sidecar = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/golden/sidecars/fresh-frontier-wdl-v4/7x7-r14/ring-02.bin");
        if !(ring1_table.exists()
            && ring1_sidecar.exists()
            && ring2_table.exists()
            && ring2_sidecar.exists())
        {
            return;
        }
        let paths = vec![
            (ring1_table.clone(), Some(ring1_sidecar.clone())),
            (ring2_table.clone(), Some(ring2_sidecar.clone())),
        ];
        let layered = GoldenLookupLayers::open(&paths, 7, 14).expect("open layered lookup");
        assert_eq!(layered.layer_count(), 2);
        assert_eq!(layered.rows(), 35_564);

        let ring2_key = decode_hex("0000000000545a556a95aaabda62").expect("Ring-2 key");
        let ring2_state =
            decode_canonical_position_key(&ring2_key, 7, 14).expect("decode Ring-2 key");
        assert_eq!(layered.lookup(ring2_state), Some(GoldenOutcome::Loss));

        let raw: Value = serde_json::from_str(
            &std::fs::read_to_string(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../research/20260830-endgame-retrograde-frontier/workspace/ring-01-candidates.jsonl"),
            )
            .expect("read Ring-1 candidates")
            .lines()
            .next()
            .expect("Ring-1 candidate exists"),
        )
        .expect("Ring-1 candidate is JSON");
        let position = raw.get("position").expect("Ring-1 candidate position");
        let ring1_state = GameState {
            config: BoardConfig::DEFAULT,
            light: position["light"].as_u64().expect("light mask"),
            dark: position["dark"].as_u64().expect("dark mask"),
            reserve: [
                position["reserve"][0].as_u64().expect("light reserve") as u8,
                position["reserve"][1].as_u64().expect("dark reserve") as u8,
            ],
            turn: if position["turn"] == "light" {
                Player::Light
            } else {
                Player::Dark
            },
            forbidden: position["forbidden"].as_u64().expect("forbidden mask"),
            last_relocated_to: [
                position["lastRelocatedTo"][0]
                    .as_u64()
                    .map(|value| value as u8),
                position["lastRelocatedTo"][1]
                    .as_u64()
                    .map(|value| value as u8),
            ],
            last_capture: 0,
            last_player: None,
            winner: None,
            ply: position["ply"].as_u64().expect("ply") as u16,
        };
        assert_eq!(layered.lookup(ring1_state), Some(GoldenOutcome::Win));
        let exact_action = layered
            .proven_action(ring1_state)
            .expect("Ring-1 exact action");
        let result = crate::search::search_best_action_with_golden_layers(
            ring1_state,
            crate::search::SearchConfig::default(),
            &layered,
        );
        assert_eq!(result.action, Some(exact_action));
        assert_eq!(result.nodes, 0);

        let ring1_table_bytes = std::fs::read(&ring1_table).expect("read Ring-1 table");
        let ring1_sidecar_bytes = std::fs::read(&ring1_sidecar).expect("read Ring-1 sidecar");
        let ring2_table_bytes = std::fs::read(&ring2_table).expect("read Ring-2 table");
        let ring2_sidecar_bytes = std::fs::read(&ring2_sidecar).expect("read Ring-2 sidecar");
        let memory = MemoryGoldenLookupLayers::open_bytes(
            &[
                (&ring1_table_bytes, Some(ring1_sidecar_bytes.as_slice())),
                (&ring2_table_bytes, Some(ring2_sidecar_bytes.as_slice())),
            ],
            7,
            14,
        )
        .expect("open memory layered lookup");
        assert_eq!(memory.layer_count(), 2);
        assert_eq!(memory.lookup(ring1_state), Some(GoldenOutcome::Win));
        assert_eq!(memory.lookup(ring2_state), Some(GoldenOutcome::Loss));
        assert_eq!(memory.proven_action(ring1_state), Some(exact_action));
    }
}
