//! Validate and promote exact replay-ring tablebase rows.
//!
//! This is the authoritative Rust-side promotion gate. It consumes compact
//! value shards plus the replay-witnessed JSONL graph, replays every legal
//! edge, checks D4 invariance and exact minimax/distance consistency, rejects
//! contradictions with existing gold, and writes compact durable artifacts.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use pathagon_engine::corpus::decode_action;
use pathagon_engine::golden::{
    canonical_position, canonical_position_key, transform_action, transform_position,
    FlatGoldenTable, GoldenOutcome,
};
use pathagon_engine::ground_truth::GroundTruthOutcome;
use pathagon_engine::tablebase::{read_value_shards, RetrogradeValue};
use pathagon_engine::{Action, BoardConfig, GameState, Player};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

const BOARD_SIZE: u8 = 7;
const RESERVE_PER_PLAYER: u8 = 14;
const ACTION_BOOK_MAGIC: &[u8; 8] = b"PGACT02\0";
const ACTION_BOOK_NONE_DISTANCE: u16 = u16::MAX;

#[derive(Clone, Debug)]
struct PromotedRow {
    value: RetrogradeValue,
    actions: Vec<(Action, RetrogradeValue)>,
}

#[derive(Default)]
struct Stats {
    graph_records: usize,
    ring_records: usize,
    exact_value_records: usize,
    closed_ring_rows: usize,
    promoted_rows: usize,
    unknown_ring_rows: usize,
    contradictions: usize,
    invalid_records: usize,
    symmetry_checks: usize,
}

fn main() {
    let args = parse_args();
    let graph_path = required(&args, "graph");
    let shards_path = required(&args, "shards");
    let existing_table_path = required(&args, "existing-table");
    let report_path = required(&args, "report");
    let ring = number(&args, "ring", 2);
    if ring < 2 {
        fail("--ring must be at least 2");
    }

    let mut values = read_value_shards(&shards_path)
        .unwrap_or_else(|error| fail(&format!("cannot read value shards: {error}")));
    let mut shard_count = read_shard_count(&shards_path)
        .unwrap_or_else(|error| fail(&format!("cannot read shard manifest: {error}")));
    let extra_shards = args
        .get("extra-shards")
        .map(|paths| {
            paths
                .split(',')
                .filter(|path| !path.is_empty())
                .map(PathBuf::from)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    for extra_shards in &extra_shards {
        let extra_values = read_value_shards(&extra_shards)
            .unwrap_or_else(|error| fail(&format!("cannot read extra value shards: {error}")));
        for (key, value) in extra_values {
            if let Some(previous) = values.insert(key.clone(), value) {
                if previous != value {
                    fail(&format!("contradictory inner value for key {key}"));
                }
            }
        }
        shard_count += read_shard_count(extra_shards)
            .unwrap_or_else(|error| fail(&format!("cannot read extra shard manifest: {error}")));
    }
    let existing = FlatGoldenTable::open(&existing_table_path, BOARD_SIZE, RESERVE_PER_PLAYER)
        .unwrap_or_else(|error| fail(&format!("cannot open existing golden table: {error}")));

    let mut promoted = BTreeMap::<String, PromotedRow>::new();
    let mut stats = Stats { ..Stats::default() };
    let source = File::open(&graph_path)
        .unwrap_or_else(|error| fail(&format!("cannot open graph: {error}")));
    for (line_number, line) in BufReader::new(source).lines().enumerate() {
        let line_number = line_number + 1;
        let line = line.unwrap_or_else(|error| {
            fail(&format!(
                "cannot read {}:{line_number}: {error}",
                graph_path.display()
            ))
        });
        if line.trim().is_empty() {
            continue;
        }
        stats.graph_records += 1;
        let record: Value = serde_json::from_str(&line).unwrap_or_else(|error| {
            fail(&format!(
                "invalid JSON at {}:{line_number}: {error}",
                graph_path.display()
            ))
        });
        if record.get("ring").and_then(Value::as_u64) != Some(ring) {
            continue;
        }
        stats.ring_records += 1;
        let key = record
            .get("key")
            .and_then(Value::as_str)
            .unwrap_or_else(|| {
                fail(&format!(
                    "{}:{line_number}: Ring row has no key",
                    graph_path.display()
                ))
            })
            .to_owned();
        let Some(value) = values.get(&key).copied() else {
            stats.unknown_ring_rows += 1;
            continue;
        };
        stats.exact_value_records += 1;
        validate_ring_record(
            &record,
            &key,
            value,
            ring,
            &values,
            &existing,
            &mut promoted,
            &mut stats,
        )
        .unwrap_or_else(|error| fail(&format!("{}:{line_number}: {error}", graph_path.display())));
    }

    let table_path = args.get("table").map(PathBuf::from);
    let sidecar_path = args.get("sidecar").map(PathBuf::from);
    let manifest_path = args.get("manifest").map(PathBuf::from);
    if !promoted.is_empty()
        && (table_path.is_none() || sidecar_path.is_none() || manifest_path.is_none())
    {
        fail("--table, --sidecar, and --manifest are required when rows are promotable");
    }
    if let (Some(table_path), Some(sidecar_path), Some(manifest_path)) =
        (table_path, sidecar_path, manifest_path)
    {
        let table_family = args
            .get("table-family")
            .cloned()
            .unwrap_or_else(|| format!("fresh-frontier-wdl-v{ring}"));
        let table_rows = write_table(&table_path, &promoted)
            .unwrap_or_else(|error| fail(&format!("cannot write promoted table: {error}")));
        write_sidecar(&sidecar_path, &promoted)
            .unwrap_or_else(|error| fail(&format!("cannot write promoted sidecar: {error}")));
        let manifest = json!({
            "schemaVersion": 1,
            "tableFamily": table_family,
            "rulesVersion": "pathagon-rules-v1",
            "ring": ring,
            "provenance": {
                "solverVersion": "pathagon-endgame-tablebase-v1",
                "rulesVersion": "pathagon-rules-v1",
                "proofLineage": "complete-forward-legal-edges-plus-exact-inner-seeds"
            },
            "rows": table_rows,
            "shard": {
                "path": table_path,
                "sha256": sha256_file(&table_path).unwrap_or_else(|error| fail(&format!("cannot hash promoted table: {error}")))
            },
            "sidecar": {
                "path": sidecar_path,
                "sha256": sha256_file(&sidecar_path).unwrap_or_else(|error| fail(&format!("cannot hash promoted sidecar: {error}")))
            },
            "source": graph_path,
            "promotion": format!("closed-ring-{ring}-only")
        });
        write_json(&manifest_path, &manifest)
            .unwrap_or_else(|error| fail(&format!("cannot write promotion manifest: {error}")));
        stats.promoted_rows = table_rows;
    }

    let promotion_decision = if stats.promoted_rows > 0 {
        "promote"
    } else {
        "retain-unknown-and-do-not-promote"
    };
    let report = json!({
        "schemaVersion": 1,
        "experiment": format!("ring-{ring}-golden-promotion"),
        "graph": graph_path,
        "innerShards": shards_path,
        "extraInnerShards": extra_shards,
        "stats": {
            "graphRecords": stats.graph_records,
            "ringRecords": stats.ring_records,
            "exactValueRecords": stats.exact_value_records,
            "closedRingRows": stats.closed_ring_rows,
            "promotedRows": stats.promoted_rows,
            "unknownRingRows": stats.unknown_ring_rows,
            "contradictions": stats.contradictions,
            "invalidRecords": stats.invalid_records,
            "symmetryChecks": stats.symmetry_checks,
            "seededInnerRows": values.len(),
            "valueShards": shard_count
        },
        "gates": {
            "inventoryAndSeededValidation": "pass",
            "forwardTransitionWitness": "pass",
            "symmetryInvariant": if stats.symmetry_checks > 0 { "pass" } else { "not-run" },
            "contradictoryExistingGold": if stats.contradictions == 0 { "pass" } else { "fail" },
            "completeActionSets": if stats.closed_ring_rows > 0 { "pass" } else { "not-run" },
            "promotionDecision": promotion_decision
        },
        "shardSamples": {}
    });
    write_json(&report_path, &report)
        .unwrap_or_else(|error| fail(&format!("cannot write promotion report: {error}")));
    println!("{report}");
}

fn validate_ring_record(
    record: &Value,
    key: &str,
    value: RetrogradeValue,
    ring: u64,
    values: &BTreeMap<String, RetrogradeValue>,
    existing: &FlatGoldenTable,
    promoted: &mut BTreeMap<String, PromotedRow>,
    stats: &mut Stats,
) -> Result<(), String> {
    if record.get("complete").and_then(Value::as_bool) != Some(true)
        || record
            .get("proof")
            .and_then(|proof| proof.get("lineage"))
            .and_then(Value::as_str)
            != Some("full-corpus-replay-plus-verified-terminal-suffix")
    {
        return Err(format!("exact Ring-{ring} row lacks complete replay proof"));
    }
    let state = state_from_json(
        record
            .get("position")
            .ok_or_else(|| "Ring row has no position".to_owned())?,
    )?;
    if hex_key(&canonical_position_key(state)) != key {
        return Err("position key is not canonical".to_owned());
    }
    let (_, canonical_symmetry) = canonical_position(state);

    let legal = state.legal_actions();
    let action_rows = record
        .get("actions")
        .and_then(Value::as_array)
        .ok_or_else(|| "Ring row has no action edges".to_owned())?;
    let mut edges = BTreeMap::<u16, (Action, String)>::new();
    for edge in action_rows {
        let token = edge
            .get("action")
            .and_then(Value::as_str)
            .ok_or_else(|| "action edge has no token".to_owned())?;
        let action = decode_action(token)?;
        let child = edge
            .get("child")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("action {token} has no child"))?
            .to_owned();
        if edges.insert(action_code(action), (action, child)).is_some() {
            return Err(format!("duplicate action edge {token}"));
        }
    }
    let legal_codes = legal
        .iter()
        .map(|action| action_code(*action))
        .collect::<BTreeSet<_>>();
    if edges.keys().copied().collect::<BTreeSet<_>>() != legal_codes {
        return Err("edge graph does not cover the legal action set".to_owned());
    }
    let record_children = record
        .get("children")
        .and_then(Value::as_array)
        .ok_or_else(|| "Ring row has no children".to_owned())?
        .iter()
        .map(|child| {
            child
                .as_str()
                .map(ToOwned::to_owned)
                .ok_or_else(|| "child key is not a string".to_owned())
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    let edge_children = edges
        .values()
        .map(|(_, child)| child.clone())
        .collect::<BTreeSet<_>>();
    if edge_children != record_children {
        return Err("edge graph children do not match the record".to_owned());
    }

    let mut action_values = Vec::with_capacity(edges.len());
    for (code, (action, child_key)) in edges {
        let actual_child = hex_key(&canonical_position_key(state.apply_legal(action).state));
        if actual_child != child_key {
            return Err(format!(
                "action code {code} does not replay to its child key"
            ));
        }
        let child = values
            .get(&child_key)
            .ok_or_else(|| format!("child {child_key} is not an exact inner value"))?;
        action_values.push((
            transform_action(action, state.config.board_size, canonical_symmetry),
            RetrogradeValue {
                outcome: child.outcome.negate(),
                distance: child.distance.map(|distance| distance.saturating_add(1)),
            },
        ));
    }
    stats.closed_ring_rows += 1;
    let action_outcomes = action_values
        .iter()
        .map(|(_, value)| value.outcome)
        .collect::<Vec<_>>();
    let expected_outcome = if action_outcomes
        .iter()
        .any(|outcome| *outcome == GroundTruthOutcome::Win)
    {
        GroundTruthOutcome::Win
    } else if action_outcomes
        .iter()
        .all(|outcome| *outcome == GroundTruthOutcome::Loss)
    {
        GroundTruthOutcome::Loss
    } else {
        GroundTruthOutcome::Draw
    };
    let expected_distance = match expected_outcome {
        GroundTruthOutcome::Win => Some(
            action_values
                .iter()
                .filter(|(_, value)| value.outcome == GroundTruthOutcome::Win)
                .filter_map(|(_, value)| value.distance)
                .min()
                .ok_or_else(|| "winning row lacks a proven distance".to_owned())?,
        ),
        GroundTruthOutcome::Loss => Some(
            action_values
                .iter()
                .filter(|(_, value)| value.outcome == GroundTruthOutcome::Loss)
                .filter_map(|(_, value)| value.distance)
                .max()
                .ok_or_else(|| "losing row lacks a proven distance".to_owned())?,
        ),
        GroundTruthOutcome::Draw | GroundTruthOutcome::Unknown => None,
    };
    if value.outcome != expected_outcome || value.distance != expected_distance {
        return Err("solved Ring value disagrees with child minimax".to_owned());
    }
    if existing.lookup(state) == Some(to_golden(value.outcome)) {
        // Existing identical gold is allowed by the monotonic table contract.
    } else if existing.lookup(state).is_some() {
        stats.contradictions += 1;
        return Err("position conflicts with existing golden value".to_owned());
    }
    for symmetry in 0..8 {
        if hex_key(&canonical_position_key(transform_position(state, symmetry))) != key {
            return Err(format!("symmetry {symmetry} changed canonicalization"));
        }
        stats.symmetry_checks += 1;
    }
    let candidate = PromotedRow {
        value,
        actions: action_values,
    };
    if let Some(previous) = promoted.get(key) {
        if previous.value != candidate.value {
            stats.contradictions += 1;
            return Err("contradictory promoted value".to_owned());
        }
    } else {
        promoted.insert(key.to_owned(), candidate);
    }
    Ok(())
}

fn state_from_json(raw: &Value) -> Result<GameState, String> {
    let object = raw
        .as_object()
        .ok_or_else(|| "position must be an object".to_owned())?;
    let board_size = number_field(object, "boardSize")? as u8;
    let reserve_per_player = number_field(object, "reservePerPlayer")? as u8;
    let config = BoardConfig::new(board_size, reserve_per_player)?;
    let reserve = array_field(object, "reserve")?;
    if reserve.len() != 2 {
        return Err("position reserve must have two entries".to_owned());
    }
    let markers = array_field(object, "lastRelocatedTo")?;
    if markers.len() != 2 {
        return Err("position relocation markers must have two entries".to_owned());
    }
    let turn = match string_field(object, "turn")? {
        "light" => Player::Light,
        "dark" => Player::Dark,
        _ => return Err("position has an invalid turn".to_owned()),
    };
    let state = GameState {
        config,
        light: number_field(object, "light")?,
        dark: number_field(object, "dark")?,
        reserve: [
            u8::try_from(
                reserve[0]
                    .as_u64()
                    .ok_or_else(|| "invalid light reserve".to_owned())?,
            )
            .map_err(|_| "light reserve exceeds u8".to_owned())?,
            u8::try_from(
                reserve[1]
                    .as_u64()
                    .ok_or_else(|| "invalid dark reserve".to_owned())?,
            )
            .map_err(|_| "dark reserve exceeds u8".to_owned())?,
        ],
        turn,
        forbidden: number_field(object, "forbidden")?,
        last_relocated_to: [optional_square(&markers[0])?, optional_square(&markers[1])?],
        last_capture: 0,
        last_player: None,
        winner: None,
        ply: object.get("ply").and_then(Value::as_u64).unwrap_or(0) as u16,
    };
    validate_inventory(&state)?;
    Ok(state)
}

fn validate_inventory(state: &GameState) -> Result<(), String> {
    let cells = usize::from(state.config.cells());
    let board_mask = if cells == u64::BITS as usize {
        u64::MAX
    } else {
        (1_u64 << cells) - 1
    };
    if state.light & !board_mask != 0
        || state.dark & !board_mask != 0
        || state.forbidden & !board_mask != 0
    {
        return Err("position has bits outside the board".to_owned());
    }
    if state.light & state.dark != 0 {
        return Err("position light/dark masks overlap".to_owned());
    }
    if state.forbidden & (state.light | state.dark) != 0 {
        return Err("position forbidden mask overlaps a piece".to_owned());
    }
    if state
        .last_relocated_to
        .into_iter()
        .flatten()
        .any(|square| usize::from(square) >= cells)
    {
        return Err("position relocation marker is outside the board".to_owned());
    }
    let light_total = state.light.count_ones() as u16 + u16::from(state.reserve[0]);
    let dark_total = state.dark.count_ones() as u16 + u16::from(state.reserve[1]);
    let expected = u16::from(state.config.reserve_per_player);
    if light_total != expected || dark_total != expected {
        return Err("position inventory does not match reserve-per-player".to_owned());
    }
    Ok(())
}

fn array_field<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Result<&'a Vec<Value>, String> {
    object
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("position field {key} must be an array"))
}

fn number_field(object: &serde_json::Map<String, Value>, key: &str) -> Result<u64, String> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("position field {key} must be an integer"))
}

fn string_field<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Result<&'a str, String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("position field {key} must be a string"))
}

fn optional_square(value: &Value) -> Result<Option<u8>, String> {
    value
        .as_u64()
        .map(|value| u8::try_from(value).map_err(|_| "relocation marker exceeds u8".to_owned()))
        .transpose()
}

fn action_code(action: Action) -> u16 {
    match action {
        Action::Place { to } => u16::from(to),
        Action::Relocate { from, to } => 49 + u16::from(from) * 49 + u16::from(to),
    }
}

fn to_golden(outcome: GroundTruthOutcome) -> GoldenOutcome {
    match outcome {
        GroundTruthOutcome::Loss => GoldenOutcome::Loss,
        GroundTruthOutcome::Draw => GoldenOutcome::Draw,
        GroundTruthOutcome::Win => GoldenOutcome::Win,
        GroundTruthOutcome::Unknown => unreachable!("unknown values cannot be promoted"),
    }
}

fn write_table(path: &Path, rows: &BTreeMap<String, PromotedRow>) -> io::Result<usize> {
    create_parent(path)?;
    let mut writer = BufWriter::new(File::create(path)?);
    for (key, row) in rows {
        writer.write_all(&decode_hex(key).map_err(invalid_data)?)?;
        writer.write_all(&[to_golden(row.value.outcome).as_byte()])?;
    }
    writer.flush()?;
    Ok(rows.len())
}

fn write_sidecar(path: &Path, rows: &BTreeMap<String, PromotedRow>) -> io::Result<()> {
    create_parent(path)?;
    let mut writer = BufWriter::new(File::create(path)?);
    writer.write_all(ACTION_BOOK_MAGIC)?;
    writer.write_all(&[BOARD_SIZE, RESERVE_PER_PLAYER, 14, 0])?;
    writer.write_all(&(rows.len() as u32).to_le_bytes())?;
    for (key, row) in rows {
        writer.write_all(&decode_hex(key).map_err(invalid_data)?)?;
        writer.write_all(&[1, to_golden(row.value.outcome).as_byte()])?;
        writer.write_all(&distance_bytes(row.value.distance).to_le_bytes())?;
        let mut actions = row.actions.clone();
        actions.sort_by_key(|(action, _)| action_code(*action));
        writer.write_all(&(actions.len() as u16).to_le_bytes())?;
        for (action, value) in actions {
            writer.write_all(&action_code(action).to_le_bytes())?;
            writer.write_all(&[to_golden(value.outcome).as_byte()])?;
            writer.write_all(&distance_bytes(value.distance).to_le_bytes())?;
        }
    }
    writer.flush()
}

fn distance_bytes(distance: Option<u16>) -> u16 {
    distance.unwrap_or(ACTION_BOOK_NONE_DISTANCE)
}

fn sha256_file(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    io::copy(&mut file, &mut DigestWriter(&mut digest))?;
    Ok(hex_string(&digest.finalize()))
}

struct DigestWriter<'a>(&'a mut Sha256);

impl Write for DigestWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn write_json(path: &Path, value: &Value) -> io::Result<()> {
    create_parent(path)?;
    fs::write(
        path,
        serde_json::to_vec_pretty(value)
            .expect("promotion JSON is serializable")
            .into_iter()
            .chain([b'\n'])
            .collect::<Vec<_>>(),
    )
}

fn read_shard_count(directory: &Path) -> io::Result<usize> {
    let manifest: Value =
        serde_json::from_str(&fs::read_to_string(directory.join("manifest.json"))?)
            .map_err(|error| invalid_data(error.to_string()))?;
    manifest
        .get("shardCount")
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .ok_or_else(|| invalid_data("shard manifest has no shardCount"))
}

fn decode_hex(value: &str) -> Result<Vec<u8>, String> {
    if value.len() % 2 != 0 {
        return Err("key has an odd number of hex digits".to_owned());
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|chunk| Ok((hex_digit(chunk[0])? << 4) | hex_digit(chunk[1])?))
        .collect()
}

fn hex_digit(value: u8) -> Result<u8, String> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err("key contains a non-hex digit".to_owned()),
    }
}

fn hex_key(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(char::from(b"0123456789abcdef"[(byte >> 4) as usize]));
        output.push(char::from(b"0123456789abcdef"[(byte & 0x0f) as usize]));
    }
    output
}

fn hex_string(bytes: &[u8]) -> String {
    hex_key(bytes)
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn create_parent(path: &Path) -> io::Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn parse_args() -> HashMap<String, String> {
    let mut values = HashMap::new();
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        let key = argument
            .strip_prefix("--")
            .unwrap_or_else(|| fail(&format!("unexpected argument {argument}")));
        let value = args
            .next()
            .unwrap_or_else(|| fail(&format!("missing value for --{key}")));
        if value.starts_with("--") {
            fail(&format!("missing value for --{key}"));
        }
        values.insert(key.to_owned(), value);
    }
    values
}

fn required(args: &HashMap<String, String>, key: &str) -> PathBuf {
    args.get(key)
        .map(PathBuf::from)
        .unwrap_or_else(|| fail(&format!("--{key} <path> is required")))
}

fn number(args: &HashMap<String, String>, key: &str, default: u64) -> u64 {
    args.get(key)
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| fail(&format!("--{key} must be an integer")))
        })
        .unwrap_or(default)
}

fn fail(message: &str) -> ! {
    eprintln!("pathagon-endgame-promote: {message}");
    std::process::exit(2);
}

#[cfg(test)]
mod tests {
    use super::*;
    use pathagon_engine::corpus::encode_action;

    #[test]
    fn action_codes_match_the_7x7_corpus_numbering() {
        assert_eq!(action_code(Action::Place { to: 48 }), 48);
        assert_eq!(action_code(Action::Relocate { from: 0, to: 0 }), 49);
        assert_eq!(action_code(Action::Relocate { from: 1, to: 2 }), 100);
        assert_eq!(
            decode_action(&encode_action(Action::Relocate { from: 1, to: 2 })).unwrap(),
            Action::Relocate { from: 1, to: 2 }
        );
    }

    #[test]
    fn inventory_validation_rejects_overlap_and_accepts_fixed_totals() {
        let mut state = GameState::with_config(BoardConfig::DEFAULT);
        assert!(validate_inventory(&state).is_ok());
        state.light = 1;
        state.dark = 1;
        assert!(validate_inventory(&state).is_err());
    }
}
