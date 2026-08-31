//! Export exact layered gold rows as training evidence.
//!
//! Rust owns table decoding, canonical-key reconstruction, and action labels.
//! The resulting JSONL is intentionally an inspection/training interchange
//! format, not a second source of truth; the compact table and sidecar remain
//! authoritative.

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use pathagon_engine::golden::{
    decode_canonical_position_key, FlatGoldenTable, GoldenActionBook, GoldenActionValue,
    GoldenOutcome, GoldenRowValue,
};
use pathagon_engine::search::{evaluate, EvaluationWeights};
use pathagon_engine::Action;
use serde_json::{json, Value};

const BOARD_SIZE: u8 = 7;
const RESERVE_PER_PLAYER: u8 = 14;
const TRAINING_WEIGHTS: EvaluationWeights = EvaluationWeights {
    path: 241,
    material: 112,
    capture: 887,
    structure: 40,
    threat: 154,
    edge: 74,
};

#[derive(Default)]
struct Stats {
    input_rows: usize,
    rows: usize,
    wins: usize,
    draws: usize,
    losses: usize,
    duplicate_rows: usize,
}

fn main() {
    let args = parse_args();
    let layers = parse_layers(
        args.get("layers")
            .unwrap_or_else(|| fail("--layers table,sidecar;table,sidecar is required")),
    );
    let output = required(&args, "out");
    let report_path = args
        .get("report")
        .map(PathBuf::from)
        .unwrap_or_else(|| output.with_extension("meta.json"));
    let max_rows = number(&args, "max-rows", 0_usize);

    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| fail(&format!("cannot create output directory: {error}")));
    }
    if let Some(parent) = report_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| fail(&format!("cannot create report directory: {error}")));
    }

    let file = File::create(&output)
        .unwrap_or_else(|error| fail(&format!("cannot create output: {error}")));
    let mut writer = BufWriter::new(file);
    let mut seen = BTreeMap::<String, GoldenOutcome>::new();
    let mut stats = Stats::default();

    for (layer_index, (table_path, sidecar_path)) in layers.iter().enumerate() {
        let table = FlatGoldenTable::open(table_path, BOARD_SIZE, RESERVE_PER_PLAYER)
            .unwrap_or_else(|error| fail(&format!("cannot open table: {error}")));
        let sidecar_path = sidecar_path
            .as_ref()
            .unwrap_or_else(|| fail("each export layer requires a sidecar"));
        let book = GoldenActionBook::load(sidecar_path, BOARD_SIZE)
            .unwrap_or_else(|error| fail(&format!("cannot open sidecar: {error}")));

        for (key, row, values) in book.rows_with_actions() {
            stats.input_rows += 1;
            let key_hex = hex_key(&key);
            if let Some(previous) = seen.get(&key_hex) {
                if *previous != row.outcome {
                    fail(&format!("layered gold contradiction for key {key_hex}"));
                }
                stats.duplicate_rows += 1;
                continue;
            }
            let state = decode_canonical_position_key(&key, BOARD_SIZE, RESERVE_PER_PLAYER)
                .unwrap_or_else(|error| {
                    fail(&format!("cannot decode gold key {key_hex}: {error}"))
                });
            if table.lookup(state) != Some(row.outcome) {
                fail(&format!("sidecar/table outcome mismatch for key {key_hex}"));
            }
            let legal = state.legal_actions();
            if values.iter().any(|value| !legal.contains(&value.action)) {
                fail(&format!(
                    "sidecar action is not legal in canonical state {key_hex}"
                ));
            }
            let record = training_record(&key_hex, state, row, &values, layer_index);
            serde_json::to_writer(&mut writer, &record)
                .unwrap_or_else(|error| fail(&format!("cannot serialize training row: {error}")));
            writer
                .write_all(b"\n")
                .unwrap_or_else(|error| fail(&format!("cannot write training row: {error}")));
            seen.insert(key_hex, row.outcome);
            stats.rows += 1;
            match row.outcome {
                GoldenOutcome::Win => stats.wins += 1,
                GoldenOutcome::Draw => stats.draws += 1,
                GoldenOutcome::Loss => stats.losses += 1,
            }
            if max_rows > 0 && stats.rows >= max_rows {
                break;
            }
        }
        if max_rows > 0 && stats.rows >= max_rows {
            break;
        }
    }
    writer
        .flush()
        .unwrap_or_else(|error| fail(&format!("cannot flush training rows: {error}")));

    let report = json!({
        "schemaVersion": 1,
        "format": "gold-training-jsonl-v2",
        "out": output,
        "layers": layers,
        "stats": {
            "inputRows": stats.input_rows,
            "rows": stats.rows,
            "wins": stats.wins,
            "draws": stats.draws,
            "losses": stats.losses,
            "duplicateRowsSkipped": stats.duplicate_rows,
        },
        "authority": "compact-table-and-action-sidecar",
        "unknownPolicy": "explicit-null-action-outcome;absent-table-key-is-unknown",
        "proof": "Rust-decoded-canonical-key-and-sidecar-with-table-outcome-agreement",
        "actionFeatures": {"width": 16, "producer": "pathagon-engine-rs", "weights": TRAINING_WEIGHTS},
        "forcedBlockTargets": "Rust-precomputed;null-means-no-forced-block-eligibility",
    });
    fs::write(
        &report_path,
        serde_json::to_vec_pretty(&report).expect("serialize export report"),
    )
    .unwrap_or_else(|error| fail(&format!("cannot write export report: {error}")));
    println!("{report}");
}

fn training_record(
    key: &str,
    state: pathagon_engine::GameState,
    row: GoldenRowValue,
    values: &[GoldenActionValue],
    layer: usize,
) -> Value {
    let legal = state.legal_actions();
    let proven_actions = values
        .iter()
        .filter(|value| value.outcome == Some(GoldenOutcome::Win))
        .map(|value| value.action)
        .collect::<Vec<_>>();
    let urgency = urgency_targets(row.outcome, values);
    let features = legal
        .iter()
        .copied()
        .into_iter()
        .map(|action| action_features(state, action))
        .collect::<Vec<_>>();
    let forced_block_actions = forced_block_actions(state, &legal);
    json!({
        "schemaVersion": 1,
        "key": key,
        "sourceLayer": layer,
        "position": {
            "boardSize": state.config.board_size,
            "reservePerPlayer": state.config.reserve_per_player,
            "light": state.light,
            "dark": state.dark,
            "reserve": state.reserve,
            "turn": state.turn.as_str(),
            "forbidden": state.forbidden,
            "lastRelocatedTo": state.last_relocated_to,
            "ply": state.ply,
        },
        "outcome": row.outcome.as_str(),
        "distance": row.distance,
        "optimalActionsComplete": row.optimal_actions_complete,
        "provenActions": proven_actions.iter().map(|action| action_json(*action)).collect::<Vec<_>>(),
        "urgencyActions": urgency.as_ref().map(|(actions, _)| actions.iter().map(|action| action_json(*action)).collect::<Vec<_>>()),
        "urgencyDistance": urgency.as_ref().map(|(_, distance)| *distance),
        "features": features,
        "forcedBlockActions": forced_block_actions.as_ref().map(|actions| actions.iter().map(|action| action_json(*action)).collect::<Vec<_>>()),
        "actions": values.iter().map(action_value_json).collect::<Vec<_>>(),
    })
}

fn forced_block_actions(
    state: pathagon_engine::GameState,
    legal: &[Action],
) -> Option<Vec<Action>> {
    let safe = legal
        .iter()
        .copied()
        .filter(|action| {
            let child = state.apply_legal(*action).state;
            let opponent = child.turn;
            !child
                .legal_actions()
                .into_iter()
                .any(|reply| child.apply_legal(reply).state.winner == Some(opponent))
        })
        .collect::<Vec<_>>();
    (safe.len() < legal.len() && !safe.is_empty()).then_some(safe)
}

fn action_features(state: pathagon_engine::GameState, action: Action) -> Vec<f32> {
    let player = state.turn;
    let child = state.apply_legal(action).state;
    let size = f32::from(state.config.board_size.saturating_sub(1).max(1));
    let destination = action.destination();
    let to_row = destination / state.config.board_size;
    let to_column = destination % state.config.board_size;
    let (from_row, from_column) = match action {
        Action::Place { .. } => (0, 0),
        Action::Relocate { from, .. } => (
            from / state.config.board_size,
            from % state.config.board_size,
        ),
    };
    let before = evaluate(state, player, TRAINING_WEIGHTS);
    let after = evaluate(child, player, TRAINING_WEIGHTS);
    let delta = ((after - before) as f32 / 5_000.0).clamp(-1.0, 1.0);
    vec![
        f32::from(matches!(action, Action::Place { .. })),
        f32::from(matches!(action, Action::Relocate { .. })),
        f32::from(to_row) / size,
        f32::from(to_column) / size,
        f32::from(from_row) / size,
        f32::from(from_column) / size,
        f32::from(child.last_capture) / 4.0,
        f32::from(child.winner == Some(player)),
        f32::from(to_row == 0 || to_row + 1 == state.config.board_size),
        f32::from(to_column == 0 || to_column + 1 == state.config.board_size),
        state.pieces(player).count_ones() as f32 / f32::from(RESERVE_PER_PLAYER),
        state.pieces(player.other()).count_ones() as f32 / f32::from(RESERVE_PER_PLAYER),
        f32::from(state.reserve[player.index()]) / f32::from(RESERVE_PER_PLAYER),
        f32::from(state.reserve[player.other().index()]) / f32::from(RESERVE_PER_PLAYER),
        f32::from(state.ply) / 180.0,
        delta,
    ]
}

fn action_value_json(value: &GoldenActionValue) -> Value {
    json!({
        "action": action_json(value.action),
        "outcome": value.outcome.map(GoldenOutcome::as_str),
        "distance": value.distance,
    })
}

fn action_json(action: Action) -> Value {
    match action {
        Action::Place { to } => json!({"kind": "place", "to": to}),
        Action::Relocate { from, to } => json!({"kind": "relocate", "from": from, "to": to}),
    }
}

fn urgency_targets(
    outcome: GoldenOutcome,
    values: &[GoldenActionValue],
) -> Option<(Vec<Action>, u16)> {
    let desired = match outcome {
        GoldenOutcome::Win => GoldenOutcome::Win,
        GoldenOutcome::Loss => GoldenOutcome::Loss,
        GoldenOutcome::Draw => return None,
    };
    let mut candidates = values
        .iter()
        .filter_map(|value| {
            if value.outcome == Some(desired) {
                value.distance.map(|distance| (value.action, distance))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let distance = if desired == GoldenOutcome::Win {
        candidates.iter().map(|(_, distance)| *distance).min()?
    } else {
        candidates.iter().map(|(_, distance)| *distance).max()?
    };
    let mut actions = candidates
        .drain(..)
        .filter(|(_, candidate_distance)| *candidate_distance == distance)
        .map(|(action, _)| action)
        .collect::<Vec<_>>();
    actions.sort_by_key(|action| action.order());
    actions.dedup();
    Some((actions, distance))
}

fn parse_layers(spec: &str) -> Vec<(PathBuf, Option<PathBuf>)> {
    let layers = spec
        .split(';')
        .filter(|layer| !layer.trim().is_empty())
        .map(|layer| {
            let (table, sidecar) = layer.split_once(',').unwrap_or((layer, ""));
            let table = table.trim();
            if table.is_empty() || sidecar.trim().is_empty() {
                fail("--layers entries must be table,sidecar pairs");
            }
            (PathBuf::from(table), Some(PathBuf::from(sidecar.trim())))
        })
        .collect::<Vec<_>>();
    if layers.is_empty() {
        fail("--layers requires at least one table,sidecar pair");
    }
    layers
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

fn number<T: std::str::FromStr>(args: &HashMap<String, String>, key: &str, fallback: T) -> T {
    args.get(key)
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}

fn hex_key(key: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(key.len() * 2);
    for byte in key {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0xf) as usize] as char);
    }
    output
}

fn fail(message: &str) -> ! {
    eprintln!("pathagon-endgame-gold-export: {message}");
    std::process::exit(2);
}
