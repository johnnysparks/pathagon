//! Emit policy targets from the native Pathfinder search.
//!
//! The input is one or more schema-v2 JSONL game archives produced by the
//! Rust self-play runner. Every archived position is replayed by this binary,
//! then a deeper Pathfinder search supplies a one-hot policy target aligned to
//! the complete legal action list. The output remains schema-v2-compatible so
//! the existing learner can train a compact policy sorter without importing a
//! second rules implementation.

use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use pathagon_engine::search::{search_best_action, SearchConfig};
use pathagon_engine::{Action, BoardConfig, GameState, Player};
use serde_json::{json, Value};

fn main() {
    let args = parse_args();
    let input = args
        .get("input")
        .map(PathBuf::from)
        .unwrap_or_else(|| fail("--input <archive.jsonl> is required"));
    let output = args
        .get("out")
        .map(PathBuf::from)
        .unwrap_or_else(|| fail("--out <targets.jsonl> is required"));
    let depth = number(&args, "depth", 5_u8);
    let max_nodes = number(&args, "nodes", 5_000_u64);
    let beam_width = number(&args, "beam", 8_usize);
    let max_games = number(&args, "max-games", 0_usize);
    let max_positions = number(&args, "max-positions", 0_usize);
    if depth == 0 || max_nodes == 0 || beam_width == 0 {
        fail("--depth, --nodes, and --beam must be positive");
    }

    let reader = BufReader::new(
        File::open(&input).unwrap_or_else(|error| fail(&format!("cannot open input: {error}"))),
    );
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| fail(&format!("cannot create output directory: {error}")));
    }
    let file = File::create(&output)
        .unwrap_or_else(|error| fail(&format!("cannot create output: {error}")));
    let mut writer = BufWriter::new(file);
    let config = SearchConfig {
        depth,
        max_nodes,
        beam_width,
        ..SearchConfig::default()
    };

    let mut games = 0_usize;
    let mut positions = 0_usize;
    for (line_number, line) in reader.lines().enumerate() {
        if max_games > 0 && games >= max_games {
            break;
        }
        if max_positions > 0 && positions >= max_positions {
            break;
        }
        let line = line.unwrap_or_else(|error| fail(&format!("cannot read line: {error}")));
        if line.trim().is_empty() {
            continue;
        }
        let raw: Value = serde_json::from_str(&line).unwrap_or_else(|error| {
            fail(&format!(
                "invalid JSON at line {}: {error}",
                line_number + 1
            ))
        });
        let record = raw.get("record").unwrap_or(&raw);
        if let Some(record_text) = record.as_str() {
            let parsed: Value = serde_json::from_str(record_text).unwrap_or_else(|error| {
                fail(&format!(
                    "invalid nested record at line {}: {error}",
                    line_number + 1
                ))
            });
            emit_record(&parsed, &mut writer, config, &mut positions, max_positions);
        } else {
            emit_record(record, &mut writer, config, &mut positions, max_positions);
        }
        games += 1;
    }
    writer
        .flush()
        .unwrap_or_else(|error| fail(&format!("cannot flush output: {error}")));
    println!(
        "{}",
        json!({
            "schemaVersion": 1,
            "input": input,
            "out": output,
            "games": games,
            "positions": positions,
            "target": {
                "depth": depth,
                "beam": beam_width,
                "nodes": max_nodes,
            },
        })
    );
}

fn emit_record(
    record: &Value,
    writer: &mut BufWriter<File>,
    config: SearchConfig,
    positions: &mut usize,
    max_positions: usize,
) {
    let moves = record
        .get("moves")
        .and_then(Value::as_array)
        .unwrap_or_else(|| fail("record is missing a moves array"));
    let source_config = record.get("config").unwrap_or(&Value::Null);
    let board_size = source_config
        .get("boardSize")
        .and_then(Value::as_u64)
        .or_else(|| record.get("boardSize").and_then(Value::as_u64))
        .unwrap_or(7) as u8;
    let reserve = source_config
        .get("reservePerPlayer")
        .and_then(Value::as_u64)
        .or_else(|| record.get("reservePerPlayer").and_then(Value::as_u64))
        .unwrap_or(u64::from(board_size.saturating_mul(2))) as u8;
    let max_plies = source_config
        .get("maxPlies")
        .and_then(Value::as_u64)
        .or_else(|| record.get("maxPlies").and_then(Value::as_u64))
        .unwrap_or(180) as u16;
    let board_config = BoardConfig::new(board_size, reserve)
        .and_then(|value| value.with_max_plies(max_plies))
        .unwrap_or_else(|error| fail(&format!("invalid record board configuration: {error}")));
    let mut state = GameState::with_config(board_config);
    let mut target_moves = Vec::with_capacity(moves.len());
    for movement in moves {
        if max_positions > 0 && *positions >= max_positions {
            break;
        }
        let Some(action_value) = movement.get("action") else {
            fail("move is missing an action");
        };
        let action = parse_action(action_value);
        let legal = state.legal_actions();
        if !legal.contains(&action) {
            fail(&format!("archived action is illegal at ply {}", state.ply));
        }
        let result = search_best_action(state, config);
        let target = result.action.unwrap_or(action);
        let target_index = legal.iter().position(|candidate| *candidate == target);
        let Some(target_index) = target_index else {
            fail("Pathfinder target is not legal in the replayed state");
        };
        let mut policy = vec![0.0_f32; legal.len()];
        policy[target_index] = 1.0;
        let mut output_move = movement.clone();
        if let Some(object) = output_move.as_object_mut() {
            object.insert("policy".to_owned(), json!(policy));
            object.insert("targetAction".to_owned(), action_value_json(target));
            object.insert("targetScore".to_owned(), json!(result.score));
            object.insert("targetNodes".to_owned(), json!(result.nodes));
            object.insert("targetDepth".to_owned(), json!(result.completed_depth));
        }
        target_moves.push(output_move);
        *positions += 1;
        state = state.apply_legal(action).state;
    }
    if target_moves.is_empty() {
        return;
    }
    let winner = state.winner.map(Player::as_str);
    let reason = if winner.is_some() {
        "path"
    } else {
        "max-plies"
    };
    let output_record = json!({
        "contractVersion": record.get("contractVersion").cloned().unwrap_or(json!(1)),
        "seed": record.get("seed").cloned().unwrap_or(json!(0)),
        "config": {
            "rulesVersion": "pathagon-rules-v1",
            "boardSize": board_size,
            "reservePerPlayer": reserve,
            "maxPlies": max_plies,
            "repetitionLimit": 3,
        },
        "engine": {"id": "rust-bitboard", "runtime": "rust", "version": "1.0.0", "rulesVersion": "pathagon-rules-v1"},
        "agents": {"light": "rust-pathfinder-target-source", "dark": "rust-pathfinder-target-source"},
        "winner": winner,
        "result": if winner.is_some() { "win" } else { "draw" },
        "reason": reason,
        "plies": target_moves.len(),
        "moves": target_moves,
    });
    serde_json::to_writer(&mut *writer, &output_record)
        .unwrap_or_else(|error| fail(&format!("cannot serialize target record: {error}")));
    writer
        .write_all(b"\n")
        .unwrap_or_else(|error| fail(&format!("cannot write target record: {error}")));
}

fn parse_action(value: &Value) -> Action {
    let kind = value
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or_else(|| fail("action is missing kind"));
    match kind {
        "place" => Action::Place {
            to: value
                .get("to")
                .and_then(Value::as_u64)
                .unwrap_or_else(|| fail("place action is missing to")) as u8,
        },
        "relocate" => Action::Relocate {
            from: value
                .get("from")
                .and_then(Value::as_u64)
                .unwrap_or_else(|| fail("relocate action is missing from")) as u8,
            to: value
                .get("to")
                .and_then(Value::as_u64)
                .unwrap_or_else(|| fail("relocate action is missing to")) as u8,
        },
        _ => fail("unsupported action kind"),
    }
}

fn action_value_json(action: Action) -> Value {
    match action {
        Action::Place { to } => json!({"kind": "place", "to": to}),
        Action::Relocate { from, to } => json!({"kind": "relocate", "from": from, "to": to}),
    }
}

fn parse_args() -> HashMap<String, String> {
    let values: Vec<String> = env::args().skip(1).collect();
    let mut parsed = HashMap::new();
    let mut index = 0;
    while index < values.len() {
        if let Some(option) = values[index].strip_prefix("--") {
            if let Some((key, value)) = option.split_once('=') {
                parsed.insert(key.to_owned(), value.to_owned());
            } else if values
                .get(index + 1)
                .is_some_and(|next| !next.starts_with("--"))
            {
                parsed.insert(option.to_owned(), values[index + 1].clone());
                index += 1;
            }
        }
        index += 1;
    }
    parsed
}

fn number<T: std::str::FromStr>(args: &HashMap<String, String>, key: &str, fallback: T) -> T {
    args.get(key)
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}

fn fail(message: &str) -> ! {
    eprintln!("pathfinder-targets: {message}");
    std::process::exit(2)
}
