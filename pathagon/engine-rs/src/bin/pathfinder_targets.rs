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

use pathagon_engine::contract::Position;
use pathagon_engine::search::{
    analyze_action, analyze_actions, ordered_root_actions, search_best_action,
    search_best_action_with_golden, search_best_action_with_tactical_filter,
    tactical_root_safe_actions, MoveEvaluation, SearchConfig, SearchResult,
};
use pathagon_engine::{golden::GoldenLookup, Action, BoardConfig, GameState, Player};
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
    let target_temperature = number(&args, "target-temperature", 0.0_f32);
    let rank_actions = number(&args, "rank-actions", 0_usize);
    let rank_nodes = number(&args, "rank-nodes", max_nodes);
    let tactical_filter = args.contains_key("tactical-filter");
    let golden = match (args.get("golden-table"), args.get("golden-sidecar")) {
        (Some(table), sidecar) => Some(
            GoldenLookup::open(PathBuf::from(table), sidecar.map(PathBuf::from), 7, 14)
                .unwrap_or_else(|error| fail(&format!("cannot load golden data: {error}"))),
        ),
        (None, Some(_)) => fail("--golden-sidecar requires --golden-table"),
        (None, None) => None,
    };
    if depth == 0
        || max_nodes == 0
        || beam_width == 0
        || target_temperature < 0.0
        || (rank_actions > 0 && rank_nodes == 0)
    {
        fail(
            "--depth, --nodes, and --beam must be positive; target temperature cannot be negative; rank nodes must be positive",
        );
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
            emit_record(
                &parsed,
                &mut writer,
                config,
                &mut positions,
                max_positions,
                target_temperature,
                rank_actions,
                rank_nodes,
                tactical_filter,
                golden.as_ref(),
            );
        } else {
            emit_record(
                record,
                &mut writer,
                config,
                &mut positions,
                max_positions,
                target_temperature,
                rank_actions,
                rank_nodes,
                tactical_filter,
                golden.as_ref(),
            );
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
                "temperature": target_temperature,
            "rankActions": rank_actions,
            "rankNodes": rank_nodes,
            "tacticalFilter": tactical_filter,
            },
            "golden": golden.as_ref().map(|lookup| json!({
                "table": lookup.table.path(),
                "rows": lookup.table.rows(),
                "actionRows": lookup.actions.as_ref().map(|book| book.rows()),
            })),
        })
    );
}

fn emit_record(
    record: &Value,
    writer: &mut BufWriter<File>,
    config: SearchConfig,
    positions: &mut usize,
    max_positions: usize,
    target_temperature: f32,
    rank_actions: usize,
    rank_nodes: u64,
    tactical_filter: bool,
    golden: Option<&GoldenLookup>,
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
    let mut state = record
        .get("initialPosition")
        .map(|value| {
            let position: Position = serde_json::from_value(value.clone())
                .unwrap_or_else(|error| fail(&format!("invalid initialPosition: {error}")));
            GameState::from_position(&position)
                .unwrap_or_else(|error| fail(&format!("invalid initialPosition: {error}")))
        })
        .unwrap_or_else(|| GameState::with_config(board_config));
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
        let rankings = if rank_actions == 0 {
            Vec::new()
        } else {
            independent_rankings(
                state,
                SearchConfig {
                    max_nodes: rank_nodes,
                    ..config
                },
                rank_actions,
                tactical_filter,
            )
        };
        let golden_outcome = golden.and_then(|lookup| lookup.lookup(state));
        let golden_action = golden.and_then(|lookup| lookup.proven_action(state));
        let (target, policy, target_score, target_nodes, target_depth) =
            if let Some(target) = golden_action {
                let target_index = legal.iter().position(|candidate| *candidate == target);
                let Some(target_index) = target_index else {
                    fail("golden action is not legal in the replayed state");
                };
                let mut policy = vec![0.0_f32; legal.len()];
                policy[target_index] = 1.0;
                (target, policy, 1_000_000_000, 0, 0)
            } else if target_temperature > 0.0 {
                let analyses = if rankings.is_empty() {
                    if tactical_filter {
                        independent_rankings(state, config, legal.len(), true)
                    } else {
                        analyze_actions(state, config, legal.len())
                    }
                } else {
                    rankings.clone()
                };
                let best = analyses.first().copied();
                let target = best.map(|result| result.action).unwrap_or(action);
                let maximum = analyses
                    .iter()
                    .map(|result| result.score)
                    .max()
                    .unwrap_or(0) as f32;
                let mut policy = vec![0.0_f32; legal.len()];
                let mut total = 0.0_f32;
                for result in analyses {
                    let weight = ((result.score as f32 - maximum) / target_temperature).exp();
                    if let Some(index) = legal
                        .iter()
                        .position(|candidate| *candidate == result.action)
                    {
                        policy[index] = weight;
                        total += weight;
                    }
                }
                if total > 0.0 {
                    for value in &mut policy {
                        *value /= total;
                    }
                }
                let Some(best) = best else {
                    fail("Pathfinder did not score a legal target action");
                };
                (target, policy, best.score, best.nodes, best.completed_depth)
            } else {
                let result = rankings.first().copied().map_or_else(
                    || {
                        if tactical_filter {
                            search_best_action_with_tactical_filter(state, config)
                        } else if let Some(golden) = golden {
                            search_best_action_with_golden(state, config, golden)
                        } else {
                            search_best_action(state, config)
                        }
                    },
                    |ranking| SearchResult {
                        action: Some(ranking.action),
                        score: ranking.score,
                        nodes: ranking.nodes,
                        exhausted: ranking.exhausted,
                        completed_depth: ranking.completed_depth,
                        table_hits: ranking.table_hits,
                    },
                );
                let target = result.action.unwrap_or(action);
                let target_index = legal.iter().position(|candidate| *candidate == target);
                let Some(target_index) = target_index else {
                    fail("Pathfinder target is not legal in the replayed state");
                };
                let mut policy = vec![0.0_f32; legal.len()];
                policy[target_index] = 1.0;
                (
                    target,
                    policy,
                    result.score,
                    result.nodes,
                    result.completed_depth,
                )
            };
        let mut output_move = movement.clone();
        if let Some(object) = output_move.as_object_mut() {
            object.insert("policy".to_owned(), json!(policy));
            object.insert("targetAction".to_owned(), action_value_json(target));
            object.insert("targetScore".to_owned(), json!(target_score));
            object.insert("targetNodes".to_owned(), json!(target_nodes));
            object.insert("targetDepth".to_owned(), json!(target_depth));
            if let Some(outcome) = golden_outcome {
                object.insert("goldenOutcome".to_owned(), json!(outcome.as_str()));
            }
            if let Some(golden_action) = golden_action {
                object.insert("goldenAction".to_owned(), action_value_json(golden_action));
                object.insert("goldenDistance".to_owned(), json!(1));
                object.insert("goldenPolicyComplete".to_owned(), json!(false));
                object.insert("targetSource".to_owned(), json!("golden-ring-1"));
            } else {
                object.insert("targetSource".to_owned(), json!("pathfinder-search"));
            }
            if !rankings.is_empty() {
                object.insert(
                    "rankActions".to_owned(),
                    json!(rankings
                        .iter()
                        .map(|result| action_value_json(result.action))
                        .collect::<Vec<_>>()),
                );
                object.insert(
                    "rankScores".to_owned(),
                    json!(rankings
                        .iter()
                        .map(|result| result.score)
                        .collect::<Vec<_>>()),
                );
                object.insert(
                    "rankExhausted".to_owned(),
                    json!(rankings
                        .iter()
                        .map(|result| result.exhausted)
                        .collect::<Vec<_>>()),
                );
                object.insert(
                    "rankNodesUsed".to_owned(),
                    json!(rankings
                        .iter()
                        .map(|result| result.nodes)
                        .collect::<Vec<_>>()),
                );
            }
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
    let mut output_record = json!({
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
    if let Some(initial) = record.get("initialPosition") {
        output_record["initialPosition"] = initial.clone();
    }
    if let Some(provenance) = record.get("provenance") {
        output_record["provenance"] = provenance.clone();
    }
    serde_json::to_writer(&mut *writer, &output_record)
        .unwrap_or_else(|error| fail(&format!("cannot serialize target record: {error}")));
    writer
        .write_all(b"\n")
        .unwrap_or_else(|error| fail(&format!("cannot write target record: {error}")));
}

fn independent_rankings(
    state: GameState,
    config: SearchConfig,
    max_actions: usize,
    tactical_filter: bool,
) -> Vec<MoveEvaluation> {
    let actions = if tactical_filter {
        tactical_root_safe_actions(state, state.turn, config.weights)
    } else {
        ordered_root_actions(state, state.turn, config.weights)
    };
    let mut results = actions
        .into_iter()
        .take(max_actions)
        .filter_map(|action| analyze_action(state, action, config).ok())
        .collect::<Vec<_>>();
    results.sort_by_key(|result| (-result.score, result.action.order()));
    results
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
            } else {
                parsed.insert(option.to_owned(), "true".to_owned());
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
