//! Measure a Rust Pathfinder strategy against exact held-out endgame rows.
//!
//! This is intentionally a one-move match for the replay-ring partition:
//! every Ring-1 row has a proven action that reaches a terminal position on
//! the next move.  The search is run without the golden lookup, then its
//! selected action is applied through the Rust rules boundary and compared
//! with the sparse exact action labels.  The harness is therefore useful for
//! model/search gates without allowing the oracle to answer the move under
//! test.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use pathagon_engine::corpus::encode_action;
use pathagon_engine::golden::{decode_canonical_position_key, GoldenLookup, GoldenOutcome};
use pathagon_engine::search::{
    search_best_action, search_best_action_with_tactical_filter,
    search_best_action_with_tactical_guard, search_best_action_with_tt_order, SearchConfig,
};
use serde_json::json;

const BOARD_SIZE: u8 = 7;
const RESERVE_PER_PLAYER: u8 = 14;

fn main() {
    let args = parse_args();
    let table = required(&args, "table");
    let sidecar = required(&args, "sidecar");
    let heldout = required(&args, "heldout");
    let output = required(&args, "out");
    let strategy = args
        .get("strategy")
        .map(String::as_str)
        .unwrap_or("baseline");
    let max_positions = number(&args, "max-positions", 0_usize);
    let config = SearchConfig {
        depth: number(&args, "depth", 4_u8),
        max_nodes: number(&args, "nodes", 5_000_u64),
        beam_width: number(&args, "beam", 8_usize),
        ..SearchConfig::default()
    };
    if config.depth == 0 || config.max_nodes == 0 || config.beam_width == 0 {
        fail("--depth, --nodes, and --beam must be positive");
    }
    if !matches!(
        strategy,
        "baseline" | "tactical-filter" | "tactical-guard" | "tt-order"
    ) {
        fail("--strategy must be baseline, tactical-filter, tactical-guard, or tt-order");
    }

    let golden = GoldenLookup::open(&table, Some(&sidecar), BOARD_SIZE, RESERVE_PER_PLAYER)
        .unwrap_or_else(|error| fail(&format!("cannot load golden data: {error}")));
    let source = fs::read_to_string(&heldout)
        .unwrap_or_else(|error| fail(&format!("cannot read {}: {error}", heldout.display())));
    let mut positions = Vec::new();
    let mut rows_seen = 0_usize;
    let mut rows_with_actions = 0_usize;
    let mut matched_proven = 0_usize;
    let mut terminal_wins = 0_usize;
    let mut illegal_results = 0_usize;
    let mut exhausted = 0_usize;
    let mut total_nodes = 0_u64;

    for (line_number, line) in source.lines().enumerate() {
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        if max_positions > 0 && rows_seen >= max_positions {
            break;
        }
        let key_text = line.trim();
        let key = decode_hex(key_text).unwrap_or_else(|error| {
            fail(&format!(
                "{}:{}: {error}",
                heldout.display(),
                line_number + 1
            ))
        });
        let state = decode_canonical_position_key(&key, BOARD_SIZE, RESERVE_PER_PLAYER)
            .unwrap_or_else(|error| {
                fail(&format!(
                    "{}:{}: {error}",
                    heldout.display(),
                    line_number + 1
                ))
            });
        if pathagon_engine::golden::canonical_position_key(state) != key {
            fail(&format!(
                "{}:{}: held-out key is not canonical",
                heldout.display(),
                line_number + 1
            ));
        }
        let outcome = golden.lookup(state).unwrap_or_else(|| {
            fail(&format!(
                "{}:{}: held-out key is absent from the exact table",
                heldout.display(),
                line_number + 1
            ))
        });
        let row = golden.row_value(state).unwrap_or_else(|| {
            fail(&format!(
                "{}:{}: held-out key is missing its action-book row",
                heldout.display(),
                line_number + 1
            ))
        });
        let legal_actions = state.legal_actions();
        let mut proven_actions = golden
            .action_values(state)
            .unwrap_or_else(|| {
                fail(&format!(
                    "{}:{}: held-out key is missing action labels",
                    heldout.display(),
                    line_number + 1
                ))
            })
            .into_iter()
            .filter(|value| value.outcome == Some(GoldenOutcome::Win))
            .map(|value| value.action)
            .filter(|action| legal_actions.contains(action))
            .collect::<Vec<_>>();
        proven_actions.sort_by_key(|action| action.order());
        proven_actions.dedup();
        rows_with_actions += usize::from(!proven_actions.is_empty());

        let result = match strategy {
            "baseline" => search_best_action(state, config),
            "tactical-filter" => search_best_action_with_tactical_filter(state, config),
            "tactical-guard" => search_best_action_with_tactical_guard(state, config),
            "tt-order" => search_best_action_with_tt_order(state, config),
            _ => unreachable!("strategy was validated above"),
        };
        let selected_legal = result
            .action
            .is_some_and(|action| legal_actions.contains(&action));
        let selected_matches_proven = result
            .action
            .is_some_and(|action| proven_actions.contains(&action));
        let terminal_win = result.action.is_some_and(|action| {
            selected_legal && state.apply_legal(action).state.winner == Some(state.turn)
        });
        matched_proven += usize::from(selected_matches_proven);
        terminal_wins += usize::from(terminal_win);
        illegal_results += usize::from(!selected_legal);
        exhausted += usize::from(result.exhausted);
        total_nodes = total_nodes.saturating_add(result.nodes);
        positions.push(json!({
            "key": key_text,
            "goldOutcome": outcome.as_str(),
            "goldDistance": row.distance,
            "optimalActionsComplete": row.optimal_actions_complete,
            "provenActions": proven_actions.iter().copied().map(encode_action).collect::<Vec<_>>(),
            "selectedAction": result.action.map(encode_action),
            "selectedLegal": selected_legal,
            "matchedProvenAction": selected_matches_proven,
            "terminalWin": terminal_win,
            "nodes": result.nodes,
            "completedDepth": result.completed_depth,
            "exhausted": result.exhausted,
            "tableHits": result.table_hits,
        }));
        rows_seen += 1;
    }
    if rows_seen == 0 {
        fail("held-out partition contains no rows");
    }

    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| fail(&format!("cannot create output directory: {error}")));
    }
    let report = json!({
        "schemaVersion": 1,
        "experiment": "rust-pathfinder-heldout-exact-match",
        "table": table,
        "sidecar": sidecar,
        "heldout": heldout,
        "strategy": strategy,
        "search": {
            "depth": config.depth,
            "maxNodes": config.max_nodes,
            "beamWidth": config.beam_width,
            "weights": {
                "path": config.weights.path,
                "material": config.weights.material,
                "capture": config.weights.capture,
                "structure": config.weights.structure,
                "threat": config.weights.threat,
                "edge": config.weights.edge,
            },
        },
        "summary": {
            "positions": rows_seen,
            "goldRows": golden.table.rows(),
            "rowsWithProvenActions": rows_with_actions,
            "matchedProvenActions": matched_proven,
            "matchRate": matched_proven as f64 / rows_seen as f64,
            "terminalWins": terminal_wins,
            "terminalWinRate": terminal_wins as f64 / rows_seen as f64,
            "illegalResults": illegal_results,
            "exhaustedSearches": exhausted,
            "totalNodes": total_nodes,
            "averageNodes": total_nodes as f64 / rows_seen as f64,
        },
        "positionsDetail": positions,
        "status": "pass",
    });
    fs::write(
        &output,
        serde_json::to_string_pretty(&report).expect("report is serializable") + "\n",
    )
    .unwrap_or_else(|error| fail(&format!("cannot write {}: {error}", output.display())));
    println!(
        "{}",
        json!({
            "experiment": "rust-pathfinder-heldout-exact-match",
            "strategy": strategy,
            "positions": rows_seen,
            "matchedProvenActions": matched_proven,
            "matchRate": matched_proven as f64 / rows_seen as f64,
            "terminalWins": terminal_wins,
            "terminalWinRate": terminal_wins as f64 / rows_seen as f64,
            "totalNodes": total_nodes,
            "status": "pass",
        })
    );
}

fn decode_hex(value: &str) -> Result<Vec<u8>, String> {
    if value.is_empty() || value.len() % 2 != 0 {
        return Err("canonical key must contain a non-empty even number of hex digits".to_owned());
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let high = hex_digit(pair[0])?;
            let low = hex_digit(pair[1])?;
            Ok((high << 4) | low)
        })
        .collect()
}

fn hex_digit(value: u8) -> Result<u8, String> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err("canonical key contains a non-hex digit".to_owned()),
    }
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

fn number<T>(args: &HashMap<String, String>, key: &str, default: T) -> T
where
    T: std::str::FromStr,
{
    args.get(key)
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| fail(&format!("--{key} must be a positive integer")))
        })
        .unwrap_or(default)
}

fn fail(message: &str) -> ! {
    eprintln!("pathagon-endgame-match: {message}");
    std::process::exit(2);
}

#[cfg(test)]
mod tests {
    use super::decode_hex;

    #[test]
    fn decodes_canonical_hex_keys() {
        assert_eq!(decode_hex("00ff").unwrap(), vec![0, 255]);
        assert!(decode_hex("").is_err());
        assert!(decode_hex("0").is_err());
        assert!(decode_hex("0g").is_err());
    }
}
