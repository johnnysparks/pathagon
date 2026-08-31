//! JSONL adapter for the Rust exact/unknown oracle.
//!
//! This small executable exists so the independent Python 3x3/4x4 solver can
//! compare action-level results against the gameplay implementation without
//! importing Rust internals or sharing its search code.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use pathagon_engine::corpus::encode_action;
use pathagon_engine::ground_truth::{analyze, GroundTruthConfig};
use pathagon_engine::{BoardConfig, GameState, Player};
use serde_json::{json, Value};

fn main() {
    let args = parse_args();
    let input = required(&args, "input");
    let horizon = args.get("horizon").map(|value| {
        value
            .parse()
            .unwrap_or_else(|_| fail("--horizon must be a non-negative integer"))
    });
    let max_nodes = args
        .get("max-nodes")
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| fail("--max-nodes must be a positive integer"))
        })
        .unwrap_or(2_000_000);
    for (line_number, line) in fs::read_to_string(&input)
        .unwrap_or_else(|error| fail(&format!("cannot read {}: {error}", input.display())))
        .lines()
        .enumerate()
    {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let raw: Value = serde_json::from_str(line).unwrap_or_else(|error| {
            fail(&format!("{}:{}: {error}", input.display(), line_number + 1))
        });
        let state = state_from_json(&raw).unwrap_or_else(|error| {
            fail(&format!("{}:{}: {error}", input.display(), line_number + 1))
        });
        let result = analyze(
            state,
            GroundTruthConfig {
                horizon,
                max_nodes,
                max_plies: None,
            },
        );
        let actions = result
            .actions
            .iter()
            .map(|action| {
                json!({
                    "token": encode_action(action.action),
                    "outcome": action.outcome,
                    "distance": action.distance,
                })
            })
            .collect::<Vec<_>>();
        println!(
            "{}",
            json!({
                "key": raw.get("id").and_then(Value::as_str),
                "outcome": result.outcome,
                "distance": result.distance,
                "optimalActionsComplete": result.optimal_actions_complete,
                "actions": actions,
            })
        );
    }
}

fn state_from_json(raw: &Value) -> Result<GameState, String> {
    let size = raw["boardSize"].as_u64().ok_or("boardSize is required")? as u8;
    let reserve = raw["reservePerPlayer"]
        .as_u64()
        .ok_or("reservePerPlayer is required")? as u8;
    let max_plies = raw["maxPlies"].as_u64().unwrap_or(180) as u16;
    let config = BoardConfig::new(size, reserve)?.with_max_plies(max_plies)?;
    let markers = raw["lastRelocatedTo"]
        .as_array()
        .ok_or("lastRelocatedTo is required")?
        .iter()
        .map(|value| value.as_u64().map(|value| value as u8))
        .collect::<Vec<_>>();
    if markers.len() != 2 {
        return Err("lastRelocatedTo must have two entries".to_owned());
    }
    Ok(GameState {
        config,
        light: raw["light"].as_u64().ok_or("light is required")?,
        dark: raw["dark"].as_u64().ok_or("dark is required")?,
        reserve: [
            raw["reserve"][0]
                .as_u64()
                .ok_or("light reserve is required")? as u8,
            raw["reserve"][1]
                .as_u64()
                .ok_or("dark reserve is required")? as u8,
        ],
        turn: match raw["turn"].as_str().ok_or("turn is required")? {
            "light" => Player::Light,
            "dark" => Player::Dark,
            other => return Err(format!("invalid turn {other}")),
        },
        forbidden: raw["forbidden"].as_u64().unwrap_or(0),
        last_relocated_to: [markers[0], markers[1]],
        last_capture: 0,
        last_player: None,
        winner: None,
        ply: raw["ply"].as_u64().unwrap_or(0) as u16,
    })
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
        values.insert(key.to_owned(), value);
    }
    values
}

fn required(args: &HashMap<String, String>, key: &str) -> PathBuf {
    args.get(key)
        .map(PathBuf::from)
        .unwrap_or_else(|| fail(&format!("--{key} <path> is required")))
}

fn fail(message: &str) -> ! {
    eprintln!("pathagon-endgame-oracle: {message}");
    std::process::exit(2);
}
