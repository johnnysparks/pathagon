//! Export exact Rust state/action/afterstate tuples for JEPA training.
//!
//! The learner may discover a compact representation, but this exporter is
//! deliberately the only world-model boundary used by the experiment. Every
//! successor state is produced by `GameState::apply_legal`, and every action
//! comes from `GameState::legal_actions`.

use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use pathagon_engine::{Action, BoardConfig, GameState};
use serde_json::{json, Value};

#[derive(Clone, Copy)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 7;
        value ^= value >> 9;
        value ^= value << 8;
        self.0 = value;
        value
    }

    fn index(&mut self, length: usize) -> usize {
        (self.next_u64() as usize) % length
    }
}

#[derive(Default)]
struct Stats {
    games: usize,
    positions: usize,
    transitions: usize,
    terminal_positions: usize,
}

fn main() {
    let args = parse_args();
    let output = required_path(&args, "out");
    let games = number(&args, "games", 64_usize);
    let max_plies = number(&args, "max-plies", 40_u16);
    let actions_per_state = number(&args, "actions-per-state", 32_usize);
    let seed = number(&args, "seed", 2026090101_u64);
    if games == 0 || max_plies == 0 || actions_per_state == 0 {
        fail("games, max-plies, and actions-per-state must be positive");
    }

    let config = BoardConfig::new(7, 14)
        .and_then(|config| config.with_max_plies(max_plies))
        .unwrap_or_else(|error| fail(&format!("invalid export configuration: {error}")));
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| fail(&format!("cannot create output directory: {error}")));
    }
    let file = File::create(&output)
        .unwrap_or_else(|error| fail(&format!("cannot create output: {error}")));
    let mut writer = BufWriter::new(file);
    let mut stats = Stats::default();

    for game_index in 0..games {
        let game_seed = seed.wrapping_add(game_index as u64);
        let mut rng = Rng::new(game_seed ^ 0x9e37_79b9_7f4a_7c15);
        let mut state = GameState::with_config(config);
        for _ in 0..max_plies {
            if state.winner.is_some() {
                stats.terminal_positions += 1;
                break;
            }
            let legal = state.legal_actions();
            if legal.is_empty() {
                break;
            }
            let chosen = legal[rng.index(legal.len())];
            let selected = sample_actions(&legal, chosen, actions_per_state, &mut rng);
            stats.positions += 1;
            for action in selected {
                let next_state = state.apply_legal(action).state;
                let row = json!({
                    "schemaVersion": 1,
                    "format": "pathagon-rust-jepa-afterstate-v1",
                    "engine": {
                        "id": "rust-bitboard",
                        "runtime": "rust",
                        "rulesVersion": "pathagon-rules-v1"
                    },
                    "game": game_index,
                    "seed": game_seed,
                    "state": state_json(state),
                    "action": action_json(action),
                    "nextState": state_json(next_state),
                    "selectedForRollout": action == chosen
                });
                serde_json::to_writer(&mut writer, &row)
                    .unwrap_or_else(|error| fail(&format!("cannot serialize transition: {error}")));
                writer
                    .write_all(b"\n")
                    .unwrap_or_else(|error| fail(&format!("cannot write transition: {error}")));
                stats.transitions += 1;
            }
            state = state.apply_legal(chosen).state;
        }
        stats.games += 1;
    }
    writer
        .flush()
        .unwrap_or_else(|error| fail(&format!("cannot flush output: {error}")));

    println!(
        "{}",
        json!({
            "schemaVersion": 1,
            "format": "pathagon-rust-jepa-afterstate-v1",
            "out": output,
            "config": {
                "boardSize": config.board_size,
                "reservePerPlayer": config.reserve_per_player,
                "maxPlies": config.max_plies
            },
            "games": stats.games,
            "positions": stats.positions,
            "transitions": stats.transitions,
            "terminalPositions": stats.terminal_positions,
            "authority": "pathagon-engine-rs::GameState::legal_actions/apply_legal"
        })
    );
}

fn sample_actions(legal: &[Action], chosen: Action, limit: usize, rng: &mut Rng) -> Vec<Action> {
    if legal.len() <= limit {
        return legal.to_vec();
    }
    let mut selected = vec![chosen];
    let mut remaining = legal
        .iter()
        .copied()
        .filter(|action| *action != chosen)
        .collect::<Vec<_>>();
    while selected.len() < limit {
        let index = rng.index(remaining.len());
        selected.push(remaining.swap_remove(index));
    }
    selected.sort_by_key(|action| action.order());
    selected
}

fn state_json(state: GameState) -> Value {
    json!({
        "boardSize": state.config.board_size,
        "reservePerPlayer": state.config.reserve_per_player,
        "maxPlies": state.config.max_plies,
        "light": state.light,
        "dark": state.dark,
        "reserve": state.reserve,
        "turn": state.turn.as_str(),
        "forbidden": state.forbidden,
        "lastRelocatedTo": state.last_relocated_to,
        "lastCapture": state.last_capture,
        "lastPlayer": state.last_player.map(|player| player.as_str()),
        "winner": state.winner.map(|player| player.as_str()),
        "ply": state.ply
    })
}

fn action_json(action: Action) -> Value {
    match action {
        Action::Place { to } => json!({"kind": "place", "to": to}),
        Action::Relocate { from, to } => json!({"kind": "relocate", "from": from, "to": to}),
    }
}

fn parse_args() -> HashMap<String, String> {
    let values = env::args().skip(1).collect::<Vec<_>>();
    let mut parsed = HashMap::new();
    let mut index = 0;
    while index < values.len() {
        let value = &values[index];
        if let Some(option) = value.strip_prefix("--") {
            if let Some((key, inline)) = option.split_once('=') {
                parsed.insert(key.to_owned(), inline.to_owned());
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

fn required_path(args: &HashMap<String, String>, key: &str) -> PathBuf {
    args.get(key)
        .map(PathBuf::from)
        .unwrap_or_else(|| fail(&format!("--{key} is required")))
}

fn number<T>(args: &HashMap<String, String>, key: &str, fallback: T) -> T
where
    T: std::str::FromStr,
{
    args.get(key)
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}

fn fail(message: &str) -> ! {
    eprintln!("pathagon-jepa-export: {message}");
    std::process::exit(2)
}
