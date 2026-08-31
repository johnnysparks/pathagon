use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::sync::{Arc, Mutex};
use std::thread;

use pathagon_engine::search::{EvaluationWeights, SearchConfig};
use pathagon_engine::selfplay::{play_game, Agent, MatchOptions};
use serde::Deserialize;

#[derive(Clone, Deserialize)]
struct ContextualWeightsFile {
    weights: BTreeMap<String, EvaluationWeights>,
}

fn number<T: std::str::FromStr>(args: &BTreeMap<String, String>, key: &str, default: T) -> T {
    args.get(key)
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn args() -> BTreeMap<String, String> {
    let values = env::args().skip(1).collect::<Vec<_>>();
    let mut parsed = BTreeMap::new();
    let mut index = 0;
    while index < values.len() {
        if let Some(key) = values[index].strip_prefix("--") {
            if let Some(value) = values
                .get(index + 1)
                .filter(|value| !value.starts_with("--"))
            {
                parsed.insert(key.to_owned(), value.clone());
                index += 1;
            } else {
                parsed.insert(key.to_owned(), "true".to_owned());
            }
        }
        index += 1;
    }
    parsed
}

fn required(args: &BTreeMap<String, String>, key: &str) -> String {
    args.get(key)
        .cloned()
        .unwrap_or_else(|| panic!("missing --{key}"))
}

fn load_weights(path: &str) -> [EvaluationWeights; 4] {
    let document: ContextualWeightsFile = serde_json::from_str(
        &fs::read_to_string(path).unwrap_or_else(|error| panic!("read {path}: {error}")),
    )
    .unwrap_or_else(|error| panic!("parse {path}: {error}"));
    [
        *document
            .weights
            .get("opening")
            .expect("contextual weights missing opening"),
        *document
            .weights
            .get("placement")
            .expect("contextual weights missing placement"),
        *document
            .weights
            .get("movement")
            .expect("contextual weights missing movement"),
        *document
            .weights
            .get("late-game")
            .expect("contextual weights missing late-game"),
    ]
}

fn make_configs(
    weights: [EvaluationWeights; 4],
    depth: u8,
    max_nodes: u64,
    beam_width: usize,
) -> [SearchConfig; 4] {
    weights.map(|weights| SearchConfig {
        depth,
        max_nodes,
        beam_width,
        weights,
        tactical_proof_horizon: None,
    })
}

fn main() {
    let args = args();
    let output = required(&args, "output");
    let contextual_path = args.get("contextual-weights").cloned();
    let contextual_light_path = args.get("contextual-weights-light").cloned();
    let contextual_dark_path = args.get("contextual-weights-dark").cloned();
    let games = number(&args, "games", 120_usize);
    let workers = number(&args, "workers", 10_usize).max(1);
    let seed = number(&args, "seed", 20_260_829_u32);
    let max_plies = number(&args, "max-plies", 100_u16);
    let opening_random_plies = number(&args, "opening-random-plies", 2_u16);
    let depth = number(&args, "depth", 7_u8);
    let max_nodes = number(&args, "nodes", 500_000_u64);
    let beam_width = number(&args, "beam", 32_usize);
    let deadline_ms = number(&args, "deadline-ms", 2_800_u32);
    assert!(games > 0, "games must be positive");
    let configs = contextual_path
        .as_deref()
        .map(load_weights)
        .map(|weights| make_configs(weights, depth, max_nodes, beam_width));
    let baseline_weights = EvaluationWeights {
        path: 241,
        material: 112,
        capture: 887,
        structure: 40,
        threat: 154,
        edge: 74,
    };
    let candidate = match (
        contextual_light_path.as_deref(),
        contextual_dark_path.as_deref(),
    ) {
        (Some(light_path), Some(dark_path)) => Agent::contextual_by_player_with_deadline(
            "pathfinder-contextual-v0.6-research",
            make_configs(load_weights(light_path), depth, max_nodes, beam_width),
            make_configs(load_weights(dark_path), depth, max_nodes, beam_width),
            deadline_ms,
        ),
        _ => Agent::contextual_with_deadline(
            "pathfinder-contextual-v0.6-research",
            configs
                .unwrap_or_else(|| panic!("missing --contextual-weights or paired weight files")),
            deadline_ms,
        ),
    };
    let incumbent = Agent::search_tactical_filter_with_deadline(
        "pathfinder-v0.5.0-trained-evaluator",
        SearchConfig {
            depth,
            max_nodes,
            beam_width,
            weights: baseline_weights,
            tactical_proof_horizon: None,
        },
        deadline_ms,
    );
    let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let records = Arc::new(Mutex::new(Vec::<(usize, String)>::with_capacity(games)));
    let mut handles = Vec::new();
    for _ in 0..workers {
        let next = Arc::clone(&next);
        let records = Arc::clone(&records);
        let candidate = candidate.clone();
        let incumbent = incumbent.clone();
        handles.push(thread::spawn(move || loop {
            let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if index >= games {
                break;
            }
            let options = MatchOptions {
                seed: seed.wrapping_add(index as u32),
                max_plies,
                opening_random_plies,
                ..MatchOptions::default()
            };
            let record = if index % 2 == 0 {
                play_game(&candidate, &incumbent, options)
            } else {
                play_game(&incumbent, &candidate, options)
            };
            records
                .lock()
                .expect("arena records lock")
                .push((index, record.to_json()));
        }));
    }
    for handle in handles {
        handle.join().expect("arena worker");
    }
    let mut records = Arc::try_unwrap(records)
        .expect("arena records references")
        .into_inner()
        .expect("arena records lock");
    records.sort_by_key(|(index, _)| *index);
    let output_text = records
        .into_iter()
        .map(|(_, record)| record)
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&output, format!("{output_text}\n"))
        .unwrap_or_else(|error| panic!("write {output}: {error}"));
    println!(
        "{{\"games\":{},\"workers\":{},\"output\":\"{}\",\"depth\":{},\"nodes\":{},\"beam\":{},\"deadlineMs\":{}}}",
        games,
        workers,
        output.replace('\\', "\\\\").replace('"', "\\\""),
        depth,
        max_nodes,
        beam_width,
        deadline_ms,
    );
}
