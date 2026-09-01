//! Compare a learned narrow-root sorter with direct Pathfinder.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::sync::{Arc, Mutex};
use std::thread;

use pathagon_engine::search::{EvaluationWeights, SearchConfig};
use pathagon_engine::selfplay::{play_game, Agent, MatchOptions};
use pathagon_engine::transition_policy::TransitionPolicyModel;

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

fn number<T: std::str::FromStr>(args: &BTreeMap<String, String>, key: &str, default: T) -> T {
    args.get(key)
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let args = args();
    let model_path = required(&args, "model");
    let output = required(&args, "output");
    let candidate_id = args
        .get("candidate-id")
        .cloned()
        .unwrap_or_else(|| "pathfinder-d5-b256-500k-sort-selection".to_owned());
    let games = number(&args, "games", 20_usize);
    let workers = number(&args, "workers", 8_usize).max(1);
    let seed = number(&args, "seed", 20_260_831_u32);
    let max_plies = number(&args, "max-plies", 80_u16);
    let opening_random_plies = number(&args, "opening-random-plies", 2_u16);
    let depth = number(&args, "depth", 5_u8);
    let nodes = number(&args, "nodes", 500_000_u64);
    let beam = number(&args, "beam", 256_usize);
    let top_k = number(&args, "top-k", 32_usize);
    let root_limit = number(&args, "root-limit", top_k);
    let deadline = number(&args, "deadline-ms", 5_000_u32);
    if games == 0 || nodes == 0 || beam == 0 || top_k == 0 || root_limit == 0 || deadline == 0 {
        panic!("games, search budgets, sorter limits, and deadline must be positive");
    }
    let model = Arc::new(
        TransitionPolicyModel::from_path(std::path::Path::new(&model_path))
            .unwrap_or_else(|error| panic!("load transition policy: {error}")),
    );
    let config = SearchConfig {
        depth,
        max_nodes: nodes,
        beam_width: beam,
        weights: EvaluationWeights::default(),
        tactical_proof_horizon: None,
    };
    let candidate = Agent::transition_policy_sorter_with_deadline(
        &candidate_id,
        config,
        top_k,
        root_limit,
        model,
        deadline,
    );
    let incumbent = Agent::search_with_deadline("pathfinder-direct-d5-b256-500k", config, deadline);
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
    let text = records
        .into_iter()
        .map(|(_, record)| record)
        .collect::<Vec<_>>()
        .join("\n");
    if let Some(parent) = std::path::Path::new(&output).parent() {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| panic!("create arena directory: {error}"));
    }
    fs::write(&output, format!("{text}\n"))
        .unwrap_or_else(|error| panic!("write {output}: {error}"));
    println!(
        "{{\"candidate\":\"{}\",\"games\":{},\"workers\":{},\"output\":\"{}\",\"depth\":{},\"nodes\":{},\"beam\":{},\"topK\":{},\"rootLimit\":{},\"deadlineMs\":{}}}",
        candidate_id,
        games,
        workers,
        output.replace('\\', "\\\\").replace('"', "\\\""),
        depth,
        nodes,
        beam,
        top_k,
        root_limit,
        deadline,
    );
}
