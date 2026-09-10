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
        .unwrap_or_else(|| "pathfinder-action-transition-policy-v1-research".to_owned());
    let games = number(&args, "games", 20_usize);
    let workers = number(&args, "workers", 4_usize).max(1);
    let seed = number(&args, "seed", 20_260_829_u32);
    let max_plies = number(&args, "max-plies", 100_u16);
    let opening_random_plies = number(&args, "opening-random-plies", 2_u16);
    let depth = number(&args, "depth", 7_u8);
    let nodes = number(&args, "nodes", 1_000_000_u64);
    let beam = number(&args, "beam", 32_usize);
    let deadline = number(&args, "deadline-ms", 2_800_u32);
    let model = Arc::new(
        TransitionPolicyModel::from_path(std::path::Path::new(&model_path))
            .unwrap_or_else(|error| panic!("load transition policy: {error}")),
    );
    let weights = EvaluationWeights {
        path: 241,
        material: 112,
        capture: 887,
        structure: 40,
        threat: 154,
        edge: 74,
    };
    let config = SearchConfig {
        depth,
        max_nodes: nodes,
        beam_width: beam,
        weights,
        tactical_proof_horizon: None,
    };
    let candidate = Agent::TransitionPolicy { id: candidate_id, config, model, deadline_ms: (deadline > 0).then_some(deadline) };
    let incumbent = if let Some(path) = args.get("opponent-model") {
        Agent::TransitionPolicy { id: "reference-model".to_owned(), config,
            model: Arc::new(TransitionPolicyModel::from_path(std::path::Path::new(path)).unwrap()),
            deadline_ms: (deadline > 0).then_some(deadline) }
    } else { Agent::SearchTacticalFilter { id: "unguided-control".to_owned(), config,
        deadline_ms: (deadline > 0).then_some(deadline) } };
    let independent_seeds = args.contains_key("independent-seeds");
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
                seed: seed.wrapping_add((if independent_seeds { index } else { index / 2 }) as u32),
                max_plies,
                opening_random_plies,
                ..MatchOptions::default()
            };
            let started = std::time::Instant::now();
            let record = if index % 2 == 0 {
                play_game(&candidate, &incumbent, options)
            } else {
                play_game(&incumbent, &candidate, options)
            };
            let mut value: serde_json::Value = serde_json::from_str(&record.to_json()).unwrap();
            value["researchWallSeconds"] = serde_json::json!(started.elapsed().as_secs_f64());
            eprintln!("game {} complete: {} plies, {}", index, record.moves.len(), record.reason.as_str());
            records
                .lock()
                .expect("arena records lock")
                .push((index, value.to_string()));
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
    fs::write(&output, format!("{text}\n"))
        .unwrap_or_else(|error| panic!("write {output}: {error}"));
    println!(
        "{{\"games\":{},\"workers\":{},\"output\":\"{}\",\"depth\":{},\"nodes\":{},\"beam\":{},\"deadlineMs\":{}}}",
        games,
        workers,
        output.replace('\\', "\\\\").replace('"', "\\\""),
        depth,
        nodes,
        beam,
        deadline,
    );
}
