use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::time::Instant;

use pathagon_engine::search::{EvaluationWeights, SearchConfig};
use pathagon_engine::training::{
    parse_weights_spec, train, write_training_output, Champion, TrainingConfig,
};

fn main() {
    let args = parse_args();
    let defaults = TrainingConfig::default();
    let weights = EvaluationWeights::default();
    let initial = initial_champion(&args, weights);
    let config = TrainingConfig {
        generations: number(&args, "generations", defaults.generations),
        population: number(&args, "population", defaults.population),
        training_pairs: number(&args, "training-pairs", defaults.training_pairs),
        evaluation_pairs: number(&args, "evaluation-pairs", defaults.evaluation_pairs),
        seed: number(&args, "seed", defaults.seed),
        mutation_per_mille: number(&args, "mutation-per-mille", defaults.mutation_per_mille),
        promotion_rate_per_mille: number(
            &args,
            "promotion-rate-per-mille",
            defaults.promotion_rate_per_mille,
        ),
        max_plies: number(&args, "max-plies", defaults.max_plies),
        opening_random_plies: number(&args, "opening-random-plies", defaults.opening_random_plies),
        tactical_filter: args.contains_key("tactical-filter"),
        search: SearchConfig {
            depth: number(&args, "depth", defaults.search.depth),
            max_nodes: number(&args, "nodes", defaults.search.max_nodes),
            beam_width: number(&args, "beam", defaults.search.beam_width),
            weights: initial.weights,
            tactical_proof_horizon: None,
        },
    };
    let output = PathBuf::from(
        args.get("out")
            .map(String::as_str)
            .unwrap_or("work/rust-v1"),
    );
    let started = Instant::now();
    let result = train(initial, config);
    let written = write_training_output(&output, &result)
        .unwrap_or_else(|error| fail(&format!("cannot write training output: {error}")));
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "{{\"schemaVersion\":1,\"seed\":{},\"champion\":\"{}\",\"generation\":{},\"promotions\":{},\"trials\":{},\"trainingGames\":{},\"evaluationGames\":{},\"trainingPositions\":{},\"evaluationPositions\":{},\"seconds\":{:.6}}}",
        config.seed,
        result.champion.id,
        result.champion.generation,
        result.trials.iter().filter(|trial| trial.promoted).count(),
        result.trials.len(),
        written.training.games,
        written.evaluation.games,
        written.training.positions,
        written.evaluation.positions,
        elapsed,
    );
}

fn initial_champion(args: &HashMap<String, String>, fallback: EvaluationWeights) -> Champion {
    if let Some(path) = args.get("initial-manifest") {
        if args.contains_key("initial-weights") {
            fail("--initial-manifest and --initial-weights are mutually exclusive");
        }
        return Champion::from_manifest_file(Path::new(path))
            .unwrap_or_else(|error| fail(&format!("cannot load initial evaluator: {error}")));
    }

    if let Some(spec) = args.get("initial-weights") {
        let weights = parse_weights_spec(spec)
            .unwrap_or_else(|error| fail(&format!("cannot parse --initial-weights: {error}")));
        return Champion {
            id: args
                .get("initial-id")
                .cloned()
                .unwrap_or_else(|| "rust-explicit-initial-evaluator".to_owned()),
            generation: number(args, "initial-generation", 0_u8),
            weights,
        };
    }

    Champion::baseline(fallback)
}

fn parse_args() -> HashMap<String, String> {
    let mut parsed = HashMap::new();
    let values: Vec<String> = env::args().skip(1).collect();
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
    eprintln!("pathagon-train: {message}");
    std::process::exit(2)
}
