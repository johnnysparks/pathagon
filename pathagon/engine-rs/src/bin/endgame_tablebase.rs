//! Solve a persisted, canonical-keyed endgame graph with exact/unknown rules.
//!
//! Input is JSONL `RetrogradeNode` records. The generator that creates the
//! graph owns legal-state construction; this executable owns propagation,
//! deterministic output, and checkpoint evidence.

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

use pathagon_engine::tablebase::{read_checkpoint, read_nodes};

fn main() {
    let args = parse_args();
    let input = required(&args, "input");
    let output = required(&args, "out");
    let checkpoint = args.get("checkpoint").map(PathBuf::from);
    let resume = args.get("resume").map(PathBuf::from);
    let shard_directory = args.get("shards").map(PathBuf::from);
    let shard_count = args
        .get("shard-count")
        .map(|value| {
            value
                .parse::<usize>()
                .unwrap_or_else(|_| fail("--shard-count must be a positive integer"))
        })
        .unwrap_or(1);
    if shard_count == 0 {
        fail("--shard-count must be a positive integer");
    }
    let graph =
        read_nodes(&input).unwrap_or_else(|error| fail(&format!("cannot read graph: {error}")));
    let seed = resume
        .as_ref()
        .map(|path| {
            read_checkpoint(path)
                .unwrap_or_else(|error| fail(&format!("cannot read checkpoint: {error}")))
                .values
        })
        .unwrap_or_default();
    let (values, stats) = graph
        .solve_from_seed(&seed)
        .unwrap_or_else(|error| fail(&format!("cannot resume tablebase: {error}")));
    let wrote_shards = shard_directory.is_some();
    graph
        .write_values(&output, &values, stats)
        .unwrap_or_else(|error| fail(&format!("cannot write tablebase: {error}")));
    if let Some(directory) = shard_directory.as_ref() {
        graph
            .write_value_shards(directory, &values, stats, shard_count)
            .unwrap_or_else(|error| fail(&format!("cannot write value shards: {error}")));
    }
    if let Some(checkpoint) = checkpoint {
        graph
            .write_checkpoint(checkpoint, stats.rounds, stats, &values)
            .unwrap_or_else(|error| fail(&format!("cannot write checkpoint: {error}")));
    }
    println!(
        "{}",
        serde_json::json!({
            "schemaVersion": 1,
            "tableFamily": "pathagon-retrograde-wdl-v1",
            "input": input,
            "out": output,
            "nodes": stats.nodes,
            "edges": stats.edges,
            "rounds": stats.rounds,
            "solved": stats.solved,
            "draws": stats.draws,
            "unknown": stats.unknown,
            "resumed": !seed.is_empty(),
            "shardCount": wrote_shards.then_some(shard_count),
        })
    );
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

fn fail(message: &str) -> ! {
    eprintln!("pathagon-endgame-tablebase: {message}");
    std::process::exit(2);
}
