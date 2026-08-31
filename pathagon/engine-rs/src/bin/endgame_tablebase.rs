//! Solve a persisted, canonical-keyed endgame graph with exact/unknown rules.
//!
//! Input is JSONL `RetrogradeNode` records. The generator that creates the
//! graph owns legal-state construction; this executable owns propagation,
//! deterministic output, and checkpoint evidence.

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

use pathagon_engine::tablebase::read_nodes;

fn main() {
    let args = parse_args();
    let input = required(&args, "input");
    let output = required(&args, "out");
    let checkpoint = args.get("checkpoint").map(PathBuf::from);
    let graph =
        read_nodes(&input).unwrap_or_else(|error| fail(&format!("cannot read graph: {error}")));
    let (values, stats) = graph.solve();
    graph
        .write_values(&output, &values, stats)
        .unwrap_or_else(|error| fail(&format!("cannot write tablebase: {error}")));
    if let Some(checkpoint) = checkpoint {
        graph
            .write_checkpoint(checkpoint, stats.rounds, stats)
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
