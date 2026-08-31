//! Deterministically merge independently written retrograde value shards.

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;

use pathagon_engine::tablebase::{read_value_shards, write_merged_values};

fn main() {
    let args = parse_args();
    let directory = required(&args, "shards");
    let output = required(&args, "out");
    let values = read_value_shards(&directory)
        .unwrap_or_else(|error| fail(&format!("cannot read shard directory: {error}")));
    write_merged_values(&output, &values)
        .unwrap_or_else(|error| fail(&format!("cannot write merged values: {error}")));
    println!(
        "{}",
        serde_json::json!({
            "schemaVersion": 1,
            "tableFamily": "pathagon-retrograde-wdl-v1",
            "shards": directory,
            "out": output,
            "values": values.len(),
            "status": "pass",
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
    eprintln!("pathagon-endgame-merge: {message}");
    std::process::exit(2);
}
