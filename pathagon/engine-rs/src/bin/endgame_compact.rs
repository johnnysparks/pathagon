//! Convert an endgame graph to deterministic PGGRF01 binary or graph-only
//! JSONL evidence.
//!
//! The compact graph is intended for ignored research workspace artifacts and
//! is accepted directly by `pathagon-endgame-tablebase`. JSONL output is the
//! reversible inspection path for compact input.

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use pathagon_engine::tablebase::read_nodes;
use serde_json::json;

fn main() {
    let args = parse_args();
    let input = required(&args, "input");
    let output = required(&args, "out");
    let format = args.get("format").map(String::as_str).unwrap_or("compact");
    if !matches!(format, "compact" | "jsonl") {
        fail("--format must be compact or jsonl");
    }
    if input == output {
        fail("--input and --out must be different paths");
    }
    let metadata = args
        .get("metadata")
        .map(PathBuf::from)
        .unwrap_or_else(|| output.with_extension("meta.json"));
    let graph =
        read_nodes(&input).unwrap_or_else(|error| fail(&format!("cannot read graph: {error}")));
    if format == "compact" {
        graph
            .write_compact_graph(&output)
            .unwrap_or_else(|error| fail(&format!("cannot write compact graph: {error}")));
    } else {
        graph
            .write_jsonl(&output)
            .unwrap_or_else(|error| fail(&format!("cannot write JSONL graph: {error}")));
    }

    if let Some(parent) = metadata
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| fail(&format!("cannot create metadata directory: {error}")));
    }
    let metadata_value = json!({
        "schemaVersion": 1,
        "tableFamily": "pathagon-retrograde-wdl-v1",
        "format": if format == "compact" {
            "compact-graph-v1"
        } else {
            "graph-jsonl-v1"
        },
        "graphPath": output,
        "nodes": graph.len(),
        "edges": graph.edge_count(),
        "actionEncoding": if format == "compact" {
            "two corpus alphabet characters packed into u16 little-endian"
        } else {
            "JSON strings"
        },
        "unknownEncoding": "incomplete nodes remain present and are solved as unknown",
    });
    fs::write(
        &metadata,
        serde_json::to_vec_pretty(&metadata_value).expect("serialize compact graph metadata"),
    )
    .unwrap_or_else(|error| fail(&format!("cannot write metadata: {error}")));
    println!("{metadata_value}");
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
    eprintln!("pathagon-endgame-compact: {message}");
    std::process::exit(2);
}
