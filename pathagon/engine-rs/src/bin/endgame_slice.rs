//! Extract a deterministic root-reachable subgraph for bounded endgame work.
//!
//! Slices are useful when a full frontier contains many unrelated roots. A
//! missing child remains missing/unknown; the command does not manufacture a
//! closed proof boundary.

use std::collections::{BTreeSet, HashMap};
use std::env;
use std::fs;
use std::path::PathBuf;

use pathagon_engine::tablebase::read_nodes;

fn main() {
    let args = parse_args();
    let input = required(&args, "input");
    let output = required(&args, "out");
    let roots_path = required(&args, "roots");
    let format = args.get("format").map(String::as_str).unwrap_or("compact");
    if !matches!(format, "compact" | "jsonl") {
        fail("--format must be compact or jsonl");
    }
    if input == output {
        fail("--input and --out must be different paths");
    }
    let roots = fs::read_to_string(&roots_path)
        .unwrap_or_else(|error| fail(&format!("cannot read roots: {error}")))
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(ToOwned::to_owned)
        .collect::<BTreeSet<_>>();
    let graph =
        read_nodes(&input).unwrap_or_else(|error| fail(&format!("cannot read graph: {error}")));
    let slice = graph
        .slice_from_roots(&roots)
        .unwrap_or_else(|error| fail(&format!("cannot slice graph: {error}")));
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| fail(&format!("cannot create output directory: {error}")));
    }
    if format == "compact" {
        slice
            .write_compact_graph(&output)
            .unwrap_or_else(|error| fail(&format!("cannot write compact slice: {error}")));
    } else {
        slice
            .write_jsonl(&output)
            .unwrap_or_else(|error| fail(&format!("cannot write JSONL slice: {error}")));
    }
    println!(
        "{}",
        serde_json::json!({
            "schemaVersion": 1,
            "tableFamily": "pathagon-retrograde-wdl-v1",
            "input": input,
            "out": output,
            "roots": roots.len(),
            "nodes": slice.len(),
            "edges": slice.edge_count(),
            "format": format,
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
    eprintln!("pathagon-endgame-slice: {message}");
    std::process::exit(2);
}
