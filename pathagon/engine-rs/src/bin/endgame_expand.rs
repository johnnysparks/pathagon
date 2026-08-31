//! Materialize legal forward edges for canonical-key graph stubs.
//!
//! The replay-ring exporter deliberately writes unknown child stubs without a
//! duplicated state payload. This executable expands those stubs in bounded,
//! repeatable passes: decode the canonical key, apply every legal Rust action,
//! canonicalize each child, and retain exact terminal detection. An omitted or
//! unexpanded stub remains incomplete and therefore unknown to the tablebase.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::PathBuf;

use pathagon_engine::corpus::encode_action;
use pathagon_engine::golden::{canonical_position_key, decode_canonical_position_key};
use pathagon_engine::tablebase::{read_nodes, RetrogradeGraph, RetrogradeNode};
use pathagon_engine::{has_winning_path, GameState};
use serde_json::{json, Map, Value};

const BOARD_SIZE: u8 = 7;
const RESERVE_PER_PLAYER: u8 = 14;

fn main() {
    let args = parse_args();
    let input = required(&args, "input");
    let output = required(&args, "out");
    let max_expand = number(&args, "max-expand");
    let format = args.get("format").map(String::as_str).unwrap_or("jsonl");
    let focus_roots = args.get("focus-roots").map(PathBuf::from);
    if !matches!(format, "jsonl" | "compact") {
        fail("--format must be jsonl or compact");
    }
    if input == output {
        fail("--input and --out must be different paths");
    }
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| fail(&format!("cannot create output directory: {error}")));
    }
    if format == "compact" || is_compact_graph(&input) {
        run_graph_mode(&input, &output, max_expand, format, focus_roots.as_ref());
        return;
    }
    let reader = BufReader::new(
        File::open(&input).unwrap_or_else(|error| fail(&format!("cannot open input: {error}"))),
    );
    let file = File::create(&output)
        .unwrap_or_else(|error| fail(&format!("cannot create output: {error}")));
    let mut writer = BufWriter::new(file);
    let mut records = 0_usize;
    let mut expanded = 0_usize;
    let mut terminalized = 0_usize;
    let mut edges = 0_usize;
    let mut remaining_unknown = 0_usize;
    let mut known_keys = BTreeSet::new();
    let mut new_child_keys = BTreeSet::new();

    for (line_number, line) in reader.lines().enumerate() {
        let line = line.unwrap_or_else(|error| fail(&format!("cannot read input line: {error}")));
        if line.trim().is_empty() {
            continue;
        }
        records += 1;
        let mut record: Value = serde_json::from_str(&line).unwrap_or_else(|error| {
            fail(&format!(
                "invalid JSON at line {}: {error}",
                line_number + 1
            ))
        });
        if let Some(key) = record.get("key").and_then(Value::as_str) {
            known_keys.insert(key.to_owned());
        }
        let expandable = record.get("complete").and_then(Value::as_bool) == Some(false)
            && record.get("seed").is_none_or(|seed| seed.is_null());
        if expandable && (max_expand == 0 || expanded < max_expand) {
            let (was_terminal, generated_edges, generated_children) = expand_record(&mut record)
                .unwrap_or_else(|error| fail(&format!("line {}: {error}", line_number + 1)));
            expanded += 1;
            edges += generated_edges;
            terminalized += usize::from(was_terminal);
            new_child_keys.extend(generated_children);
        } else if expandable {
            remaining_unknown += 1;
        }
        serde_json::to_writer(&mut writer, &record)
            .unwrap_or_else(|error| fail(&format!("cannot write output: {error}")));
        writer
            .write_all(b"\n")
            .unwrap_or_else(|error| fail(&format!("cannot write output: {error}")));
    }
    let mut appended_stubs = 0_usize;
    for key in new_child_keys {
        if known_keys.contains(&key) {
            continue;
        }
        let record = json!({
            "schemaVersion": 2,
            "tableFamily": "pathagon-retrograde-wdl-v1",
            "ring": 0,
            "key": key,
            "children": [],
            "complete": false,
            "terminal": null,
            "seed": null,
            "actions": [],
            "proof": {
                "kind": "unknown-child-stub",
                "rulesVersion": "pathagon-rules-v1",
                "solverVersion": "pathagon-endgame-expander-v1",
                "lineage": "verified-parent-forward-edge-expansion",
            },
        });
        serde_json::to_writer(&mut writer, &record)
            .unwrap_or_else(|error| fail(&format!("cannot write appended stub: {error}")));
        writer
            .write_all(b"\n")
            .unwrap_or_else(|error| fail(&format!("cannot write appended stub: {error}")));
        appended_stubs += 1;
    }
    writer
        .flush()
        .unwrap_or_else(|error| fail(&format!("cannot flush output: {error}")));
    println!(
        "{}",
        json!({
                "schemaVersion": 1,
                "tableFamily": "pathagon-retrograde-wdl-v1",
                "input": input,
                "out": output,
                "records": records,
                "expanded": expanded,
                "terminalized": terminalized,
                "completeForwardEdges": edges,
                "remainingUnexpandedUnknownStubs": remaining_unknown,
            "appendedUnknownStubs": appended_stubs,
            "outputRecords": records + appended_stubs,
            "maxExpand": max_expand,
            "format": format,
            "status": "pass",
        })
    );
}

fn run_graph_mode(
    input: &PathBuf,
    output: &PathBuf,
    max_expand: usize,
    format: &str,
    focus_roots: Option<&PathBuf>,
) {
    let graph =
        read_nodes(input).unwrap_or_else(|error| fail(&format!("cannot read graph: {error}")));
    let mut nodes = graph.into_nodes();
    let records = nodes.len();
    let mut expanded = 0_usize;
    let mut terminalized = 0_usize;
    let mut edges = 0_usize;
    let mut remaining_unknown = 0_usize;
    let known_keys = nodes
        .iter()
        .map(|node| node.key.clone())
        .collect::<BTreeSet<_>>();
    let mut new_child_keys = BTreeSet::new();

    let expansion_order = focus_roots
        .map(|path| prioritized_indices(&nodes, path))
        .transpose()
        .unwrap_or_else(|error| fail(&format!("cannot read focus roots: {error}")))
        .unwrap_or_else(|| (0..nodes.len()).collect());
    let expandable = nodes
        .iter()
        .enumerate()
        .filter_map(|(index, node)| (!node.complete && node.seed.is_none()).then_some(index))
        .collect::<BTreeSet<_>>();
    let expansion_budget = if max_expand == 0 {
        usize::MAX
    } else {
        max_expand
    };
    let selected = expansion_order
        .iter()
        .copied()
        .filter(|index| expandable.contains(index))
        .take(expansion_budget)
        .collect::<BTreeSet<_>>();

    for (index, node) in nodes.iter_mut().enumerate() {
        if selected.contains(&index) {
            let mut record = serde_json::to_value(&*node)
                .unwrap_or_else(|error| fail(&format!("cannot encode graph node: {error}")));
            let (was_terminal, generated_edges, generated_children) = expand_record(&mut record)
                .unwrap_or_else(|error| fail(&format!("cannot expand {}: {error}", node.key)));
            *node = serde_json::from_value(record)
                .unwrap_or_else(|error| fail(&format!("cannot decode expanded node: {error}")));
            expanded += 1;
            edges += generated_edges;
            terminalized += usize::from(was_terminal);
            new_child_keys.extend(generated_children);
        } else if expandable.contains(&index) {
            remaining_unknown += 1;
        }
    }

    let mut appended_stubs = 0_usize;
    for key in new_child_keys {
        if known_keys.contains(&key) {
            continue;
        }
        nodes.push(RetrogradeNode {
            key,
            children: Vec::new(),
            complete: false,
            terminal: None,
            seed: None,
            actions: Vec::new(),
        });
        appended_stubs += 1;
    }
    let mut output_graph = RetrogradeGraph::default();
    for node in nodes {
        output_graph
            .insert(node)
            .unwrap_or_else(|error| fail(&format!("cannot assemble expanded graph: {error}")));
    }
    if format == "compact" {
        output_graph
            .write_compact_graph(output)
            .unwrap_or_else(|error| fail(&format!("cannot write compact graph: {error}")));
    } else {
        output_graph
            .write_jsonl(output)
            .unwrap_or_else(|error| fail(&format!("cannot write JSONL graph: {error}")));
    }
    println!(
        "{}",
        json!({
            "schemaVersion": 1,
            "tableFamily": "pathagon-retrograde-wdl-v1",
            "input": input,
            "out": output,
            "records": records,
            "expanded": expanded,
            "terminalized": terminalized,
            "completeForwardEdges": edges,
            "remainingUnexpandedUnknownStubs": remaining_unknown,
            "appendedUnknownStubs": appended_stubs,
            "outputRecords": output_graph.len(),
            "maxExpand": max_expand,
            "format": format,
            "focusRoots": focus_roots,
            "status": "pass",
        })
    );
}

fn prioritized_indices(
    nodes: &[RetrogradeNode],
    roots_path: &PathBuf,
) -> Result<Vec<usize>, String> {
    let node_indices = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.key.as_str(), index))
        .collect::<BTreeMap<_, _>>();
    let source = fs::read_to_string(roots_path)
        .map_err(|error| format!("{}: {error}", roots_path.display()))?;
    let roots = source
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(ToOwned::to_owned)
        .collect::<BTreeSet<_>>();
    if roots.is_empty() {
        return Err("focus root file contains no keys".to_owned());
    }
    let mut distances = BTreeMap::new();
    let mut queue = VecDeque::new();
    for root in roots {
        let Some(&index) = node_indices.get(root.as_str()) else {
            return Err(format!("focus root {root} is absent from the graph"));
        };
        if distances.insert(index, 0_usize).is_none() {
            queue.push_back(index);
        }
    }
    while let Some(index) = queue.pop_front() {
        let distance = distances[&index];
        for child in &nodes[index].children {
            let Some(&child_index) = node_indices.get(child.as_str()) else {
                continue;
            };
            if distances.insert(child_index, distance + 1).is_none() {
                queue.push_back(child_index);
            }
        }
    }

    let mut order = (0..nodes.len()).collect::<Vec<_>>();
    order.sort_by_key(|index| {
        (
            distances.get(index).copied().unwrap_or(usize::MAX),
            nodes[*index].key.clone(),
        )
    });
    Ok(order)
}

fn is_compact_graph(path: &PathBuf) -> bool {
    let Ok(mut file) = File::open(path) else {
        return false;
    };
    let mut magic = [0_u8; 8];
    file.read_exact(&mut magic).is_ok() && magic == *b"PGGRF01\0"
}

fn expand_record(record: &mut Value) -> Result<(bool, usize, BTreeSet<String>), String> {
    let object = record
        .as_object_mut()
        .ok_or_else(|| "graph record must be a JSON object".to_owned())?;
    let key = object
        .get("key")
        .and_then(Value::as_str)
        .ok_or_else(|| "unknown stub is missing a key".to_owned())?
        .to_owned();
    let key_bytes = decode_hex(&key)?;
    let state = decode_canonical_position_key(&key_bytes, BOARD_SIZE, RESERVE_PER_PLAYER)?;
    if canonical_position_key(state) != key_bytes {
        return Err(format!("key {key} is not a canonical representative"));
    }
    object.insert("position".to_owned(), position_json(state));
    object.insert(
        "proof".to_owned(),
        json!({
            "kind": "canonical-key-forward-expansion",
            "rulesVersion": "pathagon-rules-v1",
            "solverVersion": "pathagon-endgame-expander-v1",
            "lineage": "verified-parent-forward-edge-expansion",
        }),
    );

    let current_path = has_winning_path(state, state.turn);
    let previous_path = has_winning_path(state, state.turn.other());
    if current_path && previous_path {
        return Err(format!("key {key} contains winning paths for both players"));
    }
    if current_path {
        return Err(format!(
            "key {key} has a winning path for the side to move and is not a reachable child"
        ));
    }
    if previous_path {
        set_edges(object, BTreeMap::new(), BTreeSet::new());
        object.insert("complete".to_owned(), Value::Bool(true));
        object.insert("terminal".to_owned(), Value::String("loss".to_owned()));
        return Ok((true, 0, BTreeSet::new()));
    }

    let legal_actions = state.legal_actions();
    if legal_actions.is_empty() {
        set_edges(object, BTreeMap::new(), BTreeSet::new());
        object.insert("complete".to_owned(), Value::Bool(true));
        object.insert("terminal".to_owned(), Value::String("draw".to_owned()));
        return Ok((true, 0, BTreeSet::new()));
    }

    let mut actions = BTreeMap::new();
    let mut children = BTreeSet::new();
    for action in legal_actions {
        let child = state.apply_legal(action).state;
        let child_key = hex_key(&canonical_position_key(child));
        children.insert(child_key.clone());
        actions.insert(encode_action(action), child_key);
    }
    let edge_count = actions.len();
    let generated_children = children.clone();
    set_edges(object, actions, children);
    object.insert("complete".to_owned(), Value::Bool(true));
    object.insert("terminal".to_owned(), Value::Null);
    Ok((false, edge_count, generated_children))
}

fn set_edges(
    object: &mut Map<String, Value>,
    actions: BTreeMap<String, String>,
    children: BTreeSet<String>,
) {
    object.insert(
        "children".to_owned(),
        Value::Array(children.into_iter().map(Value::String).collect()),
    );
    object.insert(
        "actions".to_owned(),
        Value::Array(
            actions
                .into_iter()
                .map(|(action, child)| json!({"action": action, "child": child}))
                .collect(),
        ),
    );
}

fn position_json(state: GameState) -> Value {
    json!({
        "boardSize": state.config.board_size,
        "reservePerPlayer": state.config.reserve_per_player,
        "light": state.light,
        "dark": state.dark,
        "reserve": state.reserve,
        "turn": state.turn.as_str(),
        "forbidden": state.forbidden,
        "lastRelocatedTo": state.last_relocated_to,
        "ply": state.ply,
    })
}

fn decode_hex(value: &str) -> Result<Vec<u8>, String> {
    if value.len() % 2 != 0 {
        return Err("canonical key must contain an even number of hex digits".to_owned());
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let high = hex_digit(pair[0])?;
            let low = hex_digit(pair[1])?;
            Ok((high << 4) | low)
        })
        .collect()
}

fn hex_digit(value: u8) -> Result<u8, String> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err("canonical key contains a non-hex digit".to_owned()),
    }
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

fn number(args: &HashMap<String, String>, key: &str) -> usize {
    args.get(key)
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| fail(&format!("--{key} must be a non-negative integer")))
        })
        .unwrap_or(0)
}

fn hex_key(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut key = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        key.push(HEX[(byte >> 4) as usize] as char);
        key.push(HEX[(byte & 0x0f) as usize] as char);
    }
    key
}

fn fail(message: &str) -> ! {
    eprintln!("pathagon-endgame-expand: {message}");
    std::process::exit(2);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_a_canonical_state_using_the_rust_legal_action_boundary() {
        let state = GameState::new();
        let key = hex_key(&canonical_position_key(state));
        let mut record = json!({
            "key": key,
            "complete": false,
            "seed": null,
        });
        let (terminal, edges, children) = expand_record(&mut record).expect("expand start state");
        assert!(!terminal);
        assert_eq!(edges, 49);
        assert_eq!(children.len(), 16);
        assert_eq!(record["complete"], true);
        assert_eq!(record["terminal"], Value::Null);
        assert_eq!(record["actions"].as_array().expect("actions").len(), 49);
        // Forty-nine legal actions collapse to sixteen canonical child keys
        // under the board symmetries.
        assert_eq!(record["children"].as_array().expect("children").len(), 16);
        assert!(record["position"].is_object());
    }

    #[test]
    fn focus_order_walks_roots_before_descendants() {
        let nodes = vec![
            RetrogradeNode {
                key: "0000".to_owned(),
                children: vec!["0001".to_owned(), "0002".to_owned()],
                complete: true,
                terminal: None,
                seed: None,
                actions: Vec::new(),
            },
            RetrogradeNode {
                key: "0001".to_owned(),
                children: vec!["0003".to_owned()],
                complete: true,
                terminal: None,
                seed: None,
                actions: Vec::new(),
            },
            RetrogradeNode {
                key: "0002".to_owned(),
                children: Vec::new(),
                complete: false,
                terminal: None,
                seed: None,
                actions: Vec::new(),
            },
            RetrogradeNode {
                key: "0003".to_owned(),
                children: Vec::new(),
                complete: false,
                terminal: None,
                seed: None,
                actions: Vec::new(),
            },
        ];
        let path =
            std::env::temp_dir().join(format!("pathagon-focus-roots-{}.txt", std::process::id()));
        std::fs::write(&path, "0000\n").unwrap();
        let order = prioritized_indices(&nodes, &path).unwrap();
        assert_eq!(order, vec![0, 1, 2, 3]);
        let _ = std::fs::remove_file(path);
    }
}
