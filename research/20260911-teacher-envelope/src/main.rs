use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

use pathagon_engine::corpus::{decode_action, decode_state, encode_action};
use pathagon_engine::search::{
    search_best_action_with_tactical_filter, EvaluationWeights, SearchConfig,
};
use serde_json::{json, Value};

#[derive(Clone, Copy)]
struct Budget {
    name: &'static str,
    depth: u8,
    nodes: u64,
    beam: usize,
}

const BUDGETS: [Budget; 3] = [
    Budget {
        name: "2k",
        depth: 3,
        nodes: 2_000,
        beam: 128,
    },
    Budget {
        name: "8k",
        depth: 3,
        nodes: 8_000,
        beam: 128,
    },
    Budget {
        name: "32k",
        depth: 4,
        nodes: 32_000,
        beam: 256,
    },
];

fn required_arg(name: &str) -> String {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == name {
            return args
                .next()
                .unwrap_or_else(|| panic!("missing value for {name}"));
        }
    }
    panic!("missing {name}");
}

fn weights() -> EvaluationWeights {
    EvaluationWeights {
        path: 241,
        material: 112,
        capture: 887,
        structure: 40,
        threat: 154,
        edge: 74,
    }
}

fn string_field<'a>(row: &'a Value, key: &str) -> &'a str {
    row.get(key)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("missing string field {key}"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = required_arg("--targets");
    let output = required_arg("--output");
    let reader = BufReader::new(File::open(&input)?);
    let mut roots = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        roots.push(serde_json::from_str::<Value>(&line)?);
    }
    let mut reports = Vec::new();
    let mut summary: BTreeMap<&str, BTreeMap<&str, u64>> = BTreeMap::new();
    for row in &roots {
        let state = decode_state(string_field(row, "state"))?;
        let teacher_best = decode_action(string_field(row, "teacherBest"))?;
        let candidate_tokens = row
            .get("candidateActions")
            .and_then(Value::as_array)
            .unwrap();
        let candidate_actions = candidate_tokens
            .iter()
            .map(|token| decode_action(token.as_str().unwrap()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut budget_reports = Vec::new();
        for budget in BUDGETS {
            let result = search_best_action_with_tactical_filter(
                state,
                SearchConfig {
                    depth: budget.depth,
                    max_nodes: budget.nodes,
                    beam_width: budget.beam,
                    weights: weights(),
                    tactical_proof_horizon: None,
                },
            );
            let action = result.action;
            let teacher_match = action == Some(teacher_best);
            let candidate_rank = action
                .and_then(|value| {
                    candidate_actions
                        .iter()
                        .position(|candidate| *candidate == value)
                })
                .map(|index| index + 1);
            let key = budget.name;
            let counts = summary.entry(key).or_default();
            *counts.entry("roots").or_default() += 1;
            *counts.entry("teacherMatch").or_default() += u64::from(teacher_match);
            *counts.entry("candidateHit").or_default() += u64::from(candidate_rank.is_some());
            *counts.entry("completedDepthSum").or_default() += u64::from(result.completed_depth);
            *counts.entry("nodesSum").or_default() += result.nodes;
            budget_reports.push(json!({
                "budget": key,
                "action": action.map(encode_action),
                "teacherBest": encode_action(teacher_best),
                "teacherMatch": teacher_match,
                "candidateRank": candidate_rank,
                "score": result.score,
                "nodes": result.nodes,
                "completedDepth": result.completed_depth,
            }));
        }
        reports.push(json!({
            "id": string_field(row, "id"),
            "phase": string_field(row, "phase"),
            "turn": string_field(row, "turn"),
            "targetClass": string_field(row, "targetClass"),
            "teacherMargin": row.get("teacherMargin").and_then(Value::as_i64),
            "budgets": budget_reports,
        }));
    }
    let aggregate = summary
        .into_iter()
        .map(|(name, values)| {
            let roots = values["roots"] as f64;
            let mut object = serde_json::Map::new();
            object.insert("roots".to_owned(), json!(values["roots"]));
            object.insert("teacherMatch".to_owned(), json!(values["teacherMatch"]));
            object.insert(
                "teacherMatchRate".to_owned(),
                json!(values["teacherMatch"] as f64 / roots),
            );
            object.insert("candidateHit".to_owned(), json!(values["candidateHit"]));
            object.insert(
                "candidateHitRate".to_owned(),
                json!(values["candidateHit"] as f64 / roots),
            );
            object.insert(
                "meanCompletedDepth".to_owned(),
                json!(values["completedDepthSum"] as f64 / roots),
            );
            object.insert(
                "meanNodes".to_owned(),
                json!(values["nodesSum"] as f64 / roots),
            );
            (name.to_owned(), Value::Object(object))
        })
        .collect::<serde_json::Map<_, _>>();
    let result = json!({
        "schemaVersion": 1,
        "mode": "teacher-quality-at-deployment-envelope",
        "targets": input,
        "roots": reports.len(),
        "aggregate": aggregate,
        "rootsReport": reports,
    });
    std::fs::write(&output, serde_json::to_string_pretty(&result)? + "\n")?;
    println!(
        "{}",
        serde_json::to_string(&json!({"roots": roots.len(), "output": output}))?
    );
    Ok(())
}
