use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};

use pathagon_engine::corpus::{decode_action, decode_state, encode_action};
use pathagon_engine::search::{analyze_action, EvaluationWeights, SearchConfig};
use serde_json::{json, Value};

struct Item {
    token: String,
    score: i32,
    action_order: u16,
    old_index: usize,
    exhausted: bool,
    nodes: u64,
}

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

fn relative(scores: &[i32]) -> (Vec<f32>, f32, f32) {
    let mut sorted = scores.to_vec();
    sorted.sort_unstable();
    let center = sorted[sorted.len() / 2] as f32;
    let mut deviations = sorted
        .iter()
        .map(|score| (*score as f32 - center).abs())
        .collect::<Vec<_>>();
    deviations.sort_by(f32::total_cmp);
    let scale = (1.4826 * deviations[deviations.len() / 2]).max(250.0);
    let values = scores
        .iter()
        .map(|score| ((*score as f32 - center) / scale).clamp(-1.0, 1.0))
        .collect();
    (values, center, scale)
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights = values
        .iter()
        .map(|value| ((*value - max) / 0.35).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f32>().max(1.0e-8);
    weights.into_iter().map(|value| value / total).collect()
}

fn f32_array(row: &Value, key: &str) -> Vec<f32> {
    row.get(key)
        .and_then(Value::as_array)
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap() as f32)
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = required_arg("--input");
    let output = required_arg("--output");
    let reader = BufReader::new(File::open(&input)?);
    let mut writer = std::io::BufWriter::new(File::create(&output)?);
    let config = SearchConfig {
        depth: 4,
        max_nodes: 32_000,
        beam_width: 256,
        weights: EvaluationWeights {
            path: 241,
            material: 112,
            capture: 887,
            structure: 40,
            threat: 154,
            edge: 74,
        },
        tactical_proof_horizon: None,
    };
    let mut rows = 0_u64;
    let mut analyses = 0_u64;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let mut row: Value = serde_json::from_str(&line)?;
        let state = decode_state(row.get("state").and_then(Value::as_str).unwrap())?;
        let tokens = row
            .get("candidateActions")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .map(|value| value.as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        let old_scores = row
            .get("teacherScores")
            .and_then(Value::as_array)
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap() as i32)
            .collect::<Vec<_>>();
        let old_outcomes = f32_array(&row, "continuationValues");
        let old_short = f32_array(&row, "continuationShortValues");
        let old_mid = f32_array(&row, "continuationMidValues");
        let old_long = f32_array(&row, "continuationLongValues");
        if tokens.len() < 2
            || tokens.len() != old_scores.len()
            || old_scores.len() != old_outcomes.len()
        {
            return Err(format!(
                "{}: misaligned target vectors",
                row.get("id").and_then(Value::as_str).unwrap_or("unknown")
            )
            .into());
        }
        let mut items = Vec::with_capacity(tokens.len());
        for (index, token) in tokens.iter().enumerate() {
            let action = decode_action(token)?;
            let evaluated = analyze_action(state, action, config)?;
            items.push(Item {
                token: encode_action(action),
                score: evaluated.score,
                action_order: action.order(),
                old_index: index,
                exhausted: evaluated.exhausted,
                nodes: evaluated.nodes,
            });
            analyses += 1;
        }
        items.sort_by(|left, right| {
            right
                .score
                .cmp(&left.score)
                .then_with(|| left.action_order.cmp(&right.action_order))
        });
        let scores = items.iter().map(|item| item.score).collect::<Vec<_>>();
        let (values, center, scale) = relative(&scores);
        let policy = softmax(&values);
        let continuation = items
            .iter()
            .map(|item| old_outcomes[item.old_index])
            .collect::<Vec<_>>();
        let short = items
            .iter()
            .map(|item| old_short[item.old_index])
            .collect::<Vec<_>>();
        let mid = items
            .iter()
            .map(|item| old_mid[item.old_index])
            .collect::<Vec<_>>();
        let long = items
            .iter()
            .map(|item| old_long[item.old_index])
            .collect::<Vec<_>>();
        let expected = policy
            .iter()
            .zip(continuation.iter())
            .map(|(probability, value)| probability * value)
            .sum::<f32>();
        let source_outcome = row
            .get("sourceOutcome")
            .and_then(Value::as_i64)
            .unwrap_or(0) as f32;
        let value_target = (0.75 * expected + 0.25 * source_outcome).clamp(-1.0, 1.0);
        let margin = scores[0].saturating_sub(scores[1]);
        row["candidateActions"] = json!(items
            .iter()
            .map(|item| item.token.clone())
            .collect::<Vec<_>>());
        row["teacherScores"] = json!(scores);
        row["teacherValues"] = json!(values);
        row["teacherValueCenter"] = json!(center);
        row["teacherValueScale"] = json!(scale);
        row["teacherValueMethod"] = json!("deployment-envelope-median-mad-v1");
        row["softPolicy"] = json!(policy);
        row["continuationValues"] = json!(continuation);
        row["continuationShortValues"] = json!(short);
        row["continuationMidValues"] = json!(mid);
        row["continuationLongValues"] = json!(long);
        row["expectedContinuationValue"] = json!(expected);
        row["valueTarget"] = json!(value_target);
        row["targetClass"] = json!(if value_target.abs() <= 0.20 {
            "neutral"
        } else {
            "decisive"
        });
        row["teacherBest"] = json!(items[0].token);
        row["teacherMargin"] = json!(margin);
        row["teacherExhausted"] = json!(items.iter().filter(|item| item.exhausted).count());
        row["teacherNodes"] = json!(items.iter().map(|item| item.nodes).sum::<u64>());
        row["teacherDepth"] = json!(config.depth);
        row["teacherNodeBudget"] = json!(config.max_nodes);
        row["targetProvenance"] = json!(format!("{}:depth4-beam256-nodes32000", input));
        writeln!(writer, "{}", serde_json::to_string(&row)?)?;
        rows += 1;
    }
    writer.flush()?;
    println!(
        "{}",
        json!({"rows": rows, "actionAnalyses": analyses, "teacherDepth": 4, "teacherNodes": 32000, "output": output})
    );
    Ok(())
}
