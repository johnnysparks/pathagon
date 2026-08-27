//! Chunked QAdv replay evaluator for AWS Lambda.
//!
//! The self-play Lambda evaluates one game per request. This sibling runtime
//! evaluates a bounded list of already-audited replay records, returning raw
//! metric sums so a coordinator can fan out work and aggregate without
//! averaging averages. The QAdv ONNX model is loaded once per warm runtime.

use std::env;
use std::fs;
use std::sync::{Arc, OnceLock};

use lambda_runtime::{service_fn, Error, LambdaEvent};
use serde::Deserialize;
use serde_json::{json, Value};

use pathagon_engine::inference::OnnxQAdvModel;
use pathagon_engine::{Action, BoardConfig, GameState};

#[derive(Debug, Deserialize)]
struct EvaluationJob {
    #[serde(default)]
    chunk_id: Option<String>,
    games: Vec<Value>,
}

static MODEL: OnceLock<Result<Arc<OnnxQAdvModel>, String>> = OnceLock::new();

fn model() -> Result<Arc<OnnxQAdvModel>, Error> {
    let result = MODEL.get_or_init(|| {
        let path = env::var("QADV_MODEL_PATH")
            .unwrap_or_else(|_| "/var/task/qadv-arbiter.onnx".to_owned());
        let bytes = fs::read(&path)
            .map_err(|error| format!("cannot read QAdv model {path}: {error}"))?;
        OnnxQAdvModel::from_bytes(&bytes).map(Arc::new)
    });
    match result {
        Ok(model) => Ok(Arc::clone(model)),
        Err(error) => Err(std::io::Error::other(error.clone()).into()),
    }
}

#[derive(Clone, Debug, Default)]
struct Metrics {
    positions: u64,
    visited_actions: u64,
    visited_pairs: u64,
    selected_action_is_q_max: u64,
    selected_action_q_rank: f64,
    selected_action_q_percentile: f64,
    q_spread: f64,
    q_mse: f64,
    q_mae: f64,
    q_weight: f64,
    predicted_pairwise_agreement: u64,
    predicted_pairwise_pairs: u64,
    predicted_selected_action_is_target_q_max: u64,
    predicted_selected_action_target_q_rank: f64,
}

impl Metrics {
    fn merge(&mut self, other: &Self) {
        self.positions += other.positions;
        self.visited_actions += other.visited_actions;
        self.visited_pairs += other.visited_pairs;
        self.selected_action_is_q_max += other.selected_action_is_q_max;
        self.selected_action_q_rank += other.selected_action_q_rank;
        self.selected_action_q_percentile += other.selected_action_q_percentile;
        self.q_spread += other.q_spread;
        self.q_mse += other.q_mse;
        self.q_mae += other.q_mae;
        self.q_weight += other.q_weight;
        self.predicted_pairwise_agreement += other.predicted_pairwise_agreement;
        self.predicted_pairwise_pairs += other.predicted_pairwise_pairs;
        self.predicted_selected_action_is_target_q_max +=
            other.predicted_selected_action_is_target_q_max;
        self.predicted_selected_action_target_q_rank +=
            other.predicted_selected_action_target_q_rank;
    }

    fn raw_json(&self) -> Value {
        json!({
            "positions": self.positions,
            "visitedActions": self.visited_actions,
            "visitedPairs": self.visited_pairs,
            "selectedActionIsQMax": self.selected_action_is_q_max,
            "selectedActionQRank": self.selected_action_q_rank,
            "selectedActionQPercentile": self.selected_action_q_percentile,
            "qSpread": self.q_spread,
            "qMse": self.q_mse,
            "qMae": self.q_mae,
            "qWeight": self.q_weight,
            "predictedPairwiseAgreement": self.predicted_pairwise_agreement,
            "predictedPairwisePairs": self.predicted_pairwise_pairs,
            "predictedSelectedActionIsTargetQMax": self.predicted_selected_action_is_target_q_max,
            "predictedSelectedActionTargetQRank": self.predicted_selected_action_target_q_rank,
        })
    }
}

#[derive(Clone, Debug, Default)]
struct EvaluationSummary {
    games: u64,
    positions: u64,
    q_positions: u64,
    missing_q_positions: u64,
    invalid_games: u64,
    all: Metrics,
    placement: Metrics,
    relocation: Metrics,
}

impl EvaluationSummary {
    fn raw_json(&self) -> Value {
        json!({
            "games": self.games,
            "positions": self.positions,
            "qPositions": self.q_positions,
            "missingQPositions": self.missing_q_positions,
            "invalidGames": self.invalid_games,
            "metrics": {
                "all": self.all.raw_json(),
                "placement": self.placement.raw_json(),
                "relocation": self.relocation.raw_json(),
            },
        })
    }
}

fn number(value: &Value, key: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing or invalid numeric field {key}"))
}

fn record_config(record: &Value) -> Result<BoardConfig, String> {
    let config = record.get("config").unwrap_or(&Value::Null);
    let board_size = config
        .get("boardSize")
        .and_then(Value::as_u64)
        .or_else(|| record.get("boardSize").and_then(Value::as_u64))
        .unwrap_or(7) as u8;
    let reserve = config
        .get("reservePerPlayer")
        .and_then(Value::as_u64)
        .or_else(|| record.get("reservePerPlayer").and_then(Value::as_u64))
        .unwrap_or(14) as u8;
    let max_plies = config
        .get("maxPlies")
        .and_then(Value::as_u64)
        .or_else(|| record.get("maxPlies").and_then(Value::as_u64))
        .unwrap_or(196) as u16;
    BoardConfig::new(board_size, reserve)?.with_max_plies(max_plies)
}

fn parse_action(value: &Value) -> Result<Action, String> {
    let kind = value
        .get("kind")
        .ok_or_else(|| "action has no kind".to_owned())?;
    match kind.as_str().or_else(|| kind.as_u64().map(|_| "numeric")) {
        Some("place") => Ok(Action::Place {
            to: number(value, "to")? as u8,
        }),
        Some("relocate") => Ok(Action::Relocate {
            from: number(value, "from")? as u8,
            to: number(value, "to")? as u8,
        }),
        _ => Err(format!("unsupported action kind {kind}")),
    }
}

fn floats(value: &Value, key: &str) -> Result<Vec<f32>, String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing or invalid {key}"))?
        .iter()
        .map(|entry| {
            entry
                .as_f64()
                .map(|number| number as f32)
                .ok_or_else(|| format!("{key} contains a non-number"))
        })
        .collect()
}

fn integers(value: &Value, key: &str) -> Result<Vec<u32>, String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing or invalid {key}"))?
        .iter()
        .map(|entry| {
            entry
                .as_u64()
                .map(|number| number as u32)
                .ok_or_else(|| format!("{key} contains a non-integer"))
        })
        .collect()
}

fn rank(values: &[f32], index: usize) -> usize {
    let target = values[index];
    1 + values
        .iter()
        .filter(|value| **value > target + 1.0e-6)
        .count()
}

fn pairwise(left: &[f32], right: &[f32], indices: &[usize]) -> (u64, u64) {
    let mut agreements = 0;
    let mut pairs = 0;
    for (offset, &left_index) in indices.iter().enumerate() {
        for &right_index in &indices[offset + 1..] {
            let left_delta = left[left_index] - left[right_index];
            let right_delta = right[left_index] - right[right_index];
            if left_delta.abs() < 1.0e-6 || right_delta.abs() < 1.0e-6 {
                continue;
            }
            pairs += 1;
            if left_delta * right_delta > 0.0 {
                agreements += 1;
            }
        }
    }
    (agreements, pairs)
}

fn score_position(
    metrics: &mut Metrics,
    state: GameState,
    action: Action,
    target_q: &[f32],
    visits: &[u32],
    predicted_q: &[f32],
) -> Result<(), String> {
    let actions = state.legal_actions();
    if target_q.len() != actions.len() || visits.len() != actions.len() {
        return Err(format!(
            "Q target length mismatch at ply {}: legal={}, values={}, visits={}",
            state.ply,
            actions.len(),
            target_q.len(),
            visits.len()
        ));
    }
    if predicted_q.len() < actions.len() {
        return Err(format!(
            "model returned {} Q values for {} legal actions",
            predicted_q.len(),
            actions.len()
        ));
    }
    let selected_index = actions
        .iter()
        .position(|candidate| *candidate == action)
        .ok_or_else(|| format!("recorded action is illegal at ply {}", state.ply))?;
    let visited: Vec<usize> = visits
        .iter()
        .enumerate()
        .filter_map(|(index, count)| (*count > 0).then_some(index))
        .collect();
    let target_rank = rank(target_q, selected_index);
    metrics.positions += 1;
    metrics.visited_actions += visited.len() as u64;
    metrics.visited_pairs += (visited.len() * visited.len().saturating_sub(1) / 2) as u64;
    metrics.selected_action_is_q_max += u64::from(target_rank == 1);
    metrics.selected_action_q_rank += target_rank as f64;
    metrics.selected_action_q_percentile += if target_q.len() == 1 {
        1.0
    } else {
        1.0 - (target_rank - 1) as f64 / (target_q.len() - 1) as f64
    };
    metrics.q_spread += f64::from(
        target_q
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max)
            - target_q.iter().copied().fold(f32::INFINITY, f32::min),
    );
    let (agreements, pairs) = pairwise(predicted_q, target_q, &visited);
    metrics.predicted_pairwise_agreement += agreements;
    metrics.predicted_pairwise_pairs += pairs;
    let predicted_index = visited
        .iter()
        .copied()
        .max_by(|left, right| {
            predicted_q[*left]
                .partial_cmp(&predicted_q[*right])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| right.cmp(left))
        })
        .unwrap_or_else(|| {
            (0..actions.len())
                .max_by(|left, right| {
                    predicted_q[*left]
                        .partial_cmp(&predicted_q[*right])
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| right.cmp(left))
                })
                .unwrap_or(0)
        });
    let predicted_rank = rank(target_q, predicted_index);
    metrics.predicted_selected_action_is_target_q_max += u64::from(predicted_rank == 1);
    metrics.predicted_selected_action_target_q_rank += predicted_rank as f64;
    for &index in &visited {
        let weight = f64::from(visits[index]).sqrt();
        let error = f64::from(predicted_q[index] - target_q[index]);
        metrics.q_mse += weight * error * error;
        metrics.q_mae += weight * error.abs();
        metrics.q_weight += weight;
    }
    Ok(())
}

fn evaluate_record(record: &Value, model: &OnnxQAdvModel, summary: &mut EvaluationSummary) -> Result<(), String> {
    let config = record_config(record)?;
    let seed = record.get("seed").and_then(Value::as_u64).unwrap_or(0);
    let moves = record
        .get("moves")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("seed {seed}: missing moves"))?;
    let mut state = GameState::with_config(config);
    for movement in moves {
        summary.positions += 1;
        let action = parse_action(
            movement
                .get("action")
                .ok_or_else(|| format!("seed {seed}: move has no action"))?,
        )?;
        let has_q = movement.get("actionValues").is_some() && movement.get("actionVisits").is_some();
        if has_q {
            let values = floats(movement, "actionValues")?;
            let visits = integers(movement, "actionVisits")?;
            let evaluated = model.evaluate_qadv(state).map_err(|error| error.to_string())?;
            let legal_count = state.legal_actions().len();
            if evaluated.q_values.len() < legal_count {
                return Err(format!(
                    "model returned {} Q values for {} legal actions",
                    evaluated.q_values.len(),
                    legal_count
                ));
            }
            let predicted_q = &evaluated.q_values[..legal_count];
            score_position(
                if matches!(action, Action::Place { .. }) {
                    &mut summary.placement
                } else {
                    &mut summary.relocation
                },
                state,
                action,
                &values,
                &visits,
                predicted_q,
            )?;
            // The all-phase bucket receives the same position exactly once.
            score_position(&mut summary.all, state, action, &values, &visits, predicted_q)?;
            summary.q_positions += 1;
        } else {
            summary.missing_q_positions += 1;
        }
        state = state.apply(action).map_err(|error| format!("seed {seed}: {error}"))?.state;
    }
    Ok(())
}

async fn handler(event: LambdaEvent<EvaluationJob>) -> Result<Value, Error> {
    let job = event.payload;
    let model = model()?;
    let mut summary = EvaluationSummary::default();
    for record in &job.games {
        summary.games += 1;
        if evaluate_record(record, model.as_ref(), &mut summary).is_err() {
            summary.invalid_games += 1;
        }
    }
    Ok(json!({
        "schema": "pathagon-qadv-evaluation-v1",
        "chunkId": job.chunk_id,
        "summary": summary.raw_json(),
    }))
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    lambda_runtime::run(service_fn(handler)).await
}
