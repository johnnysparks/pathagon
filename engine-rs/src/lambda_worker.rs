//! One-game QAdv worker for AWS Lambda's `provided.al2023` runtime.
//!
//! The model is loaded from `/var/task/qadv-arbiter.onnx` once per warm
//! execution environment. Each invocation produces exactly one schema-v2 game
//! record, making the function safe to drive with a bounded synchronous fanout
//! coordinator.

use std::env;
use std::fs;
use std::sync::{Arc, OnceLock};

use lambda_runtime::{service_fn, Error, LambdaEvent};
use serde::Deserialize;
use serde_json::{json, Value};

use pathagon_engine::inference::OnnxQAdvModel;
use pathagon_engine::puct::PuctConfig;
use pathagon_engine::search::SearchConfig;
use pathagon_engine::selfplay::{play_game, Agent, GnnPlayConfig, MatchOptions, QAdvPlayConfig};

#[derive(Clone, Debug, Deserialize)]
struct GameJob {
    seed: u32,
    #[serde(default = "default_opponent")]
    opponent: String,
    #[serde(default)]
    qadv_light: Option<bool>,
    #[serde(default = "default_max_plies")]
    max_plies: u16,
    #[serde(default = "default_simulations")]
    simulations: u32,
    #[serde(default = "default_temperature_moves")]
    temperature_moves: u16,
    #[serde(default = "default_policy_temperature")]
    policy_temperature: f32,
    #[serde(default = "default_opening_moves")]
    opening_moves: u16,
    #[serde(default = "default_opening_temperature")]
    opening_temperature: f32,
    #[serde(default = "default_opening_randomness")]
    opening_randomness: f32,
    #[serde(default = "default_pathfinder_guidance")]
    pathfinder_guidance: f32,
    #[serde(default = "default_placement_guidance")]
    placement_guidance: f32,
    #[serde(default = "default_pathfinder_temperature")]
    pathfinder_temperature: f32,
    #[serde(default = "default_pathfinder_depth")]
    pathfinder_depth: u8,
    #[serde(default = "default_pathfinder_beam")]
    pathfinder_beam: usize,
    #[serde(default = "default_pathfinder_nodes")]
    pathfinder_nodes: u64,
    #[serde(default = "default_qadv_weight")]
    qadv_weight: f32,
    #[serde(default = "default_tactical_simulations")]
    tactical_simulations: u32,
    #[serde(default = "default_tactical_capture_threshold")]
    tactical_capture_threshold: u8,
}

static MODEL: OnceLock<Result<Arc<OnnxQAdvModel>, String>> = OnceLock::new();

fn model() -> Result<Arc<OnnxQAdvModel>, Error> {
    let result = MODEL.get_or_init(|| {
        let path = env::var("QADV_MODEL_PATH")
            .unwrap_or_else(|_| "/var/task/qadv-arbiter.onnx".to_owned());
        let bytes =
            fs::read(&path).map_err(|error| format!("cannot read QAdv model {path}: {error}"))?;
        OnnxQAdvModel::from_bytes(&bytes).map(Arc::new)
    });
    match result {
        Ok(model) => Ok(Arc::clone(model)),
        Err(error) => Err(std::io::Error::other(error.clone()).into()),
    }
}

async fn handler(event: LambdaEvent<GameJob>) -> Result<Value, Error> {
    let job = event.payload;
    let model = model()?;
    let guided = GnnPlayConfig {
        puct: PuctConfig {
            simulations: job.simulations,
            cpuct: 1.5,
        },
        temperature_moves: job.temperature_moves,
        policy_temperature: job.policy_temperature,
        opening_moves: job.opening_moves,
        opening_temperature: job.opening_temperature,
        opening_randomness: job.opening_randomness,
        pathfinder_guidance: job.pathfinder_guidance,
        placement_guidance: job.placement_guidance,
        pathfinder_temperature: job.pathfinder_temperature,
        pathfinder_depth: job.pathfinder_depth,
        pathfinder_beam: job.pathfinder_beam,
        pathfinder_nodes: job.pathfinder_nodes,
    };
    let qadv_config = QAdvPlayConfig {
        guided,
        qadv_weight: job.qadv_weight,
        tactical_simulations: job.tactical_simulations,
        tactical_capture_threshold: job.tactical_capture_threshold,
    };
    let qadv = Agent::qadv("qadv-arbiter-7x7-v0.1.0", qadv_config, Arc::clone(&model));
    let opponent = match job.opponent.as_str() {
        "qadv" => Agent::qadv("qadv-arbiter-7x7-v0.1.0", qadv_config, Arc::clone(&model)),
        "pathfinder" => Agent::search(
            "pathfinder-v0.3.0",
            SearchConfig {
                depth: 2,
                max_nodes: 1_000,
                beam_width: 8,
                ..SearchConfig::default()
            },
        ),
        "surveyor" => Agent::search(
            "surveyor-v0.2.0",
            SearchConfig {
                depth: 1,
                max_nodes: 500,
                beam_width: 12,
                ..SearchConfig::default()
            },
        ),
        "lunatic" => Agent::lunatic("lunatic-v0.1.0"),
        "coin-flip" | "random" => Agent::random("coin-flip-v0.0.1"),
        other => return Err(format!("unsupported opponent: {other}").into()),
    };
    let qadv_light = job.qadv_light.unwrap_or(job.seed % 2 == 0);
    let (light, dark) = if qadv_light {
        (&qadv, &opponent)
    } else {
        (&opponent, &qadv)
    };
    let record = play_game(
        light,
        dark,
        MatchOptions {
            seed: job.seed,
            max_plies: job.max_plies,
            opening_random_plies: 0,
            board_size: 7,
            reserve_per_player: 14,
        },
    );
    let record: Value = serde_json::from_str(&record.to_json())?;
    Ok(json!({
        "seed": job.seed,
        "opponent": job.opponent,
        "qadvLight": qadv_light,
        "record": record,
    }))
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    lambda_runtime::run(service_fn(handler)).await
}

const fn default_max_plies() -> u16 {
    196
}
fn default_opponent() -> String {
    "qadv".to_owned()
}
const fn default_simulations() -> u32 {
    128
}
const fn default_temperature_moves() -> u16 {
    48
}
const fn default_policy_temperature() -> f32 {
    1.15
}
const fn default_opening_moves() -> u16 {
    16
}
const fn default_opening_temperature() -> f32 {
    1.8
}
const fn default_opening_randomness() -> f32 {
    0.30
}
const fn default_pathfinder_guidance() -> f32 {
    0.45
}
const fn default_placement_guidance() -> f32 {
    0.30
}
const fn default_pathfinder_temperature() -> f32 {
    1.15
}
const fn default_pathfinder_depth() -> u8 {
    2
}
const fn default_pathfinder_beam() -> usize {
    8
}
const fn default_pathfinder_nodes() -> u64 {
    512
}
const fn default_qadv_weight() -> f32 {
    1.0
}
const fn default_tactical_simulations() -> u32 {
    512
}
const fn default_tactical_capture_threshold() -> u8 {
    2
}
