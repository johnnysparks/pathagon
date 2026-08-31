//! One-seeded-game Lambda worker for the turn-balanced contextual arena.
//!
//! The request carries only a seed and optional search limits. The worker
//! returns one ordinary Pathagon game record, so a local coordinator can fan
//! out independent games and run the same legality/color audits as the local
//! runner. All state is ephemeral and no credentials are read by the worker.

use lambda_runtime::{service_fn, Error, LambdaEvent};
use pathagon_engine::search::{EvaluationWeights, SearchConfig};
use pathagon_engine::selfplay::{play_game, Agent, MatchOptions};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Clone, Debug, Deserialize)]
struct GameJob {
    seed: u32,
    #[serde(default)]
    candidate_light: Option<bool>,
    #[serde(default = "default_max_plies")]
    max_plies: u16,
    #[serde(default = "default_depth")]
    depth: u8,
    #[serde(default = "default_nodes")]
    nodes: u64,
    #[serde(default = "default_beam")]
    beam: usize,
    #[serde(default = "default_deadline_ms")]
    deadline_ms: u32,
}

const BASELINE: EvaluationWeights = EvaluationWeights {
    path: 241,
    material: 112,
    capture: 887,
    structure: 40,
    threat: 154,
    edge: 74,
};

// The small d7-only screen candidate is kept here as a reproducible cloud
// smoke-test payload. The larger 1,920-root fit can replace these constants
// after its held-out gate is passed.
const LIGHT: [EvaluationWeights; 4] = [
    EvaluationWeights {
        path: 241,
        material: 112,
        capture: 887,
        structure: 40,
        threat: 154,
        edge: 74,
    },
    EvaluationWeights {
        path: 256,
        material: 105,
        capture: 880,
        structure: 27,
        threat: 157,
        edge: 74,
    },
    EvaluationWeights {
        path: 241,
        material: 112,
        capture: 887,
        structure: 40,
        threat: 154,
        edge: 74,
    },
    EvaluationWeights {
        path: 239,
        material: 95,
        capture: 869,
        structure: 37,
        threat: 157,
        edge: 72,
    },
];
const DARK: [EvaluationWeights; 4] = [
    EvaluationWeights {
        path: 241,
        material: 112,
        capture: 887,
        structure: 40,
        threat: 154,
        edge: 74,
    },
    EvaluationWeights {
        path: 246,
        material: 72,
        capture: 847,
        structure: 42,
        threat: 179,
        edge: 79,
    },
    EvaluationWeights {
        path: 241,
        material: 112,
        capture: 887,
        structure: 40,
        threat: 154,
        edge: 74,
    },
    EvaluationWeights {
        path: 151,
        material: 97,
        capture: 872,
        structure: 45,
        threat: 137,
        edge: 79,
    },
];

fn configs(weights: [EvaluationWeights; 4], job: &GameJob) -> [SearchConfig; 4] {
    weights.map(|weights| SearchConfig {
        depth: job.depth,
        max_nodes: job.nodes,
        beam_width: job.beam,
        weights,
        tactical_proof_horizon: None,
    })
}

async fn handler(event: LambdaEvent<GameJob>) -> Result<Value, Error> {
    let job = event.payload;
    let candidate = Agent::contextual_by_player_with_deadline(
        "pathfinder-contextual-v0.6-research",
        configs(LIGHT, &job),
        configs(DARK, &job),
        job.deadline_ms,
    );
    let incumbent = Agent::search_tactical_filter_with_deadline(
        "pathfinder-v0.5.0-trained-evaluator",
        SearchConfig {
            depth: job.depth,
            max_nodes: job.nodes,
            beam_width: job.beam,
            weights: BASELINE,
            tactical_proof_horizon: None,
        },
        job.deadline_ms,
    );
    let candidate_light = job.candidate_light.unwrap_or(job.seed % 2 == 0);
    let (light, dark) = if candidate_light {
        (&candidate, &incumbent)
    } else {
        (&incumbent, &candidate)
    };
    let record = play_game(
        light,
        dark,
        MatchOptions {
            seed: job.seed,
            max_plies: job.max_plies,
            opening_random_plies: 2,
            board_size: 7,
            reserve_per_player: 14,
        },
    );
    let record: Value = serde_json::from_str(&record.to_json())?;
    Ok(json!({
        "seed": job.seed,
        "candidateLight": candidate_light,
        "record": record,
    }))
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    lambda_runtime::run(service_fn(handler)).await
}

const fn default_max_plies() -> u16 {
    100
}
const fn default_depth() -> u8 {
    7
}
const fn default_nodes() -> u64 {
    1_000_000
}
const fn default_beam() -> usize {
    32
}
const fn default_deadline_ms() -> u32 {
    2_800
}
