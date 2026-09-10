//! Measure search-generated teacher decisions against independent held-out labels.
//!
//! The terminal suite is the held-out Ring-1 partition.  Its oracle enumerates
//! every legal action and accepts exactly the actions that immediately create a
//! winning path.  The proof suite is sampled from a disjoint game-key bucket of
//! the canonical replay corpus and uses the rules-only bounded AND/OR solver.
//! Neither oracle reads the transition-policy model or the heuristic evaluator.

use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use pathagon_engine::corpus::decode_action;
use pathagon_engine::endgame::{analyze, TacticalProofConfig};
use pathagon_engine::golden::{GoldenLookup, GoldenOutcome};
use pathagon_engine::search::{
    search_best_action_with_tactical_filter, EvaluationWeights, SearchConfig, SearchResult,
};
use pathagon_engine::transition_policy::TransitionPolicyModel;
use pathagon_engine::{Action, BoardConfig, GameState};
use serde::Serialize;
use sha2::{Digest, Sha256};

const BOARD_SIZE: u8 = 7;
const RESERVE: u8 = 14;
const MAX_PLIES: u16 = 196;

#[derive(Clone, Debug)]
struct Args {
    heldout: PathBuf,
    model: PathBuf,
    golden_table: PathBuf,
    golden_sidecar: PathBuf,
    corpus_dir: PathBuf,
    out: PathBuf,
    max_terminal: usize,
    max_proof: usize,
    proof_horizon: u8,
    proof_nodes: u64,
    search_nodes: u64,
    teacher_nodes: u64,
    beam: usize,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct SearchObservation {
    action: Option<ActionRecord>,
    legal: bool,
    hit: bool,
    exhausted: bool,
    completed_depth: u8,
    nodes: u64,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct ActionRecord {
    kind: &'static str,
    from: Option<u8>,
    to: u8,
}

#[derive(Clone, Debug, Serialize)]
struct PositionRecord {
    id: String,
    source: String,
    ply: u16,
    turn: &'static str,
    pieces: usize,
    legal_actions: usize,
    oracle_actions: Vec<ActionRecord>,
    oracle_outcome: &'static str,
    deployment: SearchObservation,
    teacher: SearchObservation,
    raw_deployment: SearchObservation,
}

#[derive(Clone, Debug, Serialize)]
struct SuiteSummary {
    positions: usize,
    oracle_positions: usize,
    oracle_hits: usize,
    oracle_hit_rate: f64,
    outcome_counts: BTreeMap<String, usize>,
    deployment_hits: usize,
    teacher_hits: usize,
    raw_deployment_hits: usize,
    deployment_hit_rate: f64,
    teacher_hit_rate: f64,
    raw_deployment_hit_rate: f64,
    teacher_beats_deployment: usize,
    deployment_beats_teacher: usize,
    disagreements: usize,
    deployment_exhausted: usize,
    teacher_exhausted: usize,
    raw_deployment_exhausted: usize,
    deployment_nodes_mean: f64,
    teacher_nodes_mean: f64,
    raw_deployment_nodes_mean: f64,
    deployment_depth_mean: f64,
    teacher_depth_mean: f64,
    raw_deployment_depth_mean: f64,
}

#[derive(Clone, Debug, Serialize)]
struct Report {
    schema_version: u8,
    experiment: &'static str,
    rules_version: &'static str,
    model_sha256: String,
    protocol: BTreeMap<String, serde_json::Value>,
    terminal: SuiteSummary,
    proof: SuiteSummary,
    terminal_positions: Vec<PositionRecord>,
    proof_positions: Vec<PositionRecord>,
}

#[derive(Clone, Copy)]
struct Evaluator<'a> {
    model: &'a TransitionPolicyModel,
    deployment: SearchConfig,
    teacher: SearchConfig,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let model_bytes = fs::read(&args.model)?;
    let model_sha256 = hex_digest(&Sha256::digest(&model_bytes));
    let model = TransitionPolicyModel::from_bytes(&model_bytes)?;
    let golden = GoldenLookup::open(
        &args.golden_table,
        Some(&args.golden_sidecar),
        BOARD_SIZE,
        RESERVE,
    )?;
    let weights = EvaluationWeights {
        path: 241,
        material: 112,
        capture: 887,
        structure: 40,
        threat: 154,
        edge: 74,
    };
    let evaluator = Evaluator {
        model: &model,
        deployment: SearchConfig {
            depth: 5,
            max_nodes: args.search_nodes,
            beam_width: args.beam,
            weights,
            tactical_proof_horizon: None,
        },
        teacher: SearchConfig {
            depth: 6,
            max_nodes: args.teacher_nodes,
            beam_width: args.beam,
            weights,
            tactical_proof_horizon: None,
        },
    };

    let terminal_positions = evaluate_terminal_suite(&args, &golden, evaluator)?;
    let proof_positions = evaluate_proof_suite(&args, evaluator)?;
    let terminal = summarize(&terminal_positions);
    let proof = summarize(&proof_positions);

    if let Some(parent) = args
        .out
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let report = Report {
        schema_version: 1,
        experiment: "teacher-quality-at-deployment-envelope",
        rules_version: "pathagon-rules-v1",
        model_sha256,
        protocol: BTreeMap::from([
            ("boardSize".to_owned(), serde_json::json!(BOARD_SIZE)),
            ("reservePerPlayer".to_owned(), serde_json::json!(RESERVE)),
            ("maxPlies".to_owned(), serde_json::json!(MAX_PLIES)),
            (
                "deployment".to_owned(),
                serde_json::json!({
                    "depth": evaluator.deployment.depth,
                    "beam": evaluator.deployment.beam_width,
                    "nodes": evaluator.deployment.max_nodes,
                }),
            ),
            (
                "teacher".to_owned(),
                serde_json::json!({
                    "depth": evaluator.teacher.depth,
                    "beam": evaluator.teacher.beam_width,
                    "nodes": evaluator.teacher.max_nodes,
                }),
            ),
            (
                "proof".to_owned(),
                serde_json::json!({
                    "horizon": args.proof_horizon,
                    "maxNodes": args.proof_nodes,
                }),
            ),
            ("terminalSource".to_owned(), serde_json::json!(args.heldout)),
            ("proofSource".to_owned(), serde_json::json!(args.corpus_dir)),
            (
                "proofHoldout".to_owned(),
                serde_json::json!("sha256(game-key) mod 5 = 0"),
            ),
        ]),
        terminal,
        proof,
        terminal_positions,
        proof_positions,
    };
    fs::write(&args.out, serde_json::to_vec_pretty(&report)?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn evaluate_terminal_suite(
    args: &Args,
    golden: &GoldenLookup,
    evaluator: Evaluator<'_>,
) -> Result<Vec<PositionRecord>, Box<dyn std::error::Error>> {
    let lines = fs::read_to_string(&args.heldout)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let selected = evenly_select(&lines, args.max_terminal);
    let mut records = Vec::with_capacity(selected.len());
    for (index, key_hex) in selected.iter().enumerate() {
        let key = decode_hex(key_hex)?;
        let state = path_state(pathagon_engine::golden::decode_canonical_position_key(
            &key, BOARD_SIZE, RESERVE,
        )?)?;
        let oracle = immediate_wins(state);
        if oracle.is_empty() {
            return Err(format!("held-out key {key_hex} has no immediate winning action").into());
        }
        if golden.lookup(state) != Some(GoldenOutcome::Win) {
            return Err(
                format!("held-out key {key_hex} is absent from the promoted Ring-1 table").into(),
            );
        }
        let proven = golden
            .action_values(state)
            .unwrap_or_default()
            .into_iter()
            .filter(|value| value.outcome == Some(GoldenOutcome::Win))
            .map(|value| value.action)
            .collect::<HashSet<_>>();
        if proven.is_empty() || !proven.iter().all(|action| oracle.contains(action)) {
            return Err(format!("held-out key {key_hex} failed sidecar/oracle cross-check").into());
        }
        records.push(observe_position(
            format!("ring1-{index:04}"),
            "heldout-ring1",
            state,
            &oracle,
            "immediate-win",
            evaluator,
        ));
    }
    Ok(records)
}

fn evaluate_proof_suite(
    args: &Args,
    evaluator: Evaluator<'_>,
) -> Result<Vec<PositionRecord>, Box<dyn std::error::Error>> {
    let mut records = Vec::with_capacity(args.max_proof);
    let mut seen = HashSet::new();
    let mut holdout_games = 0_usize;
    let mut candidate_states = 0_usize;
    let mut accepted_states = 0_usize;
    let mut files = fs::read_dir(&args.corpus_dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("tsv"))
        .collect::<Vec<_>>();
    files.sort();
    'files: for path in files {
        let reader = BufReader::new(File::open(&path)?);
        for line in reader.lines() {
            let line = line?;
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let fields = line.split('\t').collect::<Vec<_>>();
            if fields.len() < 7 || !holdout_game_key(fields[0]) {
                continue;
            }
            holdout_games += 1;
            let actions = fields[6];
            let mut state = GameState::with_config(
                BoardConfig::new(BOARD_SIZE, RESERVE)?.with_max_plies(MAX_PLIES)?,
            );
            for (ply, token) in actions.as_bytes().chunks_exact(2).enumerate() {
                if records.len() >= args.max_proof {
                    break 'files;
                }
                let action_token = std::str::from_utf8(token)?;
                let action = decode_action(action_token)?;
                if state.winner.is_none()
                    && state.ply >= 8
                    && state.ply <= 90
                    && state.legal_action_count() <= 512
                    && immediate_wins(state).is_empty()
                    && (ply + usize::from(state.light.count_ones() as u16)) % 3 == 0
                {
                    let identity = (
                        state.light,
                        state.dark,
                        state.reserve,
                        state.turn,
                        state.forbidden,
                        state.last_relocated_to,
                    );
                    if seen.insert(identity) {
                        candidate_states += 1;
                        let analysis = analyze(
                            state,
                            TacticalProofConfig {
                                horizon: args.proof_horizon,
                                max_nodes: args.proof_nodes,
                            },
                        )?;
                        let discriminating = !analysis.optimal_actions.is_empty()
                            && analysis.optimal_actions.len() < state.legal_action_count();
                        if !analysis.stats.exhausted && analysis.outcome >= 0 && discriminating {
                            accepted_states += 1;
                            let outcome = match analysis.outcome {
                                1 => "win",
                                0 => "draw",
                                _ => "loss",
                            };
                            records.push(observe_position(
                                format!("proof-{:04}", records.len()),
                                "corpus-proof-holdout",
                                state,
                                &analysis.optimal_actions,
                                outcome,
                                evaluator,
                            ));
                        }
                    }
                }
                if !state.legal_actions().contains(&action) {
                    break;
                }
                state = state.apply_legal(action).state;
            }
        }
    }
    eprintln!(
        "proof collection: holdout_games={holdout_games} candidates={candidate_states} accepted={accepted_states} records={accepted_states}"
    );
    Ok(records)
}

fn observe_position(
    id: String,
    source: &'static str,
    state: GameState,
    oracle: &[Action],
    outcome: &'static str,
    evaluator: Evaluator<'_>,
) -> PositionRecord {
    let legal = state.legal_actions();
    let deployment_result = evaluator.model.search(state, evaluator.deployment, None);
    let teacher_result = evaluator.model.search(state, evaluator.teacher, None);
    let raw_result = search_best_action_with_tactical_filter(state, evaluator.deployment);
    PositionRecord {
        id,
        source: source.to_owned(),
        ply: state.ply,
        turn: state.turn.as_str(),
        pieces: (state.light | state.dark).count_ones() as usize,
        legal_actions: legal.len(),
        oracle_actions: oracle.iter().copied().map(action_record).collect(),
        oracle_outcome: outcome,
        deployment: observation(deployment_result, &legal, oracle),
        teacher: observation(teacher_result, &legal, oracle),
        raw_deployment: observation(raw_result, &legal, oracle),
    }
}

fn observation(result: SearchResult, legal: &[Action], oracle: &[Action]) -> SearchObservation {
    let action = result.action;
    SearchObservation {
        action: action.map(action_record),
        legal: action.is_some_and(|value| legal.contains(&value)),
        hit: action.is_some_and(|value| oracle.contains(&value)),
        exhausted: result.exhausted,
        completed_depth: result.completed_depth,
        nodes: result.nodes,
    }
}

fn summarize(records: &[PositionRecord]) -> SuiteSummary {
    let mut outcome_counts = BTreeMap::new();
    let mut deployment_hits = 0;
    let mut teacher_hits = 0;
    let mut raw_hits = 0;
    let mut teacher_beats = 0;
    let mut deployment_beats = 0;
    let mut disagreements = 0;
    let mut deployment_exhausted = 0;
    let mut teacher_exhausted = 0;
    let mut raw_exhausted = 0;
    let mut deployment_nodes = 0_u64;
    let mut teacher_nodes = 0_u64;
    let mut raw_nodes = 0_u64;
    let mut deployment_depth = 0_u64;
    let mut teacher_depth = 0_u64;
    let mut raw_depth = 0_u64;
    for record in records {
        *outcome_counts
            .entry(record.oracle_outcome.to_owned())
            .or_insert(0) += 1;
        deployment_hits += usize::from(record.deployment.hit);
        teacher_hits += usize::from(record.teacher.hit);
        raw_hits += usize::from(record.raw_deployment.hit);
        teacher_beats += usize::from(record.teacher.hit && !record.deployment.hit);
        deployment_beats += usize::from(record.deployment.hit && !record.teacher.hit);
        disagreements += usize::from(
            record.deployment.action.map(|a| (a.kind, a.from, a.to))
                != record.teacher.action.map(|a| (a.kind, a.from, a.to)),
        );
        deployment_exhausted += usize::from(record.deployment.exhausted);
        teacher_exhausted += usize::from(record.teacher.exhausted);
        raw_exhausted += usize::from(record.raw_deployment.exhausted);
        deployment_nodes += record.deployment.nodes;
        teacher_nodes += record.teacher.nodes;
        raw_nodes += record.raw_deployment.nodes;
        deployment_depth += u64::from(record.deployment.completed_depth);
        teacher_depth += u64::from(record.teacher.completed_depth);
        raw_depth += u64::from(record.raw_deployment.completed_depth);
    }
    let positions = records.len();
    let rate = |value| {
        if positions == 0 {
            0.0
        } else {
            value as f64 / positions as f64
        }
    };
    SuiteSummary {
        positions,
        oracle_positions: positions,
        oracle_hits: positions,
        oracle_hit_rate: rate(positions),
        outcome_counts,
        deployment_hits,
        teacher_hits,
        raw_deployment_hits: raw_hits,
        deployment_hit_rate: rate(deployment_hits),
        teacher_hit_rate: rate(teacher_hits),
        raw_deployment_hit_rate: rate(raw_hits),
        teacher_beats_deployment: teacher_beats,
        deployment_beats_teacher: deployment_beats,
        disagreements,
        deployment_exhausted,
        teacher_exhausted,
        raw_deployment_exhausted: raw_exhausted,
        deployment_nodes_mean: if positions == 0 {
            0.0
        } else {
            deployment_nodes as f64 / positions as f64
        },
        teacher_nodes_mean: if positions == 0 {
            0.0
        } else {
            teacher_nodes as f64 / positions as f64
        },
        raw_deployment_nodes_mean: if positions == 0 {
            0.0
        } else {
            raw_nodes as f64 / positions as f64
        },
        deployment_depth_mean: if positions == 0 {
            0.0
        } else {
            deployment_depth as f64 / positions as f64
        },
        teacher_depth_mean: if positions == 0 {
            0.0
        } else {
            teacher_depth as f64 / positions as f64
        },
        raw_deployment_depth_mean: if positions == 0 {
            0.0
        } else {
            raw_depth as f64 / positions as f64
        },
    }
}

fn immediate_wins(state: GameState) -> Vec<Action> {
    state
        .legal_actions()
        .into_iter()
        .filter(|action| state.apply_legal(*action).state.winner == Some(state.turn))
        .collect()
}

fn path_state(mut state: GameState) -> Result<GameState, Box<dyn std::error::Error>> {
    state.config = state.config.with_max_plies(MAX_PLIES)?;
    Ok(state)
}

fn action_record(action: Action) -> ActionRecord {
    match action {
        Action::Place { to } => ActionRecord {
            kind: "place",
            from: None,
            to,
        },
        Action::Relocate { from, to } => ActionRecord {
            kind: "relocate",
            from: Some(from),
            to,
        },
    }
}

fn holdout_game_key(key: &str) -> bool {
    let digest = Sha256::digest(key.as_bytes());
    u16::from_le_bytes([digest[0], digest[1]]) % 5 == 0
}

fn evenly_select<T: Clone>(values: &[T], limit: usize) -> Vec<T> {
    if limit == 0 {
        return Vec::new();
    }
    if values.len() <= limit {
        return values.to_vec();
    }
    (0..limit)
        .map(|index| values[index * values.len() / limit].clone())
        .collect()
}

fn decode_hex(value: &str) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    if value.len() % 2 != 0 {
        return Err("hex key has odd length".into());
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        let high = (pair[0] as char).to_digit(16).ok_or("invalid hex key")?;
        let low = (pair[1] as char).to_digit(16).ok_or("invalid hex key")?;
        bytes.push(((high << 4) | low) as u8);
    }
    Ok(bytes)
}

fn hex_digest(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn parse_args() -> Args {
    let mut values = BTreeMap::new();
    let mut iter = env::args().skip(1);
    while let Some(argument) = iter.next() {
        let key = argument
            .strip_prefix("--")
            .unwrap_or_else(|| panic!("unexpected argument {argument}"));
        let value = iter
            .next()
            .unwrap_or_else(|| panic!("missing value for --{key}"));
        values.insert(key.to_owned(), value);
    }
    let required =
        |key: &str| PathBuf::from(values.get(key).unwrap_or_else(|| panic!("missing --{key}")));
    Args {
        heldout: required("heldout"),
        model: required("model"),
        golden_table: required("golden-table"),
        golden_sidecar: required("golden-sidecar"),
        corpus_dir: required("corpus-dir"),
        out: required("out"),
        max_terminal: parse_number(&values, "max-terminal", 128_usize),
        max_proof: parse_number(&values, "max-proof", 64_usize),
        proof_horizon: parse_number(&values, "proof-horizon", 3_u8),
        proof_nodes: parse_number(&values, "proof-nodes", 100_000_u64),
        search_nodes: parse_number(&values, "search-nodes", 256_000_u64),
        teacher_nodes: parse_number(&values, "teacher-nodes", 512_000_u64),
        beam: parse_number(&values, "beam", 256_usize),
    }
}

fn parse_number<T: std::str::FromStr>(
    values: &BTreeMap<String, String>,
    key: &str,
    default: T,
) -> T {
    values
        .get(key)
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}
