//! Export quiet positions with independent action regret and continuation targets.
//!
//! Rust owns replay, legality, tactical filtering, search, and rollout targets.
//! Python only consumes the resulting JSONL for model fitting and analysis.

use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use pathagon_engine::corpus::{encode_action, encode_state, parse_unified_game};
use pathagon_engine::search::{
    analyze_action, ordered_root_actions, search_best_action_with_tactical_filter,
    EvaluationWeights, SearchConfig,
};
use pathagon_engine::{Action, GameState, Player};
use serde_json::json;

#[derive(Clone, Debug)]
struct Observation {
    seed64: String,
    light: String,
    light_model: String,
    dark: String,
    dark_model: String,
    winner: Option<Player>,
}

impl Observation {
    fn unknown() -> Self {
        Self {
            seed64: String::new(),
            light: "unknown".to_owned(),
            light_model: String::new(),
            dark: "unknown".to_owned(),
            dark_model: String::new(),
            winner: None,
        }
    }

    fn opponent_type(&self) -> String {
        format!(
            "{}-vs-{}",
            classify_agent(&self.light),
            classify_agent(&self.dark)
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Opening,
    Placement,
    Movement,
    Late,
}

impl Phase {
    fn name(self) -> &'static str {
        match self {
            Self::Opening => "opening",
            Self::Placement => "placement",
            Self::Movement => "movement",
            Self::Late => "late",
        }
    }

    fn index(self) -> u8 {
        match self {
            Self::Opening => 0,
            Self::Placement => 1,
            Self::Movement => 2,
            Self::Late => 3,
        }
    }
}

#[derive(Clone, Debug)]
struct RootCandidate {
    game_key: String,
    state: GameState,
    phase: Phase,
    played: Action,
    observation: Observation,
}

#[derive(Default)]
struct Bucket {
    seen: u64,
    roots: Vec<RootCandidate>,
}

#[derive(Clone, Copy, Debug)]
struct Config {
    max_games_scan: usize,
    game_sample_mod: u64,
    game_sample_bucket: u64,
    holdout_mod: u64,
    holdout_bucket: u64,
    per_bucket: usize,
    max_roots: usize,
    neutral_quota: usize,
    neutral_target_abs: f32,
    candidate_limit: usize,
    teacher_depth: u8,
    teacher_nodes: u64,
    teacher_beam: usize,
    continuation_plies: u8,
    continuation_mid_plies: u8,
    continuation_long_plies: u8,
    continuation_depth: u8,
    continuation_nodes: u64,
    continuation_beam: usize,
    min_margin: i32,
    allow_exhausted: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_games_scan: 5_000,
            game_sample_mod: 8,
            game_sample_bucket: 0,
            holdout_mod: 4,
            holdout_bucket: 0,
            per_bucket: 4,
            max_roots: 256,
            neutral_quota: 96,
            neutral_target_abs: 0.20,
            candidate_limit: 16,
            teacher_depth: 4,
            teacher_nodes: 8_000,
            teacher_beam: 32,
            continuation_plies: 4,
            continuation_mid_plies: 8,
            continuation_long_plies: 12,
            continuation_depth: 2,
            continuation_nodes: 4_000,
            continuation_beam: 64,
            min_margin: 500,
            allow_exhausted: false,
        }
    }
}

fn parse_args() -> HashMap<String, String> {
    let values = env::args().skip(1).collect::<Vec<_>>();
    let mut parsed = HashMap::new();
    let mut index = 0;
    while index < values.len() {
        let Some(key) = values[index].strip_prefix("--") else {
            index += 1;
            continue;
        };
        if let Some(value) = values.get(index + 1).filter(|item| !item.starts_with("--")) {
            parsed.insert(key.to_owned(), value.clone());
            index += 2;
        } else {
            parsed.insert(key.to_owned(), "true".to_owned());
            index += 1;
        }
    }
    parsed
}

fn path_arg(args: &HashMap<String, String>, key: &str, default: &str) -> PathBuf {
    args.get(key)
        .map_or_else(|| PathBuf::from(default), PathBuf::from)
}

fn number<T: std::str::FromStr>(args: &HashMap<String, String>, key: &str, default: T) -> T {
    args.get(key)
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn classify_agent(agent: &str) -> &'static str {
    let lower = agent.to_ascii_lowercase();
    if lower.contains("random") || lower.contains("coin") {
        "random"
    } else if lower.contains("lunatic") || lower.contains("heuristic") {
        "heuristic"
    } else if lower.contains("gnn")
        || lower.contains("qadv")
        || lower.contains("puct")
        || lower.contains("neural")
    {
        "neural"
    } else if lower.contains("pathfinder")
        || lower.contains("search")
        || lower.contains("transition")
    {
        "search"
    } else {
        "other"
    }
}

fn parse_player(value: &str) -> Option<Player> {
    match value {
        "L" => Some(Player::Light),
        "D" => Some(Player::Dark),
        _ => None,
    }
}

fn load_observations(directory: &Path) -> Result<HashMap<String, Vec<Observation>>, String> {
    let mut paths = fs::read_dir(directory)
        .map_err(|error| format!("read observations directory {directory:?}: {error}"))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("tsv"))
        .collect::<Vec<_>>();
    paths.sort();
    let mut observations: HashMap<String, Vec<Observation>> = HashMap::new();
    for path in paths {
        for (line_number, line) in BufReader::new(
            File::open(&path).map_err(|error| format!("open observation {path:?}: {error}"))?,
        )
        .lines()
        .enumerate()
        {
            let line = line.map_err(|error| format!("read observation {path:?}: {error}"))?;
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields = line.split('\t').collect::<Vec<_>>();
            if fields.len() < 9 {
                return Err(format!(
                    "observation {path:?}:{} has {} fields",
                    line_number + 1,
                    fields.len()
                ));
            }
            let observation = Observation {
                seed64: fields[2].to_owned(),
                light: fields[3].to_owned(),
                light_model: fields[4].to_owned(),
                dark: fields[5].to_owned(),
                dark_model: fields[6].to_owned(),
                winner: parse_player(fields[7]),
            };
            let entries = observations.entry(fields[0].to_owned()).or_default();
            let identity = (
                observation.light.clone(),
                observation.dark.clone(),
                observation.opponent_type(),
            );
            if !entries.iter().any(|item| {
                (item.light.clone(), item.dark.clone(), item.opponent_type()) == identity
            }) {
                entries.push(observation);
            }
            entries.truncate(8);
        }
    }
    Ok(observations)
}

fn stable_hash(text: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in text.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3_u64);
    }
    hash
}

fn partition_for(game_key: &str, config: Config) -> &'static str {
    if config.holdout_mod > 1 && stable_hash(game_key) % config.holdout_mod == config.holdout_bucket
    {
        "heldout"
    } else {
        "train"
    }
}

fn phase(state: GameState) -> Phase {
    let occupied = (state.light | state.dark).count_ones();
    let reserves = u32::from(state.reserve[0]) + u32::from(state.reserve[1]);
    if occupied < 8 {
        Phase::Opening
    } else if reserves == 0 {
        Phase::Movement
    } else if occupied >= 20 {
        Phase::Late
    } else {
        Phase::Placement
    }
}

fn opponent_has_immediate_win(state: GameState) -> bool {
    let opponent = state.turn;
    state
        .legal_actions()
        .into_iter()
        .any(|action| state.apply_legal(action).state.winner == Some(opponent))
}

/// Quiet means no immediate win, no forced immediate loss, and no multi-capture.
/// Single captures remain eligible because they can be strategic transitions.
fn quiet_state(state: GameState) -> bool {
    let actions = state.legal_actions();
    if actions.len() < 2 {
        return false;
    }
    if actions.iter().any(|action| {
        let transition = state.apply_legal(*action);
        transition.state.winner == Some(state.turn) || transition.captured.count_ones() >= 2
    }) {
        return false;
    }
    let risky = actions
        .iter()
        .filter(|action| {
            let next = state.apply_legal(**action).state;
            next.winner != Some(state.turn) && opponent_has_immediate_win(next)
        })
        .count();
    risky < actions.len()
}

fn add_candidate(buckets: &mut BTreeMap<String, Bucket>, candidate: RootCandidate, config: Config) {
    let key = format!(
        "{}|{}|{}|{}",
        partition_for(&candidate.game_key, config),
        candidate.phase.name(),
        candidate.state.turn.as_str(),
        candidate.observation.opponent_type()
    );
    let bucket = buckets.entry(key.clone()).or_default();
    bucket.seen = bucket.seen.saturating_add(1);
    if bucket.roots.len() < config.per_bucket {
        bucket.roots.push(candidate);
        return;
    }
    let slot = (stable_hash(&format!("{}:{}", key, bucket.seen)) % bucket.seen) as usize;
    if slot < bucket.roots.len() {
        bucket.roots[slot] = candidate;
    }
}

fn select_roots(
    buckets: BTreeMap<String, Bucket>,
    max_roots: usize,
    config: Config,
) -> Vec<RootCandidate> {
    let mut strata = BTreeMap::<(u8, u8, u8), Vec<RootCandidate>>::new();
    for (_key, bucket) in buckets {
        for root in bucket.roots {
            strata
                .entry((
                    u8::from(partition_for(&root.game_key, config) == "heldout"),
                    root.phase.index(),
                    root.state.turn.index() as u8,
                ))
                .or_default()
                .push(root);
        }
    }
    let mut groups = strata.into_values().collect::<Vec<_>>();
    for roots in &mut groups {
        roots.sort_by(|left, right| {
            left.observation
                .opponent_type()
                .cmp(&right.observation.opponent_type())
                .then_with(|| left.game_key.cmp(&right.game_key))
                .then_with(|| left.state.ply.cmp(&right.state.ply))
        });
    }
    let mut selected: Vec<RootCandidate> = Vec::new();
    let mut cursor = 0;
    while selected.len() < max_roots {
        let mut advanced = false;
        for roots in &mut groups {
            if cursor < roots.len() && selected.len() < max_roots {
                selected.push(roots[cursor].clone());
                advanced = true;
            }
        }
        if !advanced {
            break;
        }
        cursor += 1;
    }
    selected
}

fn candidate_actions(state: GameState, limit: usize, weights: EvaluationWeights) -> Vec<Action> {
    let ordered = ordered_root_actions(state, state.turn, weights);
    if ordered.len() <= limit {
        return ordered;
    }
    let mut selected = ordered.iter().copied().take(limit / 2).collect::<Vec<_>>();
    let spread_count = limit.saturating_sub(selected.len());
    for index in 0..spread_count {
        let position = index * (ordered.len() - 1) / spread_count.max(1);
        let action = ordered[position];
        if !selected.contains(&action) {
            selected.push(action);
        }
    }
    selected.truncate(limit);
    selected
}

fn normalized_score(score: i32) -> f32 {
    (score as f32 / 5_000.0).tanh()
}

fn softmax(values: &[f32], temperature: f32) -> Vec<f32> {
    if values.is_empty() {
        return Vec::new();
    }
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let weights = values
        .iter()
        .map(|value| ((*value - maximum) / temperature.max(0.001)).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f32>().max(1.0e-8);
    weights.into_iter().map(|value| value / total).collect()
}

fn rollout_values(
    state: GameState,
    root_player: Player,
    horizons: [u8; 3],
    config: SearchConfig,
) -> [f32; 3] {
    let mut state = state;
    let mut values = [f32::NAN; 3];
    let max_horizon = horizons.into_iter().max().unwrap_or(0);
    for ply in 0..max_horizon {
        if let Some(winner) = state.winner {
            let value = if winner == root_player { 1.0 } else { -1.0 };
            values.fill(value);
            return values;
        }
        let result = search_best_action_with_tactical_filter(state, config);
        let Some(action) = result.action else {
            values.fill(0.0);
            return values;
        };
        state = state.apply_legal(action).state;
        for (index, horizon) in horizons.into_iter().enumerate() {
            if ply + 1 == horizon {
                values[index] = state_value(state, root_player, config.weights);
            }
        }
    }
    for value in &mut values {
        if value.is_nan() {
            *value = state_value(state, root_player, config.weights);
        }
    }
    values
}

fn state_value(state: GameState, player: Player, weights: EvaluationWeights) -> f32 {
    (pathagon_engine::search::evaluate(state, player, weights) as f32 / 5_000.0).tanh()
}

fn export_root(root: &RootCandidate, config: Config) -> Option<serde_json::Value> {
    let weights = EvaluationWeights {
        path: 241,
        material: 112,
        capture: 887,
        structure: 40,
        threat: 154,
        edge: 74,
    };
    let teacher_config = SearchConfig {
        depth: config.teacher_depth,
        max_nodes: config.teacher_nodes,
        beam_width: config.teacher_beam,
        weights,
        tactical_proof_horizon: None,
    };
    let continuation_config = SearchConfig {
        depth: config.continuation_depth,
        max_nodes: config.continuation_nodes,
        beam_width: config.continuation_beam,
        weights,
        tactical_proof_horizon: None,
    };
    let actions = candidate_actions(root.state, config.candidate_limit, weights);
    if actions.len() < 2 {
        return None;
    }
    let evaluations = actions
        .iter()
        .copied()
        .filter_map(|action| analyze_action(root.state, action, teacher_config).ok())
        .collect::<Vec<_>>();
    if evaluations.len() < 2 {
        return None;
    }
    let mut ranked = evaluations.clone();
    ranked.sort_by(|left, right| {
        right
            .score
            .cmp(&left.score)
            .then_with(|| left.action.order().cmp(&right.action.order()))
    });
    let top = ranked[0];
    let second = ranked[1];
    if !config.allow_exhausted && (top.exhausted || second.exhausted) {
        return None;
    }
    let score_margin = top.score.saturating_sub(second.score);
    if score_margin < config.min_margin {
        return None;
    }
    let teacher_values = ranked
        .iter()
        .map(|item| normalized_score(item.score))
        .collect::<Vec<_>>();
    let soft_policy = softmax(&teacher_values, 0.15);
    let horizons = [
        config.continuation_plies,
        config.continuation_mid_plies,
        config.continuation_long_plies,
    ];
    let traces = ranked
        .iter()
        .map(|item| {
            let next = root.state.apply_legal(item.action).state;
            rollout_values(next, root.state.turn, horizons, continuation_config)
        })
        .collect::<Vec<_>>();
    let short_values = traces.iter().map(|values| values[0]).collect::<Vec<_>>();
    let mid_values = traces.iter().map(|values| values[1]).collect::<Vec<_>>();
    let long_values = traces.iter().map(|values| values[2]).collect::<Vec<_>>();
    let outcomes = traces
        .iter()
        .map(|values| 0.50 * values[0] + 0.30 * values[1] + 0.20 * values[2])
        .collect::<Vec<_>>();
    let has_outcome_difference = outcomes
        .iter()
        .any(|outcome| (*outcome - outcomes[0]).abs() > 0.05);
    let source_outcome =
        root.observation.winner.map_or(
            0_i8,
            |winner| if winner == root.state.turn { 1 } else { -1 },
        );
    let expected_continuation_value = soft_policy
        .iter()
        .zip(outcomes.iter())
        .map(|(probability, outcome)| *probability * *outcome)
        .sum::<f32>();
    let value_target =
        (0.75 * expected_continuation_value + 0.25 * f32::from(source_outcome)).clamp(-1.0, 1.0);
    let target_class = if value_target.abs() <= config.neutral_target_abs {
        "neutral"
    } else {
        "decisive"
    };
    Some(json!({
        "schemaVersion": 1,
        "id": format!("{}-{}-{}", root.game_key, root.state.ply, root.observation.opponent_type()),
        "gameKey": root.game_key,
        "sourceSeed64": root.observation.seed64,
        "state": encode_state(root.state),
        "ply": root.state.ply,
        "turn": root.state.turn.as_str(),
        "phase": root.phase.name(),
        "opponentType": root.observation.opponent_type(),
        "lightAgent": root.observation.light,
        "lightModel": root.observation.light_model,
        "darkAgent": root.observation.dark,
        "darkModel": root.observation.dark_model,
        "partition": partition_for(&root.game_key, config),
        "targetClass": target_class,
        "legalActions": root.state.legal_actions().into_iter().map(encode_action).collect::<Vec<_>>(),
        "candidateActions": ranked.iter().map(|item| encode_action(item.action)).collect::<Vec<_>>(),
        "teacherScores": ranked.iter().map(|item| item.score).collect::<Vec<_>>(),
        "teacherValues": teacher_values,
        "softPolicy": soft_policy,
        "continuationValues": outcomes,
        "continuationShortValues": short_values,
        "continuationMidValues": mid_values,
        "continuationLongValues": long_values,
        "expectedContinuationValue": expected_continuation_value,
        "valueTarget": value_target,
        "valueCalibration": {"continuationWeight": 0.75, "sourceOutcomeWeight": 0.25},
        "sourceOutcome": source_outcome,
        "playedAction": encode_action(root.played),
        "teacherBest": encode_action(top.action),
        "teacherMargin": score_margin,
        "teacherExhausted": ranked.iter().filter(|item| item.exhausted).count(),
        "teacherNodes": ranked.iter().map(|item| item.nodes).sum::<u64>(),
        "teacherDepth": config.teacher_depth,
        "teacherNodeBudget": config.teacher_nodes,
        "continuationPlies": horizons,
        "continuationDifference": has_outcome_difference,
    }))
}

fn select_labeled_rows(
    rows: Vec<serde_json::Value>,
    max_roots: usize,
    neutral_quota: usize,
) -> Vec<serde_json::Value> {
    let mut groups = BTreeMap::<String, Vec<serde_json::Value>>::new();
    for row in rows {
        let partition = row["partition"].as_str().unwrap_or("train");
        let target_class = row["targetClass"].as_str().unwrap_or("decisive");
        let phase = row["phase"].as_str().unwrap_or("unknown");
        let turn = row["turn"].as_str().unwrap_or("unknown");
        groups
            .entry(format!("{target_class}|{partition}|{phase}|{turn}"))
            .or_default()
            .push(row);
    }
    let mut selected: Vec<serde_json::Value> = Vec::new();
    for target_class in ["neutral", "decisive"] {
        let quota = if target_class == "neutral" {
            neutral_quota.min(max_roots)
        } else {
            max_roots.saturating_sub(neutral_quota.min(max_roots))
        };
        let partition_quota = quota / 2;
        for partition in ["heldout", "train"] {
            let target = if partition == "train" {
                quota.saturating_sub(partition_quota)
            } else {
                partition_quota
            };
            let mut partition_groups = groups
                .iter_mut()
                .filter(|(key, _)| key.starts_with(&format!("{target_class}|{partition}|")))
                .map(|(_, rows)| rows)
                .collect::<Vec<_>>();
            let mut cursor = 0;
            let mut taken = 0;
            while selected.len() < max_roots && taken < target {
                let mut advanced = false;
                for rows in &mut partition_groups {
                    if cursor < rows.len() && taken < target {
                        selected.push(rows[cursor].clone());
                        taken += 1;
                        advanced = true;
                    }
                }
                if !advanced {
                    break;
                }
                cursor += 1;
            }
        }
    }
    selected
}

fn main() -> Result<(), String> {
    let args = parse_args();
    let games_dir = path_arg(&args, "games-dir", "data/corpora/games-v1/games");
    let observations_dir = path_arg(
        &args,
        "observations-dir",
        "data/corpora/games-v1/observations",
    );
    let output = path_arg(
        &args,
        "output",
        "research/20260910-quiet-regret-value/workspace/quiet-targets.jsonl",
    );
    let config = Config {
        max_games_scan: number(&args, "max-games-scan", Config::default().max_games_scan),
        game_sample_mod: number(&args, "game-sample-mod", Config::default().game_sample_mod),
        game_sample_bucket: number(
            &args,
            "game-sample-bucket",
            Config::default().game_sample_bucket,
        ),
        holdout_mod: number(&args, "holdout-mod", Config::default().holdout_mod),
        holdout_bucket: number(&args, "holdout-bucket", Config::default().holdout_bucket),
        per_bucket: number(&args, "per-bucket", Config::default().per_bucket),
        max_roots: number(&args, "max-roots", Config::default().max_roots),
        neutral_quota: number(&args, "neutral-quota", Config::default().neutral_quota),
        neutral_target_abs: number(
            &args,
            "neutral-target-abs",
            Config::default().neutral_target_abs,
        ),
        candidate_limit: number(&args, "candidate-limit", Config::default().candidate_limit),
        teacher_depth: number(&args, "teacher-depth", Config::default().teacher_depth),
        teacher_nodes: number(&args, "teacher-nodes", Config::default().teacher_nodes),
        teacher_beam: number(&args, "teacher-beam", Config::default().teacher_beam),
        continuation_plies: number(
            &args,
            "continuation-plies",
            Config::default().continuation_plies,
        ),
        continuation_mid_plies: number(
            &args,
            "continuation-mid-plies",
            Config::default().continuation_mid_plies,
        ),
        continuation_long_plies: number(
            &args,
            "continuation-long-plies",
            Config::default().continuation_long_plies,
        ),
        continuation_depth: number(
            &args,
            "continuation-depth",
            Config::default().continuation_depth,
        ),
        continuation_nodes: number(
            &args,
            "continuation-nodes",
            Config::default().continuation_nodes,
        ),
        continuation_beam: number(
            &args,
            "continuation-beam",
            Config::default().continuation_beam,
        ),
        min_margin: number(&args, "min-margin", Config::default().min_margin),
        allow_exhausted: args.contains_key("allow-exhausted"),
    };
    let observations = load_observations(&observations_dir)?;
    let mut game_paths = fs::read_dir(&games_dir)
        .map_err(|error| format!("read games directory {games_dir:?}: {error}"))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some("tsv"))
        .collect::<Vec<_>>();
    game_paths.sort();
    let mut buckets = BTreeMap::<String, Bucket>::new();
    let mut games = 0_usize;
    let mut quiet = 0_usize;
    for path in game_paths {
        for line in BufReader::new(
            File::open(&path).map_err(|error| format!("open game {path:?}: {error}"))?,
        )
        .lines()
        {
            let line = line.map_err(|error| format!("read game {path:?}: {error}"))?;
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let game = parse_unified_game(&line)
                .map_err(|error| format!("parse game {path:?}: {error}"))?;
            if game.config.board_size != 7 || game.config.reserve_per_player != 14 {
                continue;
            }
            if config.game_sample_mod > 1
                && stable_hash(&game.key) % config.game_sample_mod != config.game_sample_bucket
            {
                continue;
            }
            games += 1;
            let profiles = observations
                .get(&game.key)
                .cloned()
                .unwrap_or_else(|| vec![Observation::unknown()]);
            for observation in profiles {
                let mut state = GameState::with_config(
                    game.config
                        .with_max_plies(196)
                        .map_err(|error| error.to_string())?,
                );
                for (index, played) in game.actions.iter().copied().enumerate() {
                    if state.ply < 4 {
                        state = state.apply_legal(played).state;
                        continue;
                    }
                    if quiet_state(state) {
                        quiet += 1;
                        let current_phase = phase(state);
                        add_candidate(
                            &mut buckets,
                            RootCandidate {
                                game_key: game.key.clone(),
                                state,
                                phase: current_phase,
                                played,
                                observation: observation.clone(),
                            },
                            config,
                        );
                    }
                    state = state.apply_legal(played).state;
                    if index + 1 >= game.actions.len() {
                        break;
                    }
                }
            }
        }
        if config.max_games_scan > 0 && games >= config.max_games_scan {
            break;
        }
    }
    let bucket_summary = buckets
        .iter()
        .map(|(key, bucket)| json!({"bucket": key, "seen": bucket.seen, "retained": bucket.roots.len()}))
        .collect::<Vec<_>>();
    // Over-sample each phase/turn stratum before labeling. Some early roots
    // will be rejected when the teacher exhausts its budget or finds no
    // material regret; over-sampling prevents those rejections from erasing
    // an entire phase from the final dataset.
    let roots = select_roots(buckets, config.max_roots.saturating_mul(4), config);
    output
        .parent()
        .map(fs::create_dir_all)
        .transpose()
        .map_err(|error| format!("create output directory: {error}"))?;
    let mut writer = BufWriter::new(
        File::create(&output).map_err(|error| format!("create output {output:?}: {error}"))?,
    );
    let mut labeled = Vec::new();
    let mut rejected = 0_usize;
    for root in &roots {
        if let Some(row) = export_root(root, config) {
            labeled.push(row);
        } else {
            rejected += 1;
        }
    }
    let selected = select_labeled_rows(labeled, config.max_roots, config.neutral_quota);
    for row in &selected {
        writeln!(writer, "{}", row).map_err(|error| format!("write output: {error}"))?;
    }
    writer
        .flush()
        .map_err(|error| format!("flush output: {error}"))?;
    println!(
        "{}",
        json!({
            "games": games,
            "quietPositionsSeen": quiet,
            "rootsSampled": roots.len(),
            "rootsLabeled": selected.len(),
            "rootsRejected": rejected,
            "buckets": bucket_summary,
            "output": output,
            "config": {
                "perBucket": config.per_bucket,
                "maxGamesScan": config.max_games_scan,
                "gameSampleMod": config.game_sample_mod,
                "gameSampleBucket": config.game_sample_bucket,
                "holdoutMod": config.holdout_mod,
                "holdoutBucket": config.holdout_bucket,
                "maxRoots": config.max_roots,
                "neutralQuota": config.neutral_quota,
                "neutralTargetAbs": config.neutral_target_abs,
                "candidateLimit": config.candidate_limit,
                "teacherDepth": config.teacher_depth,
                "teacherNodes": config.teacher_nodes,
                "teacherBeam": config.teacher_beam,
                "continuationPlies": config.continuation_plies,
                "continuationMidPlies": config.continuation_mid_plies,
                "continuationLongPlies": config.continuation_long_plies,
                "continuationDepth": config.continuation_depth,
                "continuationNodes": config.continuation_nodes,
                "continuationBeam": config.continuation_beam,
                "minMargin": config.min_margin,
                "allowExhausted": config.allow_exhausted,
            }
        })
    );
    Ok(())
}
