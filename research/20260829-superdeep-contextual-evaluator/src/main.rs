use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use pathagon_engine::contract::ContractAction;
use pathagon_engine::corpus::{decode_state, encode_state, parse_unified_game};
use pathagon_engine::search::{
    evaluate, search_best_action_with_tactical_filter, tactical_root_safe_actions,
    EvaluationWeights, SearchConfig,
};
use pathagon_engine::{Action, GameState};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const V05_WEIGHTS: EvaluationWeights = EvaluationWeights {
    path: 241,
    material: 112,
    capture: 887,
    structure: 40,
    threat: 154,
    edge: 74,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Root {
    schema_version: u8,
    id: String,
    source_game_id: String,
    source_ply: u16,
    phase: String,
    partition: String,
    state: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TargetAction {
    action: ContractAction,
    features: [i32; 6],
    #[serde(rename = "captureCount")]
    capture_count: u8,
    #[serde(rename = "immediateWin")]
    immediate_win: bool,
    safe: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Target {
    #[serde(rename = "schemaVersion")]
    schema_version: u8,
    id: String,
    #[serde(rename = "sourceGameId")]
    source_game_id: String,
    #[serde(rename = "sourcePly")]
    source_ply: u16,
    phase: String,
    partition: String,
    state: String,
    teacher: Teacher,
    #[serde(rename = "teacherAction")]
    teacher_action: ContractAction,
    #[serde(rename = "teacherScore")]
    teacher_score: i32,
    #[serde(rename = "teacherNodes")]
    teacher_nodes: u64,
    #[serde(rename = "completedDepth")]
    completed_depth: u8,
    exhausted: bool,
    actions: Vec<TargetAction>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Teacher {
    id: String,
    depth: u8,
    #[serde(rename = "maxNodes")]
    max_nodes: u64,
    #[serde(rename = "beamWidth")]
    beam_width: usize,
    weights: EvaluationWeights,
}

fn main() {
    let args = args();
    let result = match args.get("command").map(String::as_str) {
        Some("generate") => generate(&args),
        Some("label") => label(&args),
        Some("audit") => audit(&args),
        _ => Err("use --command generate, label, or audit".to_owned()),
    };
    if let Err(error) = result {
        eprintln!("superdeep-contextual: {error}");
        std::process::exit(2);
    }
}

fn audit(args: &HashMap<String, String>) -> Result<(), String> {
    let games = required(args, "games")?;
    let file = fs::File::open(&games).map_err(|error| format!("read {games:?}: {error}"))?;
    let mut checked_games = 0_u32;
    let mut checked_plies = 0_u32;
    let mut draws = 0_u32;
    for (line_number, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|error| format!("read {games:?}: {error}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let game: Value = serde_json::from_str(&line)
            .map_err(|error| format!("{games:?} line {} is not JSON: {error}", line_number + 1))?;
        let config = game
            .get("config")
            .ok_or_else(|| format!("{games:?} line {} missing config", line_number + 1))?;
        let board_size = config
            .get("boardSize")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("{games:?} line {} missing boardSize", line_number + 1))?
            as u8;
        let reserve = config
            .get("reservePerPlayer")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("{games:?} line {} missing reserve", line_number + 1))?
            as u8;
        let mut state = GameState::with_board_size(board_size);
        if state.config.reserve_per_player != reserve {
            state = GameState::with_config(
                pathagon_engine::BoardConfig::new(board_size, reserve).map_err(|error| {
                    format!("invalid config on line {}: {error}", line_number + 1)
                })?,
            );
        }
        let moves = game
            .get("moves")
            .and_then(Value::as_array)
            .ok_or_else(|| format!("{games:?} line {} missing moves", line_number + 1))?;
        for (ply, move_value) in moves.iter().enumerate() {
            let contract_action: ContractAction =
                serde_json::from_value(move_value.get("action").cloned().ok_or_else(|| {
                    format!(
                        "{games:?} line {} ply {} missing action",
                        line_number + 1,
                        ply
                    )
                })?)
                .map_err(|error| {
                    format!(
                        "{games:?} line {} ply {ply} invalid action: {error}",
                        line_number + 1
                    )
                })?;
            let action = Action::from(contract_action);
            let expected_captured = move_value
                .get("captured")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    format!(
                        "{games:?} line {} ply {ply} missing captured",
                        line_number + 1
                    )
                })?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .ok_or_else(|| {
                            format!(
                                "{games:?} line {} ply {ply} invalid captured square",
                                line_number + 1
                            )
                        })
                        .map(|square| square as u8)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let transition = state.apply(action).map_err(|error| {
                format!(
                    "{games:?} line {} ply {ply} illegal action {action}: {error}",
                    line_number + 1
                )
            })?;
            let actual_captured = pathagon_engine::bit_squares(transition.captured)
                .into_iter()
                .collect::<Vec<_>>();
            if actual_captured != expected_captured {
                return Err(format!(
                    "{games:?} line {} ply {ply} capture mismatch: expected {expected_captured:?}, actual {actual_captured:?}",
                    line_number + 1
                ));
            }
            state = transition.state;
            checked_plies += 1;
        }
        if game.get("winner").is_some_and(Value::is_null) {
            draws += 1;
        }
        checked_games += 1;
    }
    println!(
        "{{\"games\":{},\"plies\":{},\"draws\":{},\"legal\":true}}",
        checked_games, checked_plies, draws
    );
    Ok(())
}

fn generate(args: &HashMap<String, String>) -> Result<(), String> {
    let games_dir = required(args, "games-dir")?;
    let output = required(args, "output")?;
    let limit = number(args, "games", 480_usize);
    let turn_balanced = args.contains_key("turn-balanced");
    let excluded_sources = args
        .get("exclude-roots")
        .map(PathBuf::from)
        .map(|path| {
            let file = fs::File::open(&path)
                .map_err(|error| format!("read exclude roots {}: {error}", path.display()))?;
            BufReader::new(file)
                .lines()
                .filter(|line| line.as_ref().is_ok_and(|line| !line.trim().is_empty()))
                .map(|line| {
                    let line = line.map_err(|error| error.to_string())?;
                    let root: Root = serde_json::from_str(&line).map_err(|error| {
                        format!("parse exclude root in {}: {error}", path.display())
                    })?;
                    Ok(root.source_game_id)
                })
                .collect::<Result<HashSet<_>, String>>()
        })
        .transpose()?;
    let mut games = Vec::new();
    let mut shards = fs::read_dir(&games_dir)
        .map_err(|error| format!("read {}: {error}", games_dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "tsv"))
        .collect::<Vec<_>>();
    shards.sort();
    for shard in shards {
        for line in
            BufReader::new(fs::File::open(&shard).map_err(|error| error.to_string())?).lines()
        {
            let line = line.map_err(|error| error.to_string())?;
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let game = parse_unified_game(&line)?;
            if game.config.board_size == 7
                && game.config.reserve_per_player == 14
                && game.actions.len() >= 8
                && excluded_sources
                    .as_ref()
                    .is_none_or(|excluded| !excluded.contains(&game.key))
            {
                games.push(game);
            }
        }
    }
    if games.is_empty() {
        return Err("no eligible 7x7 games".to_owned());
    }
    let eligible_games = games.len();
    let stride = (eligible_games / limit.max(1)).max(1);
    let mut roots = Vec::new();
    for (index, game) in games.into_iter().enumerate().step_by(stride).take(limit) {
        let requested_ply = if turn_balanced {
            match index % 8 {
                0 => 4,
                1 => 15,
                2 => 32,
                3 => 63,
                4 => 5,
                5 => 14,
                6 => 33,
                _ => 64,
            }
        } else {
            match index % 4 {
                0 => 4,
                1 => 14,
                2 => 32,
                _ => 64,
            }
        };
        let mut ply = requested_ply.min(game.actions.len().saturating_sub(1));
        if turn_balanced && ply % 2 != (roots.len() % 2) {
            ply = ply.saturating_sub(1);
        }
        let mut state = GameState::with_config(game.config);
        for action in game.actions.iter().take(ply) {
            state = state.apply(*action).map_err(str::to_owned)?.state;
        }
        roots.push(Root {
            schema_version: 1,
            id: format!("{}:{ply}", game.key),
            source_game_id: game.key,
            source_ply: ply as u16,
            phase: phase(state),
            partition: if stable_hash(&roots.len().to_string()) % 5 == 0 {
                "heldout".to_owned()
            } else {
                "train".to_owned()
            },
            state: encode_state(state),
        });
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut writer = BufWriter::new(fs::File::create(&output).map_err(|error| error.to_string())?);
    for root in &roots {
        serde_json::to_writer(&mut writer, root).map_err(|error| error.to_string())?;
        writer.write_all(b"\n").map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    let heldout = roots
        .iter()
        .filter(|root| root.partition == "heldout")
        .count();
    println!(
        "{}",
        serde_json::json!({
            "roots": roots.len(),
            "heldout": heldout,
            "eligibleGames": eligible_games,
            "excludedSources": excluded_sources.as_ref().map_or(0, HashSet::len),
            "output": output
        })
    );
    Ok(())
}

fn label(args: &HashMap<String, String>) -> Result<(), String> {
    let roots_path = required(args, "roots")?;
    let output = required(args, "output")?;
    let offset = number(args, "offset", 0_usize);
    let limit = number(args, "limit", usize::MAX);
    let config = SearchConfig {
        depth: number(args, "teacher-depth", 7_u8),
        max_nodes: number(args, "teacher-nodes", 500_000_u64),
        beam_width: number(args, "teacher-beam", 32_usize),
        weights: V05_WEIGHTS,
        tactical_proof_horizon: None,
    };
    let teacher = Teacher {
        id: format!(
            "pathfinder-teacher-v2-depth{}-{}k-beam{}",
            config.depth,
            config.max_nodes / 1_000,
            config.beam_width
        ),
        depth: config.depth,
        max_nodes: config.max_nodes,
        beam_width: config.beam_width,
        weights: config.weights,
    };
    let source = BufReader::new(fs::File::open(&roots_path).map_err(|error| error.to_string())?);
    let roots = source
        .lines()
        .map(|line| {
            serde_json::from_str::<Root>(&line.map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let selected = roots
        .into_iter()
        .skip(offset)
        .take(limit)
        .collect::<Vec<_>>();
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut writer = BufWriter::new(fs::File::create(&output).map_err(|error| error.to_string())?);
    for (index, root) in selected.iter().enumerate() {
        let state = decode_state(&root.state)?;
        let legal = state.legal_actions();
        let safe = tactical_root_safe_actions(state, state.turn, V05_WEIGHTS);
        let result = search_best_action_with_tactical_filter(state, config);
        let teacher_action = result
            .action
            .ok_or_else(|| format!("{} has no teacher action", root.id))?;
        let actions = legal
            .into_iter()
            .map(|action| {
                let transition = state.apply_legal(action);
                let immediate_win = transition.state.winner == Some(state.turn);
                let features = if immediate_win {
                    [0; 6]
                } else {
                    [
                        unit_feature(transition.state, state.turn, 0),
                        unit_feature(transition.state, state.turn, 1),
                        unit_feature(transition.state, state.turn, 2),
                        unit_feature(transition.state, state.turn, 3),
                        unit_feature(transition.state, state.turn, 4),
                        unit_feature(transition.state, state.turn, 5),
                    ]
                };
                TargetAction {
                    action: action.into(),
                    features,
                    capture_count: transition.captured.count_ones() as u8,
                    immediate_win,
                    safe: safe.contains(&action),
                }
            })
            .collect::<Vec<_>>();
        let target = Target {
            schema_version: 1,
            id: root.id.clone(),
            source_game_id: root.source_game_id.clone(),
            source_ply: root.source_ply,
            phase: root.phase.clone(),
            partition: root.partition.clone(),
            state: root.state.clone(),
            teacher: teacher.clone(),
            teacher_action: teacher_action.into(),
            teacher_score: result.score,
            teacher_nodes: result.nodes,
            completed_depth: result.completed_depth,
            exhausted: result.exhausted,
            actions,
        };
        serde_json::to_writer(&mut writer, &target).map_err(|error| error.to_string())?;
        writer.write_all(b"\n").map_err(|error| error.to_string())?;
        if (index + 1) % 20 == 0 {
            eprintln!(
                "superdeep-contextual: labeled {}/{}",
                index + 1,
                selected.len()
            );
        }
    }
    writer.flush().map_err(|error| error.to_string())?;
    println!(
        "{}",
        serde_json::json!({"roots": selected.len(), "teacher": teacher, "output": output})
    );
    Ok(())
}

fn unit_feature(state: GameState, player: pathagon_engine::Player, index: usize) -> i32 {
    let mut weights = EvaluationWeights {
        path: 0,
        material: 0,
        capture: 0,
        structure: 0,
        threat: 0,
        edge: 0,
    };
    match index {
        0 => weights.path = 1,
        1 => weights.material = 1,
        2 => weights.capture = 1,
        3 => weights.structure = 1,
        4 => weights.threat = 1,
        _ => weights.edge = 1,
    }
    evaluate(state, player, weights)
}

fn phase(state: GameState) -> String {
    let occupied = (state.light | state.dark).count_ones();
    let reserves = u32::from(state.reserve[0]) + u32::from(state.reserve[1]);
    if occupied < 8 {
        "opening".to_owned()
    } else if reserves == 0 {
        "movement".to_owned()
    } else if occupied >= 20 {
        "late-game".to_owned()
    } else {
        "placement".to_owned()
    }
}

fn stable_hash(value: &str) -> u64 {
    value
        .bytes()
        .fold(14_695_981_039_346_656_037_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(1_099_511_628_211)
        })
}

fn args() -> HashMap<String, String> {
    let values = env::args().skip(1).collect::<Vec<_>>();
    let mut parsed = HashMap::new();
    let mut index = 0;
    while index < values.len() {
        if let Some(key) = values[index].strip_prefix("--") {
            if let Some(value) = values
                .get(index + 1)
                .filter(|value| !value.starts_with("--"))
            {
                parsed.insert(key.to_owned(), value.clone());
                index += 1;
            } else {
                // Preserve standalone switches such as `--turn-balanced`.
                parsed.insert(key.to_owned(), "true".to_owned());
            }
        }
        index += 1;
    }
    parsed
}

fn required(args: &HashMap<String, String>, key: &str) -> Result<PathBuf, String> {
    args.get(key)
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing --{key}"))
}

fn number<T: std::str::FromStr + Copy>(
    args: &HashMap<String, String>,
    key: &str,
    fallback: T,
) -> T {
    args.get(key)
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}
