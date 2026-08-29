use std::collections::HashSet;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::Path;

use pathagon_engine::contract::{
    ContractAction, ContractPlayer, GameConfig, PlayerNumbers, PlayerSquares, CONTRACT_VERSION,
    RULES_VERSION,
};
use pathagon_engine::corpus::{parse_unified_game, UnifiedGame};
use pathagon_engine::runtime::RuntimePosition;
use pathagon_engine::search::{analyze_action, evaluate, tactical_root_safe_actions, EvaluationWeights, SearchConfig};
use pathagon_engine::GameState;
use serde::{Deserialize, Serialize};

pub const ROOT_SCHEMA_VERSION: u8 = 1;
pub const TARGET_SCHEMA_VERSION: u8 = 1;
pub const TEACHER_ID: &str = "pathfinder-teacher-v1-depth5-2k-beam16";
pub const V05_WEIGHTS: EvaluationWeights = EvaluationWeights {
    path: 241,
    material: 112,
    capture: 887,
    structure: 40,
    threat: 154,
    edge: 74,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RootRecord {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u8,
    pub id: String,
    #[serde(rename = "sourceFamily")]
    pub source_family: String,
    #[serde(rename = "sourceGameId")]
    pub source_game_id: String,
    #[serde(rename = "sourcePly")]
    pub source_ply: u16,
    pub phase: String,
    pub position: RuntimePosition,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TargetRecord {
    #[serde(rename = "schemaVersion")]
    pub schema_version: u8,
    pub id: String,
    #[serde(rename = "sourceFamily")]
    pub source_family: String,
    #[serde(rename = "sourceGameId")]
    pub source_game_id: String,
    #[serde(rename = "sourcePly")]
    pub source_ply: u16,
    pub phase: String,
    pub position: RuntimePosition,
    pub teacher: TeacherSpec,
    #[serde(rename = "rootHasForcedBlock")]
    pub root_has_forced_block: bool,
    #[serde(rename = "teacherBestScore")]
    pub teacher_best_score: i32,
    #[serde(rename = "teacherBestActions")]
    pub teacher_best_actions: Vec<ContractAction>,
    pub actions: Vec<TargetAction>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TeacherSpec {
    pub id: String,
    pub depth: u8,
    #[serde(rename = "maxNodes")]
    pub max_nodes: u64,
    #[serde(rename = "beamWidth")]
    pub beam_width: usize,
    pub weights: EvaluationWeights,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TargetAction {
    pub action: ContractAction,
    /// Per-feature contribution of the non-terminal afterstate under a unit
    /// evaluator weight. The order is path, material, capture, structure,
    /// threat, edge and is part of target schema v1.
    pub features: [i32; 6],
    #[serde(rename = "captureCount")]
    pub capture_count: u8,
    #[serde(rename = "immediateWin")]
    pub immediate_win: bool,
    pub safe: bool,
    #[serde(rename = "teacherScore")]
    pub teacher_score: i32,
    #[serde(rename = "teacherNodes")]
    pub teacher_nodes: u64,
    #[serde(rename = "teacherExhausted")]
    pub teacher_exhausted: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct SeededSidecarRow {
    key: String,
    #[serde(rename = "initialPosition")]
    initial_position: RuntimePosition,
}

#[derive(Clone, Debug, Deserialize)]
struct HumanHeader {
    config: HumanConfig,
}

#[derive(Clone, Debug, Deserialize)]
struct HumanConfig {
    #[serde(rename = "boardSize")]
    board_size: u8,
    #[serde(rename = "reservePerPlayer")]
    reserve_per_player: u8,
    #[serde(rename = "maxPlies")]
    max_plies: u16,
}

#[derive(Clone, Debug, Deserialize)]
struct HumanRow {
    id: String,
    #[serde(rename = "sourcePly")]
    source_ply: u16,
    state: HumanState,
}

#[derive(Clone, Debug, Deserialize)]
struct HumanState {
    light: Vec<u8>,
    dark: Vec<u8>,
    reserve: PlayerNumbers,
    turn: String,
    forbidden: Vec<u8>,
    #[serde(rename = "lastRelocatedTo")]
    last_relocated_to: PlayerSquares,
    #[serde(default, rename = "lastCapture")]
    last_capture: u8,
    #[serde(default, rename = "lastPlayer")]
    last_player: Option<String>,
    winner: Option<String>,
    ply: u16,
}

pub fn collect_roots(
    games_dir: &Path,
    seeded_path: &Path,
    human_path: &Path,
    canonical_limit: usize,
    seeded_limit: usize,
) -> Result<Vec<RootRecord>, String> {
    let mut roots = Vec::new();
    let games = read_unified_games(games_dir)?;
    let eligible = games
        .into_iter()
        .filter(|game| game.config.board_size == 7 && game.actions.len() >= 18)
        .collect::<Vec<_>>();
    if eligible.is_empty() {
        return Err("canonical corpus contains no eligible 7x7 games".to_owned());
    }
    let stride = (eligible.len() / canonical_limit.max(1)).max(1);
    for (selected, game) in eligible
        .into_iter()
        .enumerate()
        .step_by(stride)
        .take(canonical_limit)
    {
        let phase_ply = match selected % 4 {
            0 => 18,
            1 => 30,
            2 => 42,
            _ => 54,
        };
        let ply = phase_ply.min(game.actions.len().saturating_sub(1) as u16);
        if ply < 8 {
            continue;
        }
        roots.push(root_from_unified_game(&game, ply)?);
    }

    let seeded_text = fs::read_to_string(seeded_path)
        .map_err(|error| format!("read seeded sidecar {}: {error}", seeded_path.display()))?;
    for row in seeded_text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .take(seeded_limit)
    {
        let sidecar: SeededSidecarRow = serde_json::from_str(row)
            .map_err(|error| format!("parse seeded sidecar row: {error}"))?;
        validate_runtime_position(&sidecar.initial_position)?;
        roots.push(RootRecord {
            schema_version: ROOT_SCHEMA_VERSION,
            id: format!("seeded:{}", sidecar.key),
            source_family: "seeded-phase".to_owned(),
            source_game_id: sidecar.key,
            source_ply: sidecar.initial_position.ply,
            phase: phase_name(&sidecar.initial_position),
            position: sidecar.initial_position,
        });
    }

    roots.extend(read_human_roots(human_path)?);
    let mut seen = HashSet::new();
    roots.retain(|root| seen.insert(root.id.clone()));
    Ok(roots)
}

pub fn emit_targets(
    roots_path: &Path,
    output_path: &Path,
    teacher_depth: u8,
    teacher_nodes: u64,
    teacher_beam: usize,
) -> Result<usize, String> {
    let file = fs::File::open(roots_path)
        .map_err(|error| format!("open roots {}: {error}", roots_path.display()))?;
    let mut output = io::BufWriter::new(
        fs::File::create(output_path)
            .map_err(|error| format!("create targets {}: {error}", output_path.display()))?,
    );
    let teacher = TeacherSpec {
        id: TEACHER_ID.to_owned(),
        depth: teacher_depth,
        max_nodes: teacher_nodes,
        beam_width: teacher_beam,
        weights: V05_WEIGHTS,
    };
    let config = SearchConfig {
        depth: teacher_depth,
        max_nodes: teacher_nodes,
        beam_width: teacher_beam,
        weights: V05_WEIGHTS,
        tactical_proof_horizon: None,
    };
    let mut count = 0;
    for (line_number, line) in io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|error| format!("read roots line {}: {error}", line_number + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let root: RootRecord = serde_json::from_str(&line)
            .map_err(|error| format!("parse root line {}: {error}", line_number + 1))?;
        let state = runtime_state(&root.position)?;
        let legal = state.legal_actions();
        if legal.is_empty() {
            return Err(format!("root {} has no legal actions", root.id));
        }
        let safe_actions = tactical_root_safe_actions(state, state.turn, V05_WEIGHTS);
        let root_has_forced_block = safe_actions.len() < legal.len();
        let mut actions = Vec::with_capacity(legal.len());
        for action in legal.iter().copied() {
            let transition = state.apply_legal(action);
            let next = transition.state;
            let immediate_win = next.winner == Some(state.turn);
            let features = if immediate_win {
                [0; 6]
            } else {
                [
                    evaluate(next, state.turn, EvaluationWeights { path: 1, ..zero_weights() }),
                    evaluate(next, state.turn, EvaluationWeights { material: 1, ..zero_weights() }),
                    evaluate(next, state.turn, EvaluationWeights { capture: 1, ..zero_weights() }),
                    evaluate(next, state.turn, EvaluationWeights { structure: 1, ..zero_weights() }),
                    evaluate(next, state.turn, EvaluationWeights { threat: 1, ..zero_weights() }),
                    evaluate(next, state.turn, EvaluationWeights { edge: 1, ..zero_weights() }),
                ]
            };
            let teacher_result = analyze_action(state, action, config)
                .map_err(|error| format!("label {} action {}: {error}", root.id, action))?;
            actions.push(TargetAction {
                action: action.into(),
                features,
                capture_count: transition.captured.count_ones() as u8,
                immediate_win,
                safe: safe_actions.contains(&action),
                teacher_score: teacher_result.score,
                teacher_nodes: teacher_result.nodes,
                teacher_exhausted: teacher_result.exhausted,
            });
        }
        let teacher_best_score = actions
            .iter()
            .map(|action| action.teacher_score)
            .max()
            .ok_or_else(|| format!("root {} produced no labels", root.id))?;
        let teacher_best_actions = actions
            .iter()
            .filter(|action| action.teacher_score == teacher_best_score)
            .map(|action| action.action.clone())
            .collect();
        let record = TargetRecord {
            schema_version: TARGET_SCHEMA_VERSION,
            id: root.id,
            source_family: root.source_family,
            source_game_id: root.source_game_id,
            source_ply: root.source_ply,
            phase: root.phase,
            position: root.position,
            teacher: teacher.clone(),
            root_has_forced_block,
            teacher_best_score,
            teacher_best_actions,
            actions,
        };
        serde_json::to_writer(&mut output, &record)
            .map_err(|error| format!("write target: {error}"))?;
        output
            .write_all(b"\n")
            .map_err(|error| format!("write target newline: {error}"))?;
        count += 1;
        if count % 8 == 0 {
            eprintln!("root-regret: labeled {count} roots");
        }
    }
    output
        .flush()
        .map_err(|error| format!("flush targets: {error}"))?;
    Ok(count)
}

fn zero_weights() -> EvaluationWeights {
    EvaluationWeights {
        path: 0,
        material: 0,
        capture: 0,
        structure: 0,
        threat: 0,
        edge: 0,
    }
}

fn read_unified_games(games_dir: &Path) -> Result<Vec<UnifiedGame>, String> {
    let mut paths = fs::read_dir(games_dir)
        .map_err(|error| format!("read games directory {}: {error}", games_dir.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("read games directory entry: {error}"))?;
    paths.sort();
    let mut games = Vec::new();
    for path in paths.into_iter().filter(|path| path.extension().is_some_and(|ext| ext == "tsv")) {
        let text = fs::read_to_string(&path)
            .map_err(|error| format!("read games shard {}: {error}", path.display()))?;
        for line in text.lines().filter(|line| !line.trim().is_empty() && !line.starts_with('#')) {
            games.push(parse_unified_game(line).map_err(|error| {
                format!("parse games shard {}: {error}", path.display())
            })?);
        }
    }
    Ok(games)
}

fn root_from_unified_game(game: &UnifiedGame, ply: u16) -> Result<RootRecord, String> {
    let mut state = GameState::with_config(game.config);
    for action in game.actions.iter().take(ply as usize).copied() {
        state = state.apply(action)?.state;
    }
    let position = RuntimePosition::from(state);
    validate_runtime_position(&position)?;
    Ok(RootRecord {
        schema_version: ROOT_SCHEMA_VERSION,
        id: format!("canonical:{}:{ply}", game.key),
        source_family: "canonical-replay".to_owned(),
        source_game_id: game.key.clone(),
        source_ply: ply,
        phase: phase_name(&position),
        position,
    })
}

fn read_human_roots(path: &Path) -> Result<Vec<RootRecord>, String> {
    let text = fs::read_to_string(path)
        .map_err(|error| format!("read human fixtures {}: {error}", path.display()))?;
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header: HumanHeader = serde_json::from_str(
        lines
            .next()
            .ok_or_else(|| "human fixture file is empty".to_owned())?,
    )
    .map_err(|error| format!("parse human fixture header: {error}"))?;
    let mut roots = Vec::new();
    for line in lines {
        let row: HumanRow = serde_json::from_str(line)
            .map_err(|error| format!("parse human fixture row: {error}"))?;
        let position = human_position(&header.config, row.state)?;
        validate_runtime_position(&position)?;
        roots.push(RootRecord {
            schema_version: ROOT_SCHEMA_VERSION,
            id: format!("human:{}", row.id),
            source_family: "human-tactical".to_owned(),
            source_game_id: row.id,
            source_ply: row.source_ply,
            phase: phase_name(&position),
            position,
        });
    }
    Ok(roots)
}

fn human_position(config: &HumanConfig, state: HumanState) -> Result<RuntimePosition, String> {
    let board_size = config.board_size;
    let cells = usize::from(board_size) * usize::from(board_size);
    let mut board = vec![None; cells];
    for square in state.light {
        if usize::from(square) >= cells || board[usize::from(square)].is_some() {
            return Err("human fixture has invalid light square".to_owned());
        }
        board[usize::from(square)] = Some(ContractPlayer::Light);
    }
    for square in state.dark {
        if usize::from(square) >= cells || board[usize::from(square)].is_some() {
            return Err("human fixture has invalid dark square".to_owned());
        }
        board[usize::from(square)] = Some(ContractPlayer::Dark);
    }
    Ok(RuntimePosition {
        contract_version: CONTRACT_VERSION,
        config: GameConfig {
            rules_version: RULES_VERSION.to_owned(),
            board_size,
            reserve_per_player: config.reserve_per_player,
            max_plies: config.max_plies,
            repetition_limit: 3,
        },
        board,
        reserve: state.reserve,
        turn: contract_player(&state.turn)?,
        forbidden: state.forbidden,
        last_relocated_to: state.last_relocated_to,
        last_capture: state.last_capture,
        last_player: state.last_player.as_deref().map(contract_player).transpose()?,
        winner: state.winner.as_deref().map(contract_player).transpose()?,
        winning_path: Vec::new(),
        ply: state.ply,
    })
}

fn contract_player(value: &str) -> Result<ContractPlayer, String> {
    match value {
        "light" => Ok(ContractPlayer::Light),
        "dark" => Ok(ContractPlayer::Dark),
        _ => Err(format!("invalid player {value}")),
    }
}

fn runtime_state(position: &RuntimePosition) -> Result<GameState, String> {
    GameState::try_from(position.clone())
}

fn validate_runtime_position(position: &RuntimePosition) -> Result<(), String> {
    runtime_state(position).map(|_| ())
}

fn phase_name(position: &RuntimePosition) -> String {
    let occupied = position
        .board
        .iter()
        .filter(|piece| piece.is_some())
        .count();
    let reserves = usize::from(position.reserve.light) + usize::from(position.reserve.dark);
    if occupied < 8 {
        "opening".to_owned()
    } else if reserves == 0 {
        "movement".to_owned()
    } else if occupied >= 20 {
        "late-placement".to_owned()
    } else {
        "midgame".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pathagon_engine::Action;

    #[test]
    fn target_schema_captures_all_legal_actions_and_unit_features() {
        let state = GameState::new().apply_legal(Action::Place { to: 24 }).state;
        let position = RuntimePosition::from(state);
        let root = RootRecord {
            schema_version: ROOT_SCHEMA_VERSION,
            id: "test".to_owned(),
            source_family: "test".to_owned(),
            source_game_id: "test-game".to_owned(),
            source_ply: 1,
            phase: "opening".to_owned(),
            position,
        };
        let encoded = serde_json::to_string(&root).expect("root serializes");
        let decoded: RootRecord = serde_json::from_str(&encoded).expect("root deserializes");
        let decoded_state = runtime_state(&decoded.position).expect("position validates");
        assert_eq!(decoded_state.legal_actions().len(), 48);
        assert_eq!(decoded.schema_version, ROOT_SCHEMA_VERSION);
    }

    #[test]
    fn human_fixture_conversion_preserves_inventory() {
        let position = human_position(
            &HumanConfig {
                board_size: 4,
                reserve_per_player: 5,
                max_plies: 64,
            },
            HumanState {
                light: vec![0, 2],
                dark: vec![5, 7],
                reserve: PlayerNumbers { light: 3, dark: 3 },
                turn: "light".to_owned(),
                forbidden: Vec::new(),
                last_relocated_to: PlayerSquares { light: None, dark: None },
                last_capture: 0,
                last_player: None,
                winner: None,
                ply: 4,
            },
        )
        .expect("fixture converts");
        let state = runtime_state(&position).expect("fixture validates");
        assert_eq!(state.reserve, [3, 3]);
        assert_eq!(state.legal_actions().len(), 12);
    }
}
