use std::collections::{BTreeMap, HashMap};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use pathagon_engine::corpus::{
    decode_action, decode_state, encode_action, encode_state, parse_unified_game,
};
use pathagon_engine::search::{
    search_best_action_with_tactical_filter, EvaluationWeights, SearchConfig,
};
use pathagon_engine::{Action, GameState, Player};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default)]
struct OutcomeCounts {
    visits: u32,
    wins: u32,
    losses: u32,
    draws: u32,
}

#[derive(Clone, Debug, Deserialize)]
struct PriorRow {
    state: String,
    action: String,
    ply: u16,
    visits: u32,
    wins: u32,
    losses: u32,
    draws: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct LabelRow {
    state: String,
    ply: u16,
    corpus_visits: u32,
    corpus_action: String,
    corpus_points_per_mille: u32,
    incumbent_action: String,
    teacher_action: String,
    teacher_score: i32,
    teacher_nodes: u64,
    teacher_completed_depth: u8,
    teacher_exhausted: bool,
    teacher_agrees_with_corpus: bool,
    teacher_differs_from_incumbent: bool,
}

#[derive(Clone, Copy)]
enum Symmetry {
    Identity,
    Rotate90,
    Rotate180,
    Rotate270,
    FlipRows,
    FlipColumns,
    Transpose,
    AntiTranspose,
}

const SYMMETRIES: [Symmetry; 8] = [
    Symmetry::Identity,
    Symmetry::Rotate90,
    Symmetry::Rotate180,
    Symmetry::Rotate270,
    Symmetry::FlipRows,
    Symmetry::FlipColumns,
    Symmetry::Transpose,
    Symmetry::AntiTranspose,
];

fn main() {
    let args = args();
    match args.get("command").map(String::as_str) {
        Some("aggregate") => aggregate(&args).unwrap_or_else(|error| fail(error)),
        Some("label") => label(&args).unwrap_or_else(|error| fail(error)),
        Some("select") => select(&args).unwrap_or_else(|error| fail(error)),
        Some("select-corpus") => select_corpus(&args).unwrap_or_else(|error| fail(error)),
        _ => fail("use --command aggregate, --command label, --command select, or --command select-corpus"),
    }
}

fn select_corpus(args: &HashMap<String, String>) -> Result<(), String> {
    let priors = required_path(args, "priors")?;
    let labels_directory = required_path(args, "labels-directory")?;
    let output = required_path(args, "output")?;
    let minimum_state_visits = number(args, "minimum-state-visits", 32_u32);
    let minimum_action_visits = number(args, "minimum-action-visits", 4_u32);
    let margin = number(args, "margin-per-mille", 30_u32);
    let prior_visits = number(args, "prior-visits", 8_u32);
    let max_ply = number(args, "max-ply", 2_u16);

    let mut incumbents = HashMap::<String, String>::new();
    let mut paths = fs::read_dir(&labels_directory)
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("labels-shard-") && name.ends_with(".jsonl"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    for path in paths {
        for line in BufReader::new(fs::File::open(path).map_err(|error| error.to_string())?).lines()
        {
            let row: LabelRow = serde_json::from_str(&line.map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())?;
            incumbents.insert(row.state, row.incumbent_action);
        }
    }

    let mut by_state = HashMap::<String, Vec<PriorRow>>::new();
    for line in BufReader::new(fs::File::open(priors).map_err(|error| error.to_string())?).lines() {
        let row: PriorRow = serde_json::from_str(&line.map_err(|error| error.to_string())?)
            .map_err(|error| error.to_string())?;
        if row.ply >= 2 && row.ply <= max_ply {
            by_state.entry(row.state.clone()).or_default().push(row);
        }
    }

    let mut entries = BTreeMap::<String, (String, u8, u64)>::new();
    let mut selected = 0_u32;
    for (state_key, rows) in by_state {
        let total = rows.iter().map(|row| row.visits).sum::<u32>();
        if total < minimum_state_visits {
            continue;
        }
        let Some(incumbent_action) = incumbents.get(&state_key) else {
            continue;
        };
        let incumbent_rate = rows
            .iter()
            .find(|row| &row.action == incumbent_action)
            .map_or(500, |row| posterior_per_mille(row, prior_visits));
        let Some(candidate) = rows
            .iter()
            .filter(|row| row.visits >= minimum_action_visits && &row.action != incumbent_action)
            .max_by_key(|row| (posterior_per_mille(row, prior_visits), row.visits))
        else {
            continue;
        };
        let candidate_rate = posterior_per_mille(candidate, prior_visits);
        if candidate_rate < incumbent_rate.saturating_add(margin) {
            continue;
        }
        selected += 1;
        let state = decode_state(&state_key)?;
        let action = decode_action(&candidate.action)?;
        for symmetry in SYMMETRIES {
            entries.insert(
                encode_state(transform_state(state, symmetry)),
                (
                    encode_action(transform_action(action, state.config.board_size, symmetry)),
                    0,
                    u64::from(candidate.visits),
                ),
            );
        }
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut writer = BufWriter::new(fs::File::create(&output).map_err(|error| error.to_string())?);
    writer
        .write_all(b"# state\taction\tteacher\tcompleted-depth\tnodes\n")
        .map_err(|error| error.to_string())?;
    for (state, (action, depth, visits)) in &entries {
        writeln!(
            writer,
            "{state}\t{action}\tcorpus-v1-shrunk\t{depth}\t{visits}"
        )
        .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    println!(
        "{}",
        serde_json::json!({
            "selectedCanonicalRoots": selected,
            "expandedEntries": entries.len(),
            "minimumStateVisits": minimum_state_visits,
            "minimumActionVisits": minimum_action_visits,
            "marginPerMille": margin,
            "priorVisits": prior_visits,
            "maxPly": max_ply,
            "output": output,
        })
    );
    Ok(())
}

fn posterior_per_mille(row: &PriorRow, prior_visits: u32) -> u32 {
    let numerator = u64::from(points(row)) * 1_000 + u64::from(prior_visits) * 1_000;
    let denominator = 2 * u64::from(row.visits.saturating_add(prior_visits));
    (numerator / denominator) as u32
}

fn select(args: &HashMap<String, String>) -> Result<(), String> {
    let labels_directory = required_path(args, "labels-directory")?;
    let output = required_path(args, "output")?;
    let minimum_depth = number(args, "minimum-depth", 6_u8);
    let mut paths = fs::read_dir(&labels_directory)
        .map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("labels-shard-") && name.ends_with(".jsonl"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    let mut entries = BTreeMap::<String, (String, u8, u64)>::new();
    let mut selected = 0_u32;
    for path in paths {
        let source = fs::File::open(&path).map_err(|error| error.to_string())?;
        for line in BufReader::new(source).lines() {
            let row: LabelRow = serde_json::from_str(&line.map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())?;
            if row.teacher_completed_depth < minimum_depth || !row.teacher_differs_from_incumbent {
                continue;
            }
            selected += 1;
            let state = decode_state(&row.state)?;
            let action = decode_action(&row.teacher_action)?;
            for symmetry in SYMMETRIES {
                let transformed_state = transform_state(state, symmetry);
                let transformed_action =
                    transform_action(action, state.config.board_size, symmetry);
                entries.insert(
                    encode_state(transformed_state),
                    (
                        encode_action(transformed_action),
                        row.teacher_completed_depth,
                        row.teacher_nodes,
                    ),
                );
            }
        }
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut writer = BufWriter::new(fs::File::create(&output).map_err(|error| error.to_string())?);
    writer
        .write_all(b"# state\taction\tteacher\tcompleted-depth\tnodes\n")
        .map_err(|error| error.to_string())?;
    for (state, (action, depth, nodes)) in &entries {
        writeln!(
            writer,
            "{state}\t{action}\tpathfinder-v0.5-d6-100k-b16\t{depth}\t{nodes}"
        )
        .map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    println!(
        "{}",
        serde_json::json!({
            "selectedCanonicalRoots": selected,
            "expandedEntries": entries.len(),
            "minimumDepth": minimum_depth,
            "output": output,
        })
    );
    Ok(())
}

fn aggregate(args: &HashMap<String, String>) -> Result<(), String> {
    let corpus = required_path(args, "corpus")?;
    let output = required_path(args, "output")?;
    let min_ply = number(args, "min-ply", 2_u16);
    let max_ply = number(args, "max-ply", 8_u16);
    let mut counts = HashMap::<(GameState, Action), OutcomeCounts>::new();
    let mut games = 0_u32;
    let mut positions = 0_u64;

    let mut shards = fs::read_dir(&corpus)
        .map_err(|error| format!("cannot read {}: {error}", corpus.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "tsv"))
        .collect::<Vec<_>>();
    shards.sort();

    for shard in shards {
        let source = fs::File::open(&shard)
            .map_err(|error| format!("cannot open {}: {error}", shard.display()))?;
        for line in BufReader::new(source).lines() {
            let line = line.map_err(|error| format!("cannot read {}: {error}", shard.display()))?;
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let game = parse_unified_game(&line)
                .map_err(|error| format!("{}: {error}", shard.display()))?;
            if game.config.board_size != 7 || game.config.reserve_per_player != 14 {
                continue;
            }
            let final_state = game
                .replay()
                .map_err(|error| format!("{}: {error}", game.key))?;
            let winner = final_state.winner;
            let mut state = GameState::with_config(game.config);
            for action in game.actions {
                if state.ply >= min_ply && state.ply <= max_ply {
                    let (canonical_state, symmetry) = canonicalize(state);
                    let canonical_action =
                        transform_action(action, state.config.board_size, symmetry);
                    let entry = counts
                        .entry((canonical_state, canonical_action))
                        .or_default();
                    entry.visits = entry.visits.saturating_add(1);
                    match winner {
                        Some(player) if player == state.turn => {
                            entry.wins = entry.wins.saturating_add(1)
                        }
                        Some(_) => entry.losses = entry.losses.saturating_add(1),
                        None => entry.draws = entry.draws.saturating_add(1),
                    }
                    positions += 1;
                }
                state = state.apply(action).map_err(str::to_owned)?.state;
                if state.ply > max_ply {
                    break;
                }
            }
            games += 1;
        }
    }

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut rows = counts.into_iter().collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        left.0
             .0
            .ply
            .cmp(&right.0 .0.ply)
            .then_with(|| encode_state(left.0 .0).cmp(&encode_state(right.0 .0)))
            .then_with(|| left.0 .1.order().cmp(&right.0 .1.order()))
    });
    let mut writer = BufWriter::new(fs::File::create(&output).map_err(|error| error.to_string())?);
    for ((state, action), count) in &rows {
        serde_json::to_writer(
            &mut writer,
            &serde_json::json!({
                "state": encode_state(*state),
                "action": encode_action(*action),
                "ply": state.ply,
                "visits": count.visits,
                "wins": count.wins,
                "losses": count.losses,
                "draws": count.draws,
            }),
        )
        .map_err(|error| error.to_string())?;
        writer.write_all(b"\n").map_err(|error| error.to_string())?;
    }
    writer.flush().map_err(|error| error.to_string())?;
    println!(
        "{}",
        serde_json::json!({
            "games": games,
            "observedPositions": positions,
            "stateActions": rows.len(),
            "minPly": min_ply,
            "maxPly": max_ply,
            "output": output,
        })
    );
    Ok(())
}

fn canonicalize(state: GameState) -> (GameState, Symmetry) {
    SYMMETRIES
        .into_iter()
        .map(|symmetry| (transform_state(state, symmetry), symmetry))
        .min_by_key(|(candidate, _)| encode_state(*candidate))
        .expect("the D4 symmetry set is non-empty")
}

fn transform_state(state: GameState, symmetry: Symmetry) -> GameState {
    let size = state.config.board_size;
    let swaps = symmetry_swaps_players(symmetry);
    let light = transform_mask(state.light, size, symmetry);
    let dark = transform_mask(state.dark, size, symmetry);
    let relocated = [
        state.last_relocated_to[0].map(|square| transform_square(square, size, symmetry)),
        state.last_relocated_to[1].map(|square| transform_square(square, size, symmetry)),
    ];
    GameState {
        config: state.config,
        light: if swaps { dark } else { light },
        dark: if swaps { light } else { dark },
        reserve: if swaps {
            [state.reserve[1], state.reserve[0]]
        } else {
            state.reserve
        },
        turn: if swaps { other(state.turn) } else { state.turn },
        forbidden: transform_mask(state.forbidden, size, symmetry),
        last_relocated_to: if swaps {
            [relocated[1], relocated[0]]
        } else {
            relocated
        },
        last_capture: state.last_capture,
        last_player: state
            .last_player
            .map(|player| if swaps { other(player) } else { player }),
        winner: state
            .winner
            .map(|player| if swaps { other(player) } else { player }),
        ply: state.ply,
    }
}

fn transform_action(action: Action, size: u8, symmetry: Symmetry) -> Action {
    match action {
        Action::Place { to } => Action::Place {
            to: transform_square(to, size, symmetry),
        },
        Action::Relocate { from, to } => Action::Relocate {
            from: transform_square(from, size, symmetry),
            to: transform_square(to, size, symmetry),
        },
    }
}

fn transform_mask(mask: u64, size: u8, symmetry: Symmetry) -> u64 {
    (0..size.saturating_mul(size)).fold(0_u64, |transformed, square| {
        if mask & (1_u64 << square) == 0 {
            transformed
        } else {
            transformed | (1_u64 << transform_square(square, size, symmetry))
        }
    })
}

fn transform_square(square: u8, size: u8, symmetry: Symmetry) -> u8 {
    let row = square / size;
    let column = square % size;
    let last = size - 1;
    let (new_row, new_column) = match symmetry {
        Symmetry::Identity => (row, column),
        Symmetry::Rotate90 => (column, last - row),
        Symmetry::Rotate180 => (last - row, last - column),
        Symmetry::Rotate270 => (last - column, row),
        Symmetry::FlipRows => (last - row, column),
        Symmetry::FlipColumns => (row, last - column),
        Symmetry::Transpose => (column, row),
        Symmetry::AntiTranspose => (last - column, last - row),
    };
    new_row * size + new_column
}

fn symmetry_swaps_players(symmetry: Symmetry) -> bool {
    matches!(
        symmetry,
        Symmetry::Rotate90 | Symmetry::Rotate270 | Symmetry::Transpose | Symmetry::AntiTranspose
    )
}

fn other(player: Player) -> Player {
    match player {
        Player::Light => Player::Dark,
        Player::Dark => Player::Light,
    }
}

fn label(args: &HashMap<String, String>) -> Result<(), String> {
    let priors = required_path(args, "priors")?;
    let output = required_path(args, "output")?;
    let minimum_visits = number(args, "minimum-visits", 4_u32);
    let offset = number(args, "offset", 0_usize);
    let limit = number(args, "limit", usize::MAX);
    let target_ply = number(args, "ply", 2_u16);
    let teacher = SearchConfig {
        depth: number(args, "teacher-depth", 6_u8),
        max_nodes: number(args, "teacher-nodes", 100_000_u64),
        beam_width: number(args, "teacher-beam", 16_usize),
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
    let incumbent = SearchConfig {
        depth: 4,
        max_nodes: 2_000,
        beam_width: 8,
        weights: teacher.weights,
        tactical_proof_horizon: None,
    };
    let source = fs::File::open(&priors).map_err(|error| error.to_string())?;
    let mut by_state = HashMap::<String, Vec<PriorRow>>::new();
    for line in BufReader::new(source).lines() {
        let line = line.map_err(|error| error.to_string())?;
        let row: PriorRow = serde_json::from_str(&line).map_err(|error| error.to_string())?;
        if row.ply == target_ply {
            by_state.entry(row.state.clone()).or_default().push(row);
        }
    }
    let mut states = by_state
        .into_iter()
        .filter_map(|(state, rows)| {
            let total = rows.iter().map(|row| row.visits).sum::<u32>();
            (total >= minimum_visits).then_some((state, total, rows))
        })
        .collect::<Vec<_>>();
    states.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    let offset = offset.min(states.len());
    states.drain(..offset);
    states.truncate(limit);

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let mut writer = BufWriter::new(fs::File::create(&output).map_err(|error| error.to_string())?);
    let mut agreement = 0_u32;
    let mut changed = 0_u32;
    let mut exhausted = 0_u32;
    for (index, (state_key, total, rows)) in states.iter().enumerate() {
        let state = decode_state(state_key)?;
        let corpus = rows
            .iter()
            .max_by(|left, right| {
                points(left)
                    .saturating_mul(right.visits)
                    .cmp(&points(right).saturating_mul(left.visits))
                    .then_with(|| left.visits.cmp(&right.visits))
            })
            .expect("state has prior rows");
        let corpus_action = decode_action(&corpus.action)?;
        let incumbent_result = search_best_action_with_tactical_filter(state, incumbent);
        let teacher_result = search_best_action_with_tactical_filter(state, teacher);
        let incumbent_action = incumbent_result
            .action
            .ok_or("incumbent returned no action")?;
        let teacher_action = teacher_result.action.ok_or("teacher returned no action")?;
        agreement += u32::from(teacher_action == corpus_action);
        changed += u32::from(teacher_action != incumbent_action);
        exhausted += u32::from(teacher_result.exhausted);
        let row = LabelRow {
            state: state_key.clone(),
            ply: state.ply,
            corpus_visits: *total,
            corpus_action: encode_action(corpus_action),
            corpus_points_per_mille: points_per_mille(corpus),
            incumbent_action: encode_action(incumbent_action),
            teacher_action: encode_action(teacher_action),
            teacher_score: teacher_result.score,
            teacher_nodes: teacher_result.nodes,
            teacher_completed_depth: teacher_result.completed_depth,
            teacher_exhausted: teacher_result.exhausted,
            teacher_agrees_with_corpus: teacher_action == corpus_action,
            teacher_differs_from_incumbent: teacher_action != incumbent_action,
        };
        serde_json::to_writer(&mut writer, &row).map_err(|error| error.to_string())?;
        writer.write_all(b"\n").map_err(|error| error.to_string())?;
        if (index + 1) % 100 == 0 {
            eprintln!("labeled {}/{} roots", index + 1, states.len());
        }
    }
    writer.flush().map_err(|error| error.to_string())?;
    println!(
        "{}",
        serde_json::json!({
            "roots": states.len(),
            "offset": offset,
            "teacherCorpusAgreement": agreement,
            "teacherChanges": changed,
            "teacherExhausted": exhausted,
            "teacher": {"depth": teacher.depth, "nodes": teacher.max_nodes, "beam": teacher.beam_width},
            "output": output,
        })
    );
    Ok(())
}

fn points(row: &PriorRow) -> u32 {
    row.wins.saturating_mul(2).saturating_add(row.draws)
}

fn points_per_mille(row: &PriorRow) -> u32 {
    if row.visits == 0 {
        0
    } else {
        points(row).saturating_mul(1_000) / row.visits.saturating_mul(2)
    }
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
            }
        }
        index += 1;
    }
    parsed
}

fn required_path(args: &HashMap<String, String>, key: &str) -> Result<PathBuf, String> {
    args.get(key)
        .map(PathBuf::from)
        .ok_or_else(|| format!("missing --{key}"))
}

fn number<T: std::str::FromStr + Copy>(args: &HashMap<String, String>, key: &str, default: T) -> T {
    args.get(key)
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn fail(error: impl ToString) -> ! {
    eprintln!("repertoire-research: {}", error.to_string());
    std::process::exit(2)
}
