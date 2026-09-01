//! Build source-disjoint Pathfinder transition-policy roots from the canonical corpus.
//!
//! The teacher is the Rust Pathfinder search itself. This research binary owns
//! replay, legality, action features, and labels; the Python trainer only sees
//! the resulting JSONL rows.

use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use pathagon_engine::corpus::{encode_state, parse_unified_game};
use pathagon_engine::search::{
    search_best_action, search_best_action_with_tactical_filter, EvaluationWeights, SearchConfig,
};
use pathagon_engine::transition_policy::action_features;
use pathagon_engine::{Action, GameState};
use serde_json::{json, Value};

const DEFAULT_ROOT_PLIES: &[u16] = &[
    4, 5, 8, 9, 16, 17, 24, 25, 32, 33, 48, 49, 64, 65, 80, 81, 96, 97,
];
const FEATURE_COUNT: usize = 6;

#[derive(Clone)]
struct Candidate {
    source_game_id: String,
    source_ply: u16,
    partition: &'static str,
    state: GameState,
}

fn main() {
    let args = parse_args();
    let input = args
        .get("input")
        .map(PathBuf::from)
        .unwrap_or_else(|| fail("--input <games.tsv|games-dir> is required"));
    let output = args
        .get("out")
        .map(PathBuf::from)
        .unwrap_or_else(|| fail("--out <targets.jsonl> is required"));
    let depth = number(&args, "depth", 5_u8);
    let max_nodes = number(&args, "nodes", 500_000_u64);
    let beam_width = number(&args, "beam", 256_usize);
    let max_roots = number(&args, "max-roots", 1_920_usize);
    let skip_roots = number(&args, "skip-roots", 0_usize);
    let selection_roots = number(&args, "selection-roots", max_roots);
    let heldout_fraction = number(&args, "heldout-percent", 20_u8).min(99);
    let roots_per_game = number(&args, "roots-per-game", 18_usize).max(1);
    let tactical_filter = args.contains_key("tactical-filter");
    let root_plies = parse_root_plies(args.get("root-plies"));
    if depth == 0 || max_nodes == 0 || beam_width == 0 || max_roots == 0 {
        fail("--depth, --nodes, --beam, and --max-roots must be positive");
    }

    let paths = input_paths(&input);
    let candidates = collect_candidates(&paths, &root_plies, roots_per_game, heldout_fraction);
    let selected = select_candidates(candidates, selection_roots, skip_roots, max_roots);
    if selected.is_empty() {
        fail("no eligible roots were found in the canonical corpus");
    }

    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| fail(&format!("cannot create {}: {error}", parent.display())));
    }
    let file = File::create(&output)
        .unwrap_or_else(|error| fail(&format!("cannot create {}: {error}", output.display())));
    let mut writer = BufWriter::new(file);
    let config = SearchConfig {
        depth,
        max_nodes,
        beam_width,
        ..SearchConfig::default()
    };
    let mut counts = [0_usize; 2];
    let mut total_nodes = 0_u64;
    let mut exhausted = 0_usize;
    let mut emitted = 0_usize;
    for candidate in selected {
        let legal = candidate.state.legal_actions();
        if legal.is_empty() {
            continue;
        }
        let safe = safe_actions(candidate.state, &legal);
        let result = if tactical_filter {
            search_best_action_with_tactical_filter(candidate.state, config)
        } else {
            search_best_action(candidate.state, config)
        };
        let Some(teacher_action) = result.action.filter(|action| legal.contains(action)) else {
            continue;
        };
        let actions = legal
            .iter()
            .copied()
            .map(|action| action_row(candidate.state, action, safe.contains(&action)))
            .collect::<Vec<_>>();
        let phase = phase(candidate.state);
        let row = json!({
            "schemaVersion": 1,
            "id": format!("{}:{}", candidate.source_game_id, candidate.source_ply),
            "sourceGameId": candidate.source_game_id,
            "sourcePly": candidate.source_ply,
            "phase": phase,
            "partition": candidate.partition,
            "state": encode_state(candidate.state),
            "teacher": {
                "id": "pathfinder-teacher-d5-b256-500k",
                "depth": depth,
                "maxNodes": max_nodes,
                "beamWidth": beam_width,
                "tacticalFilter": tactical_filter,
                "weights": weights_json(config.weights),
            },
            "teacherAction": action_json(teacher_action),
            "teacherScore": result.score,
            "teacherNodes": result.nodes,
            "completedDepth": result.completed_depth,
            "exhausted": result.exhausted,
            "actions": actions,
        });
        serde_json::to_writer(&mut writer, &row)
            .unwrap_or_else(|error| fail(&format!("cannot serialize target row: {error}")));
        writer
            .write_all(b"\n")
            .unwrap_or_else(|error| fail(&format!("cannot write target row: {error}")));
        counts[if candidate.partition == "heldout" {
            1
        } else {
            0
        }] += 1;
        total_nodes = total_nodes.saturating_add(result.nodes);
        exhausted += usize::from(result.exhausted);
        emitted += 1;
        if emitted % 25 == 0 {
            eprintln!("labeled {emitted}/{} roots", max_roots);
        }
    }
    writer
        .flush()
        .unwrap_or_else(|error| fail(&format!("cannot flush targets: {error}")));
    println!(
        "{}",
        json!({
            "schemaVersion": 1,
            "out": output,
            "inputFiles": paths,
            "roots": emitted,
            "trainRoots": counts[0],
            "heldoutRoots": counts[1],
            "teacher": {
                "depth": depth,
                "maxNodes": max_nodes,
                "beamWidth": beam_width,
                "tacticalFilter": tactical_filter,
                "weights": weights_json(config.weights),
            },
            "teacherNodes": total_nodes,
            "exhaustedRoots": exhausted,
            "rootPlies": root_plies,
        })
    );
}

fn collect_candidates(
    paths: &[PathBuf],
    root_plies: &[u16],
    roots_per_game: usize,
    heldout_fraction: u8,
) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    for path in paths {
        let source = fs::read_to_string(path)
            .unwrap_or_else(|error| fail(&format!("cannot read {}: {error}", path.display())));
        for (line_number, line) in source.lines().enumerate() {
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            let game = parse_unified_game(line).unwrap_or_else(|error| {
                fail(&format!(
                    "invalid corpus row {}:{}: {error}",
                    path.display(),
                    line_number + 1
                ))
            });
            if game.config.board_size != 7 {
                continue;
            }
            let partition = if stable_hash(&game.key) % 100 < u64::from(heldout_fraction) {
                "heldout"
            } else {
                "train"
            };
            let mut state = GameState::with_config(game.config);
            let mut added = 0_usize;
            for (ply, action) in game.actions.iter().copied().enumerate() {
                let ply = ply as u16;
                if root_plies.contains(&ply) && added < roots_per_game && state.winner.is_none() {
                    candidates.push(Candidate {
                        source_game_id: game.key.clone(),
                        source_ply: ply,
                        partition,
                        state,
                    });
                    added += 1;
                }
                state = state
                    .apply(action)
                    .unwrap_or_else(|error| {
                        fail(&format!("illegal action in {}: {error}", game.key))
                    })
                    .state;
            }
        }
    }
    candidates
}

fn select_candidates(
    mut candidates: Vec<Candidate>,
    selection_roots: usize,
    skip_roots: usize,
    max_roots: usize,
) -> Vec<Candidate> {
    candidates
        .sort_by_key(|candidate| (stable_hash(&candidate.source_game_id), candidate.source_ply));
    let selection_limit = selection_roots.max(skip_roots.saturating_add(max_roots));
    let mut train = candidates
        .iter()
        .filter(|candidate| candidate.partition == "train")
        .take(selection_limit.saturating_mul(4) / 5)
        .cloned()
        .collect::<Vec<_>>();
    let mut heldout = candidates
        .iter()
        .filter(|candidate| candidate.partition == "heldout")
        .take(selection_limit.saturating_sub(train.len()))
        .cloned()
        .collect::<Vec<_>>();
    if heldout.len() < selection_limit / 5 {
        let needed = selection_limit / 5 - heldout.len();
        let extra = candidates
            .iter()
            .filter(|candidate| candidate.partition == "train")
            .skip(train.len())
            .take(needed)
            .cloned()
            .collect::<Vec<_>>();
        train.truncate(selection_limit.saturating_sub(selection_limit / 5));
        heldout.extend(extra);
    }
    train.extend(heldout);
    train.sort_by_key(|candidate| (stable_hash(&candidate.source_game_id), candidate.source_ply));
    train.into_iter().skip(skip_roots).take(max_roots).collect()
}

fn action_row(state: GameState, action: Action, safe: bool) -> Value {
    let features = action_features(state, action, safe, false);
    let primary = features[..FEATURE_COUNT]
        .iter()
        .map(|value| *value as i32)
        .collect::<Vec<_>>();
    let next = state.apply_legal(action).state;
    json!({
        "action": action_json(action),
        "features": primary,
        "captureCount": next.last_capture,
        "immediateWin": next.winner == Some(state.turn),
        "safe": safe,
    })
}

fn safe_actions(state: GameState, legal: &[Action]) -> Vec<Action> {
    let opponent = state.turn.other();
    legal
        .iter()
        .copied()
        .filter(|action| {
            let next = state.apply_legal(*action).state;
            if next.winner == Some(state.turn) {
                return true;
            }
            let opponent_view = if next.turn == opponent {
                next
            } else {
                GameState {
                    turn: opponent,
                    ..next
                }
            };
            !opponent_view
                .legal_actions()
                .iter()
                .any(|reply| opponent_view.apply_legal(*reply).state.winner == Some(opponent))
        })
        .collect()
}

fn phase(state: GameState) -> &'static str {
    let occupied = (state.light | state.dark).count_ones();
    let reserves = u32::from(state.reserve[0]) + u32::from(state.reserve[1]);
    if occupied < 8 {
        "opening"
    } else if state.reserve[state.turn.index()] > 0 {
        "placement"
    } else if reserves == 0 {
        "movement"
    } else {
        "late"
    }
}

fn action_json(action: Action) -> Value {
    match action {
        Action::Place { to } => json!({"kind": "place", "to": to}),
        Action::Relocate { from, to } => json!({"kind": "relocate", "from": from, "to": to}),
    }
}

fn weights_json(weights: EvaluationWeights) -> Value {
    json!({
        "path": weights.path,
        "material": weights.material,
        "capture": weights.capture,
        "structure": weights.structure,
        "threat": weights.threat,
        "edge": weights.edge,
    })
}

fn input_paths(input: &Path) -> Vec<PathBuf> {
    if input.is_file() {
        return vec![input.to_path_buf()];
    }
    let mut paths = fs::read_dir(input)
        .unwrap_or_else(|error| fail(&format!("cannot read {}: {error}", input.display())))
        .filter_map(|entry| entry.ok().map(|item| item.path()))
        .filter(|path| path.extension().is_some_and(|extension| extension == "tsv"))
        .collect::<Vec<_>>();
    paths.sort();
    if paths.is_empty() {
        fail(&format!("{} contains no .tsv files", input.display()));
    }
    paths
}

fn parse_root_plies(value: Option<&String>) -> Vec<u16> {
    let Some(value) = value else {
        return DEFAULT_ROOT_PLIES.to_vec();
    };
    let mut result = value
        .split(',')
        .filter_map(|item| item.parse::<u16>().ok())
        .collect::<Vec<_>>();
    result.sort_unstable();
    result.dedup();
    if result.is_empty() {
        fail("--root-plies must contain comma-separated nonnegative integers");
    }
    result
}

fn parse_args() -> std::collections::HashMap<String, String> {
    let mut args = std::collections::HashMap::new();
    let mut values = env::args().skip(1);
    while let Some(argument) = values.next() {
        let Some(name) = argument.strip_prefix("--") else {
            fail(&format!("unexpected argument {argument}"));
        };
        if let Some(value) = values.next().filter(|value| !value.starts_with("--")) {
            args.insert(name.to_owned(), value);
        } else {
            args.insert(name.to_owned(), String::new());
        }
    }
    args
}

fn number<T>(args: &std::collections::HashMap<String, String>, name: &str, default: T) -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    args.get(name)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|error| fail(&format!("invalid --{name}: {error}")))
        })
        .unwrap_or(default)
}

fn stable_hash(value: &str) -> u64 {
    value.bytes().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3_u64)
    })
}

fn fail(message: &str) -> ! {
    eprintln!("optimized-sort-selection: {message}");
    std::process::exit(2);
}
