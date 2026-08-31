//! Extract verified endgame frontier positions from replayable corpus games.
//!
//! Ring 1 is intentionally conservative: a candidate is admitted only when
//! replaying a corpus action produces a path terminal for the player to move
//! in the candidate parent. The output is a research-side JSONL interchange
//! file; promotion canonicalizes and writes the durable golden shard.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use pathagon_engine::corpus::{encode_action, parse_unified_game};
use pathagon_engine::golden::canonical_position_key;
use pathagon_engine::{has_winning_path, Action, BoardConfig, GameState, Player};
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct PositionKey {
    board_size: u8,
    reserve_per_player: u8,
    light: u64,
    dark: u64,
    reserve: [u8; 2],
    turn: Player,
    forbidden: u64,
    last_relocated_to: [Option<u8>; 2],
}

impl From<GameState> for PositionKey {
    fn from(state: GameState) -> Self {
        Self {
            board_size: state.config.board_size,
            reserve_per_player: state.config.reserve_per_player,
            light: state.light,
            dark: state.dark,
            reserve: state.reserve,
            turn: state.turn,
            forbidden: state.forbidden,
            last_relocated_to: state.last_relocated_to,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct ActionKey {
    kind: u8,
    from: u8,
    to: u8,
}

impl From<Action> for ActionKey {
    fn from(action: Action) -> Self {
        match action {
            Action::Place { to } => Self {
                kind: 0,
                from: u8::MAX,
                to,
            },
            Action::Relocate { from, to } => Self { kind: 1, from, to },
        }
    }
}

impl ActionKey {
    fn action(self) -> Action {
        if self.kind == 0 {
            Action::Place { to: self.to }
        } else {
            Action::Relocate {
                from: self.from,
                to: self.to,
            }
        }
    }
}

#[derive(Default)]
struct Candidate {
    state: Option<GameState>,
    witnesses: BTreeMap<ActionKey, BTreeSet<String>>,
    proof_kind: String,
}

#[derive(Default)]
struct Ring2Candidate {
    state: Option<GameState>,
    children: BTreeSet<String>,
    actions: BTreeMap<String, String>,
    witnesses: BTreeSet<(String, String)>,
    seed: Option<pathagon_engine::tablebase::RetrogradeValue>,
}

fn main() {
    let args = parse_args();
    let corpus = required_path(&args, "corpus");
    let output = required_path(&args, "out");
    let max_games = number(&args, "max-games");
    let max_candidates = number(&args, "max-candidates");
    let mode = args.get("mode").map(String::as_str).unwrap_or("both");
    let ring = args
        .get("ring")
        .map(|value| {
            value
                .parse::<u8>()
                .unwrap_or_else(|_| fail("--ring must be 1 or 2"))
        })
        .unwrap_or(1);
    if ring == 2 {
        extract_ring2(&corpus, &output, max_games, max_candidates);
        return;
    }
    if ring != 1 {
        fail("--ring must be 1 or 2");
    }
    if !matches!(mode, "replay" | "constructive" | "both") {
        fail("--mode must be replay, constructive, or both");
    }

    let files = corpus_files(&corpus);
    let mut candidates = BTreeMap::<PositionKey, Candidate>::new();
    let mut games_seen = 0_usize;
    let mut terminal_games = 0_usize;
    let mut skipped_non_terminal = 0_usize;
    let mut skipped_non_default = 0_usize;
    let mut duplicate_witnesses = 0_usize;
    let mut constructive_transitions = 0_usize;
    let mut replay_witnesses = 0_usize;
    let mut replay_placement_witnesses = 0_usize;
    let mut replay_relocation_witnesses = 0_usize;
    let mut replay_capture_witnesses = 0_usize;
    let mut piece_density = BTreeMap::<u8, usize>::new();

    'files: for path in files {
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| fail(&format!("cannot read {}: {error}", path.display())));
        for (line_number, line) in source.lines().enumerate() {
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            if max_games > 0 && games_seen >= max_games {
                break 'files;
            }
            let game = parse_unified_game(line).unwrap_or_else(|error| {
                fail(&format!("{}:{}: {error}", path.display(), line_number + 1))
            });
            games_seen += 1;
            if game.config != BoardConfig::DEFAULT {
                skipped_non_default += 1;
                continue;
            }

            let (states, terminal) = replay(&game.actions, game.config).unwrap_or_else(|error| {
                fail(&format!("{}:{}: {error}", path.display(), line_number + 1))
            });
            let Some(terminal) = terminal else {
                skipped_non_terminal += 1;
                continue;
            };
            terminal_games += 1;
            if mode == "replay" || mode == "both" {
                let Some(action) = game.actions.last().copied() else {
                    continue;
                };
                let parent = states[states.len() - 2];
                if parent.turn != terminal.winner.expect("terminal winner")
                    || !parent.legal_actions().contains(&action)
                    || parent.apply_legal(action).state != terminal
                {
                    fail(&format!(
                        "{}:{}: terminal action is not a verified parent transition",
                        path.display(),
                        line_number + 1
                    ));
                }
                if !insert_candidate(
                    &mut candidates,
                    max_candidates,
                    parent,
                    action,
                    game.key.clone(),
                    "forward-replayed-terminal",
                    &mut duplicate_witnesses,
                ) {
                    break 'files;
                }
                replay_witnesses += 1;
                match action {
                    Action::Place { .. } => replay_placement_witnesses += 1,
                    Action::Relocate { .. } => replay_relocation_witnesses += 1,
                }
                if terminal.forbidden.count_ones() > 0 {
                    replay_capture_witnesses += 1;
                }
                *piece_density
                    .entry((parent.light | parent.dark).count_ones() as u8)
                    .or_default() += 1;
            }
            if mode == "constructive" || mode == "both" {
                for (parent, action) in constructive_predecessors(terminal) {
                    if insert_candidate(
                        &mut candidates,
                        max_candidates,
                        parent,
                        action,
                        format!("constructive:{}", game.key),
                        "constructive-verified-transition",
                        &mut duplicate_witnesses,
                    ) {
                        constructive_transitions += 1;
                    } else if max_candidates > 0 && candidates.len() >= max_candidates {
                        break 'files;
                    }
                }
            }
        }
    }

    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| fail(&format!("cannot create {}: {error}", parent.display())));
    }
    let file = File::create(&output)
        .unwrap_or_else(|error| fail(&format!("cannot create {}: {error}", output.display())));
    let mut writer = BufWriter::new(file);
    for candidate in candidates.values() {
        let state = candidate.state.expect("candidate state is populated");
        let actions = candidate
            .witnesses
            .iter()
            .map(|(action_key, witnesses)| {
                json!({
                    "token": encode_action(action_key.action()),
                    "outcome": "win",
                    "distance": 1,
                    "known": true,
                    "witnessCount": witnesses.len(),
                })
            })
            .collect::<Vec<_>>();
        let witnesses = candidate
            .witnesses
            .iter()
            .flat_map(|(action_key, game_keys)| {
                game_keys.iter().map(move |game_key| {
                    json!({
                        "gameKey": game_key,
                        "action": encode_action(action_key.action()),
                    })
                })
            })
            .collect::<Vec<_>>();
        let record = json!({
            "schemaVersion": 1,
            "tableFamily": "fresh-frontier-wdl-v1",
            "ring": 1,
            "position": position_json(state),
            "outcome": "win",
            "distance": 1,
            "optimalActionsKnown": false,
            "legalActionCount": state.legal_action_count(),
            "actions": actions,
            "witnesses": witnesses,
            "proof": {
                "kind": candidate.proof_kind,
                "rulesVersion": "pathagon-rules-v1",
                "solverVersion": "pathagon-endgame-frontier-v1",
            },
        });
        serde_json::to_writer(&mut writer, &record)
            .unwrap_or_else(|error| fail(&format!("cannot write {}: {error}", output.display())));
        writer
            .write_all(b"\n")
            .unwrap_or_else(|error| fail(&format!("cannot write {}: {error}", output.display())));
    }
    writer
        .flush()
        .unwrap_or_else(|error| fail(&format!("cannot flush {}: {error}", output.display())));

    let known_actions = candidates
        .values()
        .map(|candidate| candidate.witnesses.len())
        .sum::<usize>();
    println!(
        "{}",
        json!({
            "schemaVersion": 1,
            "tableFamily": "fresh-frontier-wdl-v1",
            "ring": 1,
            "corpus": corpus,
            "out": output,
            "gamesSeen": games_seen,
            "terminalGames": terminal_games,
            "skippedNonTerminal": skipped_non_terminal,
            "skippedNonDefault": skipped_non_default,
            "uniquePositions": candidates.len(),
            "knownWinningActions": known_actions,
            "duplicateWitnesses": duplicate_witnesses,
            "constructiveTransitions": constructive_transitions,
            "replayWitnesses": replay_witnesses,
            "replayPlacementWitnesses": replay_placement_witnesses,
            "replayRelocationWitnesses": replay_relocation_witnesses,
            "replayCaptureWitnesses": replay_capture_witnesses,
            "replayPieceDensity": piece_density,
            "mode": mode,
        })
    );
}

/// Export the next predecessor ring as a complete forward-edge graph.  The
/// parent is proven reachable from the replay corpus, and its child list is
/// generated from the Rust legal-action boundary.  Children not present in
/// this export deliberately remain unknown to the retrograde solver until a
/// later shard supplies them.
fn extract_ring2(corpus: &Path, output: &Path, max_games: usize, max_candidates: usize) {
    let mut candidates = BTreeMap::<String, Ring2Candidate>::new();
    let mut seed_keys = BTreeSet::new();
    let mut games_seen = 0_usize;
    let mut terminal_games = 0_usize;
    let mut skipped = 0_usize;
    let mut witnesses = 0_usize;
    'files: for path in corpus_files(corpus) {
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| fail(&format!("cannot read {}: {error}", path.display())));
        for (line_number, line) in source.lines().enumerate() {
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            if max_games > 0 && games_seen >= max_games {
                break 'files;
            }
            let game = parse_unified_game(line).unwrap_or_else(|error| {
                fail(&format!("{}:{}: {error}", path.display(), line_number + 1))
            });
            games_seen += 1;
            if game.config != BoardConfig::DEFAULT || game.actions.len() < 2 {
                skipped += 1;
                continue;
            }
            let (states, terminal) = replay(&game.actions, game.config).unwrap_or_else(|error| {
                fail(&format!("{}:{}: {error}", path.display(), line_number + 1))
            });
            if terminal.is_none() {
                skipped += 1;
                continue;
            }
            terminal_games += 1;
            let parent = states[states.len() - 3];
            let child = states[states.len() - 2];
            let ring1_action = game.actions[game.actions.len() - 1];
            let ring2_action = game.actions[game.actions.len() - 2];
            if child.winner.is_some()
                || parent.winner.is_some()
                || !child.legal_actions().contains(&ring1_action)
                || child.apply_legal(ring1_action).state != states[states.len() - 1]
                || !parent.legal_actions().contains(&ring2_action)
                || parent.apply_legal(ring2_action).state != child
            {
                fail(&format!(
                    "{}:{}: Ring 2 replay lineage is not a verified two-edge suffix",
                    path.display(),
                    line_number + 1
                ));
            }
            let key = canonical_position_key(parent)
                .into_iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            if max_candidates > 0
                && !candidates.contains_key(&key)
                && candidates.len() >= max_candidates
            {
                break 'files;
            }
            let child_key = canonical_position_key(child)
                .into_iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            seed_keys.insert(child_key);
            let candidate = candidates.entry(key).or_default();
            candidate.state = Some(parent);
            for legal_action in parent.legal_actions() {
                let legal_child = parent.apply_legal(legal_action).state;
                let legal_child_key = canonical_position_key(legal_child)
                    .into_iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect::<String>();
                candidate.children.insert(legal_child_key.clone());
                candidate
                    .actions
                    .insert(encode_action(legal_action), legal_child_key);
            }
            candidate
                .witnesses
                .insert((game.key.clone(), encode_action(ring2_action)));
            witnesses += 1;
        }
    }
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .unwrap_or_else(|error| fail(&format!("cannot create {}: {error}", parent.display())));
    }
    let file = File::create(output)
        .unwrap_or_else(|error| fail(&format!("cannot create {}: {error}", output.display())));
    let mut writer = BufWriter::new(file);
    let child_keys = candidates
        .values()
        .flat_map(|candidate| candidate.children.iter().cloned())
        .collect::<BTreeSet<_>>();
    for child_key in &child_keys {
        if let Some(candidate) = candidates.get_mut(child_key) {
            if seed_keys.contains(child_key) {
                candidate.seed = Some(pathagon_engine::tablebase::RetrogradeValue {
                    outcome: pathagon_engine::ground_truth::GroundTruthOutcome::Win,
                    distance: Some(1),
                });
            }
        }
    }
    for (key, candidate) in &candidates {
        let state = candidate
            .state
            .expect("Ring 2 candidate state is populated");
        let witness = candidate.witnesses.iter().next().expect("Ring 2 witness");
        let record = json!({
            "schemaVersion": 2,
            "tableFamily": "pathagon-retrograde-wdl-v1",
            "ring": 2,
            "key": key,
            "position": position_json(state),
            "children": candidate.children,
            "actions": candidate
                .actions
                .iter()
                .map(|(action, child)| json!({"action": action, "child": child}))
                .collect::<Vec<_>>(),
            "complete": true,
            "terminal": null,
            "seed": candidate.seed,
            "witness": {"gameKey": witness.0, "action": witness.1},
            "witnessCount": candidate.witnesses.len(),
            "proof": {
                "kind": "two-edge-forward-replayed-terminal",
                "rulesVersion": "pathagon-rules-v1",
                "solverVersion": "pathagon-endgame-frontier-v2",
                "lineage": "full-corpus-replay-plus-verified-terminal-suffix",
            },
        });
        serde_json::to_writer(&mut writer, &record)
            .unwrap_or_else(|error| fail(&format!("cannot write {}: {error}", output.display())));
        writer
            .write_all(b"\n")
            .unwrap_or_else(|error| fail(&format!("cannot write {}: {error}", output.display())));
    }
    for child_key in &child_keys {
        if candidates.contains_key(child_key) {
            continue;
        }
        let seed = seed_keys.contains(child_key).then_some(json!({
            "outcome": "win",
            "distance": 1,
        }));
        let record = json!({
            "schemaVersion": 2,
            "tableFamily": "pathagon-retrograde-wdl-v1",
            "ring": 1,
            "key": child_key,
            "children": [],
            "complete": false,
            "terminal": null,
            "seed": seed,
            "actions": [],
            "proof": {
                "kind": if seed.is_some() { "ring-1-seed-stub" } else { "unknown-child-stub" },
                "rulesVersion": "pathagon-rules-v1",
                "solverVersion": "pathagon-endgame-frontier-v2",
            },
        });
        serde_json::to_writer(&mut writer, &record)
            .unwrap_or_else(|error| fail(&format!("cannot write {}: {error}", output.display())));
        writer
            .write_all(b"\n")
            .unwrap_or_else(|error| fail(&format!("cannot write {}: {error}", output.display())));
    }
    writer
        .flush()
        .unwrap_or_else(|error| fail(&format!("cannot flush {}: {error}", output.display())));
    println!(
        "{}",
        json!({
            "schemaVersion": 2,
            "tableFamily": "pathagon-retrograde-wdl-v1",
            "ring": 2,
            "corpus": corpus,
            "out": output,
            "gamesSeen": games_seen,
            "terminalGames": terminal_games,
            "skipped": skipped,
            "uniquePositions": candidates.len(),
            "replayWitnesses": witnesses,
            "completeForwardEdges": candidates.values().map(|candidate| candidate.children.len()).sum::<usize>(),
            "graphNodesWithExplicitUnknowns": candidates.len() + child_keys.len(),
            "seededRing1Children": seed_keys.len(),
        })
    );
}

fn insert_candidate(
    candidates: &mut BTreeMap<PositionKey, Candidate>,
    max_candidates: usize,
    state: GameState,
    action: Action,
    witness: String,
    proof_kind: &str,
    duplicate_witnesses: &mut usize,
) -> bool {
    let key = PositionKey::from(state);
    if max_candidates > 0 && !candidates.contains_key(&key) && candidates.len() >= max_candidates {
        return false;
    }
    let candidate = candidates.entry(key).or_default();
    candidate.state = Some(state);
    if candidate.proof_kind.is_empty() || proof_kind == "forward-replayed-terminal" {
        candidate.proof_kind = proof_kind.to_owned();
    }
    let action_key = ActionKey::from(action);
    let witnesses = candidate.witnesses.entry(action_key).or_default();
    if !witnesses.insert(witness) {
        *duplicate_witnesses += 1;
    }
    true
}

/// Construct fresh-root predecessors directly from a terminal board.  This
/// generator is deliberately one-sided: it enumerates legal parents and
/// verifies each candidate by replaying the final action, but does not claim
/// that the arbitrary fresh-root metadata is reachable from an earlier game.
fn constructive_predecessors(terminal: GameState) -> Vec<(GameState, Action)> {
    let Some(winner) = terminal.winner else {
        return Vec::new();
    };
    if terminal.turn != winner.other() {
        return Vec::new();
    }
    let mover = winner;
    let opponent = mover.other();
    let captured = terminal.forbidden;
    let captured_count = captured.count_ones() as u8;
    if terminal.reserve[opponent.index()] < captured_count {
        return Vec::new();
    }
    let mut reserve = terminal.reserve;
    reserve[opponent.index()] -= captured_count;
    let opponent_pieces = terminal.pieces(opponent) | captured;
    let mut output = Vec::new();

    if terminal.last_relocated_to[mover.index()].is_none() {
        reserve[mover.index()] = reserve[mover.index()].saturating_add(1);
        if reserve[mover.index()] <= terminal.config.reserve_per_player {
            for to in squares(terminal.pieces(mover)) {
                let mut markers = terminal.last_relocated_to;
                markers[mover.index()] = None;
                let parent = candidate_parent(
                    terminal,
                    mover,
                    opponent_pieces,
                    reserve,
                    markers,
                    Action::Place { to },
                );
                if let Some(parent) = parent {
                    output.push((parent, Action::Place { to }));
                }
            }
        }
    } else if let Some(to) = terminal.last_relocated_to[mover.index()] {
        let empty_in_child =
            terminal.config.full_board() & !(terminal.light | terminal.dark | terminal.forbidden);
        for from in squares(empty_in_child & !bit(to)) {
            let mut parent_mover = terminal.pieces(mover) & !bit(to);
            parent_mover |= bit(from);
            let mut light = terminal.light;
            let mut dark = terminal.dark;
            match mover {
                Player::Light => light = parent_mover,
                Player::Dark => dark = parent_mover,
            }
            let parent = GameState {
                config: terminal.config,
                light,
                dark,
                reserve,
                turn: mover,
                forbidden: 0,
                last_relocated_to: [
                    if mover == Player::Light {
                        None
                    } else {
                        terminal.last_relocated_to[Player::Light.index()]
                    },
                    if mover == Player::Dark {
                        None
                    } else {
                        terminal.last_relocated_to[Player::Dark.index()]
                    },
                ],
                last_capture: 0,
                last_player: None,
                winner: None,
                ply: terminal.ply.saturating_sub(1),
            };
            let action = Action::Relocate { from, to };
            if parent_has_valid_transition(parent, action, terminal) {
                output.push((parent, action));
            }
        }
    }
    output
}

fn candidate_parent(
    terminal: GameState,
    mover: Player,
    opponent_pieces: u64,
    reserve: [u8; 2],
    last_relocated_to: [Option<u8>; 2],
    action: Action,
) -> Option<GameState> {
    let to = action.destination();
    let mover_pieces = terminal.pieces(mover) & !bit(to);
    let (light, dark) = match mover {
        Player::Light => (mover_pieces, opponent_pieces),
        Player::Dark => (opponent_pieces, mover_pieces),
    };
    let parent = GameState {
        config: terminal.config,
        light,
        dark,
        reserve,
        turn: mover,
        forbidden: 0,
        last_relocated_to,
        last_capture: 0,
        last_player: None,
        winner: None,
        ply: terminal.ply.saturating_sub(1),
    };
    parent_has_valid_transition(parent, action, terminal).then_some(parent)
}

fn parent_has_valid_transition(parent: GameState, action: Action, terminal: GameState) -> bool {
    if has_winning_path(parent, Player::Light) || has_winning_path(parent, Player::Dark) {
        return false;
    }
    parent.legal_actions().contains(&action) && parent.apply_legal(action).state == terminal
}

fn bit(square: u8) -> u64 {
    1_u64 << square
}

fn squares(mask: u64) -> impl Iterator<Item = u8> {
    (0..64).filter(move |square| mask & bit(*square) != 0)
}

fn replay(
    actions: &[Action],
    config: BoardConfig,
) -> Result<(Vec<GameState>, Option<GameState>), String> {
    let mut state = GameState::with_config(config);
    let mut states = vec![state];
    for (index, action) in actions.iter().copied().enumerate() {
        if state.winner.is_some() {
            return Err(format!("action {} follows a terminal state", index));
        }
        if !state.legal_actions().contains(&action) {
            return Err(format!("illegal action {action} at ply {}", state.ply));
        }
        state = state.apply_legal(action).state;
        states.push(state);
    }
    Ok((states, state.winner.map(|_| state)))
}

fn position_json(state: GameState) -> Value {
    json!({
        "boardSize": state.config.board_size,
        "reservePerPlayer": state.config.reserve_per_player,
        "light": state.light,
        "dark": state.dark,
        "reserve": state.reserve,
        "turn": state.turn.as_str(),
        "forbidden": state.forbidden,
        "lastRelocatedTo": state.last_relocated_to,
        "ply": state.ply,
    })
}

fn corpus_files(path: &Path) -> Vec<PathBuf> {
    if path.is_file() {
        return vec![path.to_path_buf()];
    }
    let mut files = fs::read_dir(path)
        .unwrap_or_else(|error| {
            fail(&format!(
                "cannot read corpus directory {}: {error}",
                path.display()
            ))
        })
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("games-") && name.ends_with(".tsv"))
        })
        .collect::<Vec<_>>();
    files.sort();
    if files.is_empty() {
        fail(&format!(
            "no games-*.tsv files found under {}",
            path.display()
        ));
    }
    files
}

fn parse_args() -> HashMap<String, String> {
    let mut args = env::args().skip(1);
    let mut values = HashMap::new();
    while let Some(argument) = args.next() {
        let key = argument
            .strip_prefix("--")
            .unwrap_or_else(|| fail(&format!("unexpected argument {argument}")));
        let value = args
            .next()
            .unwrap_or_else(|| fail(&format!("missing value for --{key}")));
        if value.starts_with("--") {
            fail(&format!("missing value for --{key}"));
        }
        values.insert(key.to_owned(), value);
    }
    values
}

fn required_path(args: &HashMap<String, String>, key: &str) -> PathBuf {
    args.get(key)
        .map(PathBuf::from)
        .unwrap_or_else(|| fail(&format!("--{key} <path> is required")))
}

fn number(args: &HashMap<String, String>, key: &str) -> usize {
    args.get(key)
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|_| fail(&format!("--{key} must be a non-negative integer")))
        })
        .unwrap_or(0)
}

fn fail(message: &str) -> ! {
    eprintln!("pathagon-endgame-frontier: {message}");
    std::process::exit(2);
}
