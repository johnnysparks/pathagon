use std::collections::{BTreeMap, HashSet};
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};

use pathagon_engine::contract::{ContractAction, ContractPlayer, ReplayRecord};
use pathagon_engine::{Action, BoardConfig, GameState, Player};
use serde_json::json;

fn required_arg(name: &str) -> String {
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == name {
            return args
                .next()
                .unwrap_or_else(|| panic!("missing value for {name}"));
        }
    }
    panic!("missing {name}");
}

fn player(value: ContractPlayer) -> Player {
    match value {
        ContractPlayer::Light => Player::Light,
        ContractPlayer::Dark => Player::Dark,
    }
}

fn action(value: &ContractAction) -> Action {
    match value {
        ContractAction::Place { to } => Action::Place { to: *to },
        ContractAction::Relocate { from, to } => Action::Relocate {
            from: *from,
            to: *to,
        },
    }
}

fn squares(mask: u64, cells: u8) -> Vec<u8> {
    (0..cells)
        .filter(|square| mask & (1_u64 << square) != 0)
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = required_arg("--arena");
    let file = File::open(&path)?;
    let mut games = 0_u64;
    let mut plies = 0_u64;
    let mut captures = 0_u64;
    let mut action_kinds = BTreeMap::<&str, u64>::from([("place", 0), ("relocate", 0)]);
    let mut by_color = BTreeMap::<&str, u64>::from([("light", 0), ("dark", 0)]);
    let mut reasons = BTreeMap::<String, u64>::new();
    let mut winners = BTreeMap::<String, u64>::new();
    let mut sequence_set = HashSet::<String>::new();
    let mut duplicate_sequences = 0_u64;

    for (line_number, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record = ReplayRecord::from_json(&line)
            .map_err(|error| format!("line {} contract validation: {error}", line_number + 1))?;
        let mut state = match record.initial_position.as_ref() {
            Some(position) => GameState::from_position(position)
                .map_err(|error| format!("line {} initial position: {error}", line_number + 1))?,
            None => BoardConfig::from_contract(&record.config)
                .map(GameState::with_config)
                .map_err(|error| format!("line {} config: {error}", line_number + 1))?,
        };
        let mut sequence = Vec::with_capacity(record.moves.len());
        for movement in &record.moves {
            let expected_player = player(movement.player);
            if state.turn != expected_player {
                return Err(format!(
                    "line {} ply {} turn mismatch: state={:?}, record={:?}",
                    line_number + 1,
                    movement.ply,
                    state.turn,
                    expected_player
                )
                .into());
            }
            let next = state.apply(action(&movement.action)).map_err(|error| {
                format!(
                    "line {} ply {} illegal action: {error}",
                    line_number + 1,
                    movement.ply
                )
            })?;
            let actual_captured = squares(next.captured, state.config.cells());
            if actual_captured != movement.captured {
                return Err(format!(
                    "line {} ply {} capture mismatch: record={:?}, replay={:?}",
                    line_number + 1,
                    movement.ply,
                    movement.captured,
                    actual_captured
                )
                .into());
            }
            match movement.action {
                ContractAction::Place { .. } => *action_kinds.get_mut("place").unwrap() += 1,
                ContractAction::Relocate { .. } => *action_kinds.get_mut("relocate").unwrap() += 1,
            }
            *by_color.get_mut(expected_player.as_str()).unwrap() += 1;
            captures += movement.captured.len() as u64;
            sequence.push(match movement.action {
                ContractAction::Place { to } => format!("P{to}"),
                ContractAction::Relocate { from, to } => format!("R{from}>{to}"),
            });
            state = next.state;
        }
        let replay_winner = state.winner.map(|value| match value {
            Player::Light => ContractPlayer::Light,
            Player::Dark => ContractPlayer::Dark,
        });
        if replay_winner != record.winner {
            return Err(format!(
                "line {} winner mismatch: record={:?}, replay={:?}",
                line_number + 1,
                record.winner,
                replay_winner
            )
            .into());
        }
        let sequence = sequence.join(" ");
        if !sequence_set.insert(sequence) {
            duplicate_sequences += 1;
        }
        *reasons.entry(record.reason.clone()).or_default() += 1;
        let winner = record
            .winner
            .map(|value| match value {
                ContractPlayer::Light => "light",
                ContractPlayer::Dark => "dark",
            })
            .unwrap_or("draw");
        *winners.entry(winner.to_owned()).or_default() += 1;
        games += 1;
        plies += record.moves.len() as u64;
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "arena": path,
            "games": games,
            "plies": plies,
            "captures": captures,
            "actionKinds": action_kinds,
            "byColor": by_color,
            "reasons": reasons,
            "winners": winners,
            "duplicateSequences": duplicate_sequences,
            "uniqueSequences": sequence_set.len(),
        }))?
    );
    Ok(())
}
