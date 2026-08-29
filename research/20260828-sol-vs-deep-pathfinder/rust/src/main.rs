use std::collections::HashMap;
use std::env;

use pathagon_engine::search::{
    analyze_actions, search_best_action, EvaluationWeights, SearchConfig,
};
use pathagon_engine::{Action, BoardConfig, GameState, Player};
use serde_json::{json, Value};

fn main() {
    let args = parse_args();
    let sequence = args.get("sequence").map(String::as_str).unwrap_or("");
    let state = replay(sequence).unwrap_or_else(|error| fail(&error));
    let mode = args.get("mode").map(String::as_str).unwrap_or("inspect");

    let output = match mode {
        "inspect" => inspect(state, sequence, number(&args, "count", 12_usize)),
        "pathfinder" => pathfinder(state, sequence),
        other => fail(&format!("unknown --mode {other}")),
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&output).expect("serialize referee output")
    );
}

fn inspect(state: GameState, sequence: &str, count: usize) -> Value {
    let config = SearchConfig {
        depth: 2,
        max_nodes: 20_000,
        beam_width: 24,
        weights: EvaluationWeights::default(),
        tactical_proof_horizon: None,
    };
    let analyses = if state.winner.is_none() {
        analyze_actions(state, config, count)
    } else {
        Vec::new()
    };
    json!({
        "sequence": sequence,
        "ply": state.ply,
        "turn": state.turn.as_str(),
        "winner": state.winner.map(Player::as_str),
        "reserve": {"light": state.reserve[0], "dark": state.reserve[1]},
        "lastCapture": state.last_capture,
        "forbidden": squares(state.forbidden),
        "board": board_rows(state),
        "legalActionCount": state.legal_action_count(),
        "shortlist": analyses.into_iter().map(|item| json!({
            "action": item.action.to_string(),
            "score": item.score,
            "delta": item.delta,
            "nodes": item.nodes,
            "exhausted": item.exhausted,
        })).collect::<Vec<_>>(),
    })
}

fn pathfinder(state: GameState, sequence: &str) -> Value {
    if state.winner.is_some() {
        return inspect(state, sequence, 0);
    }
    if state.turn != Player::Dark {
        fail("deep Pathfinder is Dark; --mode pathfinder requires Dark to move");
    }
    let result = search_best_action(
        state,
        SearchConfig {
            depth: 5,
            max_nodes: 2_000,
            beam_width: 8,
            weights: EvaluationWeights::default(),
            tactical_proof_horizon: None,
        },
    );
    let post_move = result.action.map(|action| transition_value(state, action));
    json!({
        "sequence": sequence,
        "ply": state.ply,
        "turn": state.turn.as_str(),
        "board": board_rows(state),
        "action": result.action.map(|action| action.to_string()),
        "score": result.score,
        "nodes": result.nodes,
        "completedDepth": result.completed_depth,
        "exhausted": result.exhausted,
        "tableHits": result.table_hits,
        "postMove": post_move,
    })
}

fn transition_value(before: GameState, action: Action) -> Value {
    let transition = before.apply_legal(action);
    let after = transition.state;
    json!({
        "player": before.turn.as_str(),
        "action": action.to_string(),
        "captured": squares(transition.captured),
        "board": board_rows(after),
        "reserve": {"light": after.reserve[0], "dark": after.reserve[1]},
        "forbidden": squares(after.forbidden),
        "lastCapture": after.last_capture,
        "turn": after.turn.as_str(),
        "winner": after.winner.map(Player::as_str),
        "ply": after.ply,
    })
}

fn replay(sequence: &str) -> Result<GameState, String> {
    let config = BoardConfig::new(7, 14)?.with_max_plies(160)?;
    let mut state = GameState::with_config(config);
    for (index, token) in sequence
        .split(',')
        .filter(|token| !token.is_empty())
        .enumerate()
    {
        if state.winner.is_some() {
            return Err(format!(
                "sequence continues after the winner at ply {}",
                state.ply
            ));
        }
        let action = parse_action(token)?;
        if !state.legal_actions().contains(&action) {
            return Err(format!(
                "illegal action {token} at sequence item {}",
                index + 1
            ));
        }
        state = state.apply_legal(action).state;
    }
    Ok(state)
}

fn parse_action(token: &str) -> Result<Action, String> {
    if let Some(to) = token.strip_prefix('P') {
        return Ok(Action::Place {
            to: to
                .parse()
                .map_err(|_| format!("invalid placement {token}"))?,
        });
    }
    if let Some((from, to)) = token
        .strip_prefix('R')
        .and_then(|value| value.split_once('>'))
    {
        return Ok(Action::Relocate {
            from: from
                .parse()
                .map_err(|_| format!("invalid relocation {token}"))?,
            to: to
                .parse()
                .map_err(|_| format!("invalid relocation {token}"))?,
        });
    }
    Err(format!("invalid action {token}"))
}

fn board_rows(state: GameState) -> Vec<String> {
    (0..state.config.board_size)
        .map(|row| {
            (0..state.config.board_size)
                .map(|column| {
                    let square = row * state.config.board_size + column;
                    match state.board_at(square) {
                        Some(Player::Light) => 'L',
                        Some(Player::Dark) => 'D',
                        None if state.forbidden & (1_u64 << square) != 0 => '*',
                        None => '.',
                    }
                })
                .collect()
        })
        .collect()
}

fn squares(mut mask: u64) -> Vec<u8> {
    let mut values = Vec::new();
    while mask != 0 {
        let square = mask.trailing_zeros() as u8;
        values.push(square);
        mask &= mask - 1;
    }
    values
}

fn parse_args() -> HashMap<String, String> {
    let mut parsed = HashMap::new();
    let values: Vec<String> = env::args().skip(1).collect();
    let mut index = 0;
    while index < values.len() {
        let value = &values[index];
        if let Some(key) = value.strip_prefix("--") {
            if let Some(next) = values.get(index + 1).filter(|next| !next.starts_with("--")) {
                parsed.insert(key.to_owned(), next.clone());
                index += 1;
            } else {
                parsed.insert(key.to_owned(), "true".to_owned());
            }
        }
        index += 1;
    }
    parsed
}

fn number<T>(args: &HashMap<String, String>, key: &str, fallback: T) -> T
where
    T: std::str::FromStr,
{
    args.get(key)
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}

fn fail(message: &str) -> ! {
    eprintln!("sol-pathfinder-referee: {message}");
    std::process::exit(2)
}
