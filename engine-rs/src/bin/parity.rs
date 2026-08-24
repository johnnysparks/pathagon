use std::env;
use std::fs;

use pathagon_engine::{bit_squares, Action, BoardConfig, GameState, Player};
use serde_json::{json, Value};

fn main() {
    let path = env::args()
        .nth(1)
        .expect("usage: parity <fixture.json>");
    let fixture: Value =
        serde_json::from_str(&fs::read_to_string(path).expect("read parity fixture"))
            .expect("parse parity fixture");
    assert_eq!(
        fixture["fixtureVersion"], 1,
        "unsupported parity fixture version"
    );
    let output = fixture["cases"]
        .as_array()
        .expect("parity cases")
        .iter()
        .map(run_case)
        .collect::<Vec<_>>();
    println!(
        "{}",
        serde_json::to_string(&output).expect("serialize parity output")
    );
}

fn run_case(case: &Value) -> Value {
    let state = make_state(&case["position"]);
    let actions = state.legal_actions();
    let transitions = actions.iter().map(|action| {
        json!({"action": action_value(*action), "state": state_value(state.apply_legal(*action).state)})
    }).collect::<Vec<_>>();
    json!({
        "name": case["name"],
        "config": {
            "rulesVersion": "pathagon-rules-v1",
            "boardSize": state.config.board_size,
            "reservePerPlayer": state.config.reserve_per_player,
            "maxPlies": state.config.board_size as u16 * state.config.board_size as u16 * 4,
            "repetitionLimit": 3,
        },
        "state": state_value(state),
        "legalActions": actions.iter().map(|action| action_value(*action)).collect::<Vec<_>>(),
        "transitions": transitions,
    })
}

fn make_state(raw: &Value) -> GameState {
    let config_value = &raw["config"];
    let board_size = config_value["boardSize"].as_u64().expect("board size") as u8;
    let reserve_per_player = config_value["reservePerPlayer"].as_u64().expect("reserve") as u8;
    let config = BoardConfig::new(board_size, reserve_per_player).expect("valid board config");
    let mut state = GameState::with_config(config);
    for (square, piece) in raw["board"].as_array().expect("board").iter().enumerate() {
        if piece == "light" {
            state.light |= 1_u64 << square;
        }
        if piece == "dark" {
            state.dark |= 1_u64 << square;
        }
    }
    state.reserve = [
        raw["reserve"]["light"].as_u64().expect("light reserve") as u8,
        raw["reserve"]["dark"].as_u64().expect("dark reserve") as u8,
    ];
    state.turn = player(&raw["turn"]);
    state.forbidden = squares_mask(&raw["forbidden"]);
    state.last_relocated_to = [
        optional_square(&raw["lastRelocatedTo"]["light"]),
        optional_square(&raw["lastRelocatedTo"]["dark"]),
    ];
    state.winner = raw["winner"].as_str().map(player_name);
    state.ply = raw["ply"].as_u64().expect("ply") as u16;
    state
}

fn state_value(state: GameState) -> Value {
    let board = (0..state.config.cells())
        .map(|square| state.board_at(square).map(Player::as_str))
        .collect::<Vec<_>>();
    json!({
        "board": board,
        "reserve": {"light": state.reserve[0], "dark": state.reserve[1]},
        "turn": state.turn.as_str(),
        "forbidden": bit_squares(state.forbidden),
        "lastRelocatedTo": {"light": state.last_relocated_to[0], "dark": state.last_relocated_to[1]},
        "winner": state.winner.map(Player::as_str),
        "ply": state.ply,
    })
}

fn action_value(action: Action) -> Value {
    match action {
        Action::Place { to } => json!({"kind": "place", "to": to}),
        Action::Relocate { from, to } => json!({"kind": "relocate", "from": from, "to": to}),
    }
}

fn player(raw: &Value) -> Player {
    player_name(raw.as_str().expect("player"))
}

fn player_name(value: &str) -> Player {
    match value {
        "light" => Player::Light,
        "dark" => Player::Dark,
        _ => panic!("invalid player"),
    }
}

fn optional_square(value: &Value) -> Option<u8> {
    value.as_u64().map(|square| square as u8)
}

fn squares_mask(value: &Value) -> u64 {
    value
        .as_array()
        .expect("squares")
        .iter()
        .fold(0, |mask, square| {
            mask | (1_u64 << square.as_u64().expect("square") as u8)
        })
}
