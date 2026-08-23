use std::fs;
use std::path::PathBuf;

use pathagon_engine::{bit_squares, parse_action, GameState, Player};

#[test]
fn shared_rule_fixtures_match_rust_engine() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures/rules-parity.tsv");
    let fixture = fs::read_to_string(path).expect("read shared parity fixture");
    for (line_number, line) in fixture.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), 12, "fixture line {} has wrong field count", line_number + 1);
        let name = fields[0];
        let mut state = GameState::new();
        state.light = 0;
        state.dark = 0;
        if fields[1] != "-" {
            for placement in fields[1].split(',') {
                let (square, color) = placement.split_at(placement.len() - 1);
                let mask = 1_u64 << square.parse::<u8>().expect("placement square");
                match color {
                    "L" => state.light |= mask,
                    "D" => state.dark |= mask,
                    _ => panic!("{name}: invalid placement color"),
                }
            }
        }
        state.turn = player(fields[2]);
        state.reserve = [fields[3].parse().unwrap(), fields[4].parse().unwrap()];
        state.forbidden = mask(fields[5]);
        state.last_relocated_to = [optional_square(fields[6]), optional_square(fields[7])];
        let action = parse_action(fields[8]).unwrap();
        let expected_legal = fields[9] == "true";
        let transition = state.apply(action);
        assert_eq!(transition.is_ok(), expected_legal, "{name}: legality mismatch");
        if let Ok(transition) = transition {
            let expected_winner = if fields[10] == "-" { None } else { Some(player(fields[10])) };
            assert_eq!(transition.state.winner, expected_winner, "{name}: winner mismatch");
            assert_eq!(bit_squares(transition.captured), squares(fields[11]), "{name}: captures mismatch");
        }
    }
}

fn player(value: &str) -> Player {
    match value {
        "light" => Player::Light,
        "dark" => Player::Dark,
        _ => panic!("invalid player: {value}"),
    }
}

fn optional_square(value: &str) -> Option<u8> {
    (value != "-").then(|| value.parse().unwrap())
}

fn squares(value: &str) -> Vec<u8> {
    if value == "-" { Vec::new() } else { value.split(',').map(|square| square.parse().unwrap()).collect() }
}

fn mask(value: &str) -> u64 {
    squares(value).into_iter().fold(0, |result, square| result | (1_u64 << square))
}
