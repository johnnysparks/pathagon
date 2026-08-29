use pathagon_engine::{BoardConfig, GameState, Player};
use serde::Deserialize;

const SUITE: &str = include_str!("../../../data/fixtures/pathfinder-browser-suite-v1.jsonl");

#[derive(Debug, Deserialize)]
struct Header {
    schema: String,
    #[serde(rename = "fixtureVersion")]
    fixture_version: u8,
    config: Config,
    count: usize,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct Config {
    #[serde(rename = "boardSize")]
    board_size: u8,
    #[serde(rename = "reservePerPlayer")]
    reserve_per_player: u8,
    #[serde(rename = "maxPlies")]
    max_plies: u16,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    id: String,
    #[serde(rename = "expectedLegalActions")]
    expected_legal_actions: usize,
    state: State,
}

#[derive(Debug, Deserialize)]
struct State {
    light: Vec<u8>,
    dark: Vec<u8>,
    reserve: [u8; 2],
    turn: String,
    forbidden: Vec<u8>,
    #[serde(rename = "lastRelocatedTo")]
    last_relocated_to: [Option<u8>; 2],
    #[serde(rename = "lastCapture")]
    last_capture: u8,
    #[serde(rename = "lastPlayer")]
    last_player: Option<String>,
    winner: Option<String>,
    ply: u16,
}

fn mask(squares: &[u8]) -> u64 {
    squares
        .iter()
        .fold(0, |result, square| result | (1_u64 << square))
}

fn player(value: &str) -> Player {
    match value {
        "light" => Player::Light,
        "dark" => Player::Dark,
        _ => panic!("invalid player: {value}"),
    }
}

fn load_suite() -> (Header, Vec<Fixture>) {
    let mut lines = SUITE.lines().filter(|line| !line.trim().is_empty());
    let header = serde_json::from_str(lines.next().expect("suite header")).expect("valid header");
    let fixtures = lines
        .map(|line| serde_json::from_str(line).expect("valid fixture"))
        .collect();
    (header, fixtures)
}

#[test]
fn browser_benchmark_fixture_matches_rust_legal_action_counts() {
    let (header, fixtures) = load_suite();
    assert_eq!(header.schema, "pathagon-search-browser-suite-v1");
    assert_eq!(header.fixture_version, 1);
    assert_eq!(header.config.board_size, 7);
    assert_eq!(header.config.reserve_per_player, 14);
    assert_eq!(header.config.max_plies, 180);
    assert_eq!(fixtures.len(), header.count);

    let config = BoardConfig::new(header.config.board_size, header.config.reserve_per_player)
        .expect("valid board configuration")
        .with_max_plies(header.config.max_plies)
        .expect("valid ply limit");
    for fixture in fixtures {
        let state = GameState {
            config,
            light: mask(&fixture.state.light),
            dark: mask(&fixture.state.dark),
            reserve: fixture.state.reserve,
            turn: player(&fixture.state.turn),
            forbidden: mask(&fixture.state.forbidden),
            last_relocated_to: fixture.state.last_relocated_to,
            last_capture: fixture.state.last_capture,
            last_player: fixture.state.last_player.as_deref().map(player),
            winner: fixture.state.winner.as_deref().map(player),
            ply: fixture.state.ply,
        };
        assert_eq!(
            state.legal_actions().len(),
            fixture.expected_legal_actions,
            "{}",
            fixture.id
        );
    }
}
