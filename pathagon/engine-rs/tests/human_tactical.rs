use pathagon_engine::{Action, BoardConfig, GameState, Player};
use serde::Deserialize;

const SUITE: &str = include_str!("../../../data/fixtures/human-tactical-suite-v1.jsonl");

#[derive(Debug, Deserialize)]
struct SuiteHeader {
    schema: String,
    #[serde(rename = "fixtureVersion")]
    fixture_version: u8,
    #[serde(rename = "sourceGameId")]
    source_game_id: String,
    #[serde(rename = "sourceOpponent")]
    source_opponent: String,
    #[serde(rename = "sourceWinner")]
    source_winner: String,
    #[serde(rename = "sourcePlies")]
    source_plies: u16,
    provenance: String,
    config: SuiteConfig,
    count: usize,
}

#[derive(Debug, Deserialize)]
struct SuiteConfig {
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
    #[serde(rename = "sourcePly")]
    source_ply: u16,
    categories: Vec<String>,
    state: FixtureState,
    labels: FixtureLabels,
}

#[derive(Debug, Deserialize)]
struct FixtureState {
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

#[derive(Debug, Deserialize)]
struct FixtureLabels {
    #[serde(rename = "expectedOutcome")]
    expected_outcome: String,
    #[serde(rename = "humanAction")]
    human_action: Option<FixtureAction>,
    #[serde(rename = "resultingFixture")]
    resulting_fixture: Option<String>,
    #[serde(rename = "forcedLightReplies", default)]
    forced_light_replies: Vec<FixtureAction>,
    #[serde(rename = "safeDarkActions", default)]
    safe_dark_actions: Vec<FixtureAction>,
    #[serde(rename = "recordedDarkAction")]
    recorded_dark_action: Option<FixtureAction>,
    #[serde(rename = "recordedReply")]
    recorded_reply: Option<FixtureAction>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum FixtureAction {
    Place { to: u8 },
    Relocate { from: u8, to: u8 },
}

impl From<FixtureAction> for Action {
    fn from(action: FixtureAction) -> Self {
        match action {
            FixtureAction::Place { to } => Self::Place { to },
            FixtureAction::Relocate { from, to } => Self::Relocate { from, to },
        }
    }
}

fn load_suite() -> (SuiteHeader, Vec<Fixture>) {
    let mut lines = SUITE.lines().filter(|line| !line.trim().is_empty());
    let header: SuiteHeader = serde_json::from_str(lines.next().expect("suite header"))
        .expect("valid human tactical suite header");
    let fixtures = lines
        .map(|line| serde_json::from_str(line).expect("valid human tactical fixture"))
        .collect::<Vec<Fixture>>();
    (header, fixtures)
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

fn state(fixture: &FixtureState, config: &SuiteConfig) -> GameState {
    let config = BoardConfig::new(config.board_size, config.reserve_per_player)
        .expect("valid fixture board configuration")
        .with_max_plies(config.max_plies)
        .expect("valid fixture ply limit");
    GameState {
        config,
        light: mask(&fixture.light),
        dark: mask(&fixture.dark),
        reserve: fixture.reserve,
        turn: player(&fixture.turn),
        forbidden: mask(&fixture.forbidden),
        last_relocated_to: fixture.last_relocated_to,
        last_capture: fixture.last_capture,
        last_player: fixture.last_player.as_deref().map(player),
        winner: fixture.winner.as_deref().map(player),
        ply: fixture.ply,
    }
}

fn find<'a>(fixtures: &'a [Fixture], id: &str) -> &'a Fixture {
    fixtures
        .iter()
        .find(|fixture| fixture.id == id)
        .unwrap_or_else(|| panic!("missing fixture: {id}"))
}

#[test]
fn human_tactical_suite_has_replay_provenance_and_two_positions() {
    let (header, fixtures) = load_suite();
    assert_eq!(header.schema, "pathagon-human-tactical-suite-v1");
    assert_eq!(header.fixture_version, 1);
    assert_eq!(
        header.source_game_id,
        "08de361a-de4b-425a-8a98-801408c49dee"
    );
    assert_eq!(header.source_opponent, "pathfinder-v0");
    assert_eq!(header.source_winner, "light");
    assert_eq!(header.source_plies, 33);
    assert!(!header.provenance.is_empty());
    assert_eq!(header.config.board_size, 7);
    assert_eq!(header.config.reserve_per_player, 14);
    assert_eq!(header.config.max_plies, 180);
    assert_eq!(header.count, 2);
    assert_eq!(fixtures.len(), header.count);
    assert!(fixtures
        .iter()
        .all(|fixture| fixture.categories.contains(&"human-refutation".to_owned())));
}

#[test]
fn human_d7_move_reconstructs_the_mined_fork_setup() {
    let (header, fixtures) = load_suite();
    let before = find(
        &fixtures,
        "08de361a-de4b-425a-8a98-801408c49dee-before-ply31",
    );
    let after = find(
        &fixtures,
        "08de361a-de4b-425a-8a98-801408c49dee-after-ply31",
    );
    assert_eq!(before.source_ply, 30);
    assert_eq!(after.source_ply, 31);
    assert_eq!(before.labels.expected_outcome, "light-forced-win");
    assert_eq!(
        before.labels.resulting_fixture.as_deref(),
        Some(after.id.as_str())
    );

    let before_state = state(&before.state, &header.config);
    let action: Action = before
        .labels
        .human_action
        .expect("fork setup has a mined human move")
        .into();
    let transition = before_state
        .apply(action)
        .expect("mined human move is legal");
    assert_eq!(transition.state, state(&after.state, &header.config));
    assert_eq!(transition.state.turn, Player::Dark);
    assert_eq!(transition.state.winner, None);
    assert_eq!(transition.captured, mask(&[46]));
}

#[test]
fn post_d7_position_proves_the_forced_c3_or_d3_finish() {
    let (header, fixtures) = load_suite();
    let after = find(
        &fixtures,
        "08de361a-de4b-425a-8a98-801408c49dee-after-ply31",
    );
    let state = state(&after.state, &header.config);
    assert_eq!(state.turn, Player::Dark);
    assert_eq!(after.labels.expected_outcome, "dark-forced-loss");

    let expected: Vec<Action> = after
        .labels
        .forced_light_replies
        .iter()
        .copied()
        .map(Into::into)
        .collect();
    let safe_dark: Vec<Action> = state
        .legal_actions()
        .into_iter()
        .filter(|dark_action| {
            let child = state.apply_legal(*dark_action).state;
            child.winner.is_none()
                && child.legal_actions().into_iter().all(|light_action| {
                    child.apply_legal(light_action).state.winner != Some(Player::Light)
                })
        })
        .collect();
    let declared_safe: Vec<Action> = after
        .labels
        .safe_dark_actions
        .iter()
        .copied()
        .map(Into::into)
        .collect();
    assert_eq!(safe_dark, declared_safe);
    assert!(
        safe_dark.is_empty(),
        "the mined position must remain a forced loss"
    );

    let mut winning_union = Vec::new();
    for dark_action in state.legal_actions() {
        let child = state.apply_legal(dark_action).state;
        assert_ne!(child.winner, Some(Player::Dark));
        let light_wins: Vec<Action> = child
            .legal_actions()
            .into_iter()
            .filter(|light_action| {
                child.apply_legal(*light_action).state.winner == Some(Player::Light)
            })
            .collect();
        assert!(
            !light_wins.is_empty(),
            "dark reply {dark_action} must leave a light win"
        );
        assert!(light_wins.iter().all(|action| expected.contains(action)));
        for action in light_wins {
            if !winning_union.contains(&action) {
                winning_union.push(action);
            }
        }
    }
    winning_union.sort_by_key(|action| action.order());
    let mut expected_sorted = expected;
    expected_sorted.sort_by_key(|action| action.order());
    assert_eq!(winning_union, expected_sorted);

    let recorded_dark: Action = after
        .labels
        .recorded_dark_action
        .expect("fixture records the played dark reply")
        .into();
    let recorded_reply: Action = after
        .labels
        .recorded_reply
        .expect("fixture records the played light finish")
        .into();
    let child = state
        .apply(recorded_dark)
        .expect("recorded dark move is legal")
        .state;
    assert!(child
        .legal_actions()
        .into_iter()
        .any(|action| action == recorded_reply
            && child.apply_legal(action).state.winner == Some(Player::Light)));
}
