use std::fs;
use std::path::PathBuf;

use pathagon_engine::contract::{Position, ReplayRecord};

#[test]
fn shared_replay_fixture_validates_in_rust() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../contracts/fixtures/replay-v1.json");
    let record = ReplayRecord::from_json(&fs::read_to_string(path).expect("read contract fixture"))
        .expect("valid contract fixture");
    assert_eq!(record.contract_version, 1);
    assert_eq!(record.config.board_size, 3);
    assert_eq!(record.agents.light, record.agent_specifications.light.id);
}

#[test]
fn root_q_targets_are_optional_but_must_be_complete_and_aligned() {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../contracts/fixtures/replay-v1.json");
    let mut record =
        ReplayRecord::from_json(&fs::read_to_string(path).expect("read contract fixture"))
            .expect("valid contract fixture");
    record.moves[0].action_values = Some(vec![-0.25, 0.75]);
    record.moves[0].action_visits = Some(vec![2, 10]);
    record.moves[0].action_value_source = Some("mcts-root-q-v1".to_owned());
    record.validate().expect("valid root-Q fields");

    record.moves[0].action_visits = Some(vec![2]);
    assert!(record.validate().is_err());
    record.moves[0].action_visits = Some(vec![2, 10]);
    record.moves[0].action_value_source = Some("other-source".to_owned());
    assert!(record.validate().is_err());
}

#[test]
fn position_contract_validates_rule_relevant_state() {
    let position: Position = serde_json::from_value(serde_json::json!({
        "contractVersion": 1,
        "config": {"rulesVersion": "pathagon-rules-v1", "boardSize": 3, "reservePerPlayer": 6, "maxPlies": 36, "repetitionLimit": 3},
        "board": ["light", null, null, null, "dark", null, null, null, null],
        "reserve": {"light": 5, "dark": 6},
        "turn": "dark",
        "forbidden": [],
        "lastRelocatedTo": {"light": null, "dark": null},
        "winner": null,
        "ply": 1
    })).expect("decode position");
    position.validate().expect("valid position");
}
