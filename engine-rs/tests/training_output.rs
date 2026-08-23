use std::fs;
use std::path::Path;

use pathagon_engine::corpus::{parse_compact_game, StrategyBook};

#[test]
fn tracked_training_splits_are_replayable_and_separate() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../training/rust-v1/corpus");
    verify_split(&root.join("training"), 32, 808);
    verify_split(&root.join("evaluation"), 16, 797);
}

fn verify_split(directory: &Path, expected_games: usize, expected_positions: usize) {
    let source = fs::read_to_string(directory.join("games.tsv")).expect("read games");
    let mut games = 0;
    for (line_number, line) in source.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        parse_compact_game(line)
            .unwrap_or_else(|error| panic!("game line {}: {error}", line_number + 1))
            .replay()
            .unwrap_or_else(|error| panic!("game line {}: {error}", line_number + 1));
        games += 1;
    }
    assert_eq!(games, expected_games);
    assert_eq!(StrategyBook::load(&directory.join("positions.tsv")).unwrap().len(), expected_positions);
}
