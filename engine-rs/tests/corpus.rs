use std::fs;
use std::path::PathBuf;

use pathagon_engine::corpus::{parse_compact_game, StrategyBook};

#[test]
fn tracked_corpus_is_replayable_and_indexed() {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../corpus/rust-v1");
    let source = fs::read_to_string(directory.join("games.tsv")).expect("read tracked games");
    let mut games = 0;
    for (line_number, line) in source.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let game = parse_compact_game(line)
            .unwrap_or_else(|error| panic!("game line {}: {error}", line_number + 1));
        game.replay()
            .unwrap_or_else(|error| panic!("game line {}: {error}", line_number + 1));
        games += 1;
    }
    assert_eq!(games, 64);

    let book = StrategyBook::load(&directory.join("positions.tsv")).expect("read strategy book");
    assert_eq!(book.len(), 2_076);
}
