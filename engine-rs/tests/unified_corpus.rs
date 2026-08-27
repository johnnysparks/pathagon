use std::fs;
use std::path::PathBuf;

use pathagon_engine::corpus::parse_unified_game;

#[test]
fn checked_in_unified_corpus_parses_and_replays_in_rust() {
    let corpus = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../research/corpora/games-v1");
    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(corpus.join("manifest.json")).unwrap()).unwrap();
    let expected = manifest["games"].as_u64().unwrap() as usize;
    let mut games = 0;
    let mut shards = fs::read_dir(corpus.join("games"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    shards.sort();
    for shard in shards {
        for (line_number, line) in fs::read_to_string(&shard).unwrap().lines().enumerate() {
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let game = parse_unified_game(line)
                .unwrap_or_else(|error| panic!("{}:{}: {error}", shard.display(), line_number + 1));
            game.replay().unwrap_or_else(|error| {
                panic!(
                    "{}:{}: replay failed: {error}",
                    shard.display(),
                    line_number + 1
                )
            });
            games += 1;
        }
    }
    assert_eq!(games, expected);
}
