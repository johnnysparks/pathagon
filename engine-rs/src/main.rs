use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use pathagon_engine::corpus::{write_corpus, StrategyBook};
use pathagon_engine::search::{EvaluationWeights, SearchConfig};
use pathagon_engine::selfplay::{play_game, Agent, MatchOptions};
use pathagon_engine::Player;

fn main() {
    let args = parse_args();
    let games = number(&args, "games", 20_u32);
    let seed = number(&args, "seed", 20_260_823_u32);
    let max_plies = number(&args, "max-plies", 180_u16);
    let opening_random_plies = number(&args, "opening-random-plies", 2_u16);
    let depth = number(&args, "depth", 4_u8);
    let max_nodes = number(&args, "nodes", 90_000_u64);
    let beam_width = number(&args, "beam", 40_usize);
    let opponent_name = args.get("opponent").map(String::as_str).unwrap_or("random");
    let jsonl = args.contains_key("jsonl");
    let corpus_directory = args.get("corpus").map(PathBuf::from);
    let book = corpus_directory.as_ref()
        .map(|directory| StrategyBook::load(&directory.join("positions.tsv")))
        .transpose()
        .unwrap_or_else(|error| fail(&format!("cannot load strategy book: {error}")))
        .map(Arc::new);

    let config = SearchConfig {
        depth,
        max_nodes,
        beam_width,
        weights: EvaluationWeights::default(),
    };
    let champion = with_optional_book(Agent::search("rust-pathfinder-v0.1.0", config), &book);
    let opponent = if opponent_name == "search" {
        with_optional_book(Agent::search(
            "rust-surveyor-v0.1.0",
            SearchConfig { depth: 2, max_nodes: 12_000, beam_width: 64, ..config },
        ), &book)
    } else {
        Agent::random("coin-flip-seeded")
    };

    let started = Instant::now();
    let mut wins = 0_u32;
    let mut losses = 0_u32;
    let mut draws = 0_u32;
    let mut total_plies = 0_u64;
    let mut total_nodes = 0_u64;
    let mut book_hits = 0_u64;
    let mut records = Vec::with_capacity(games as usize);
    for game in 0..games {
        let champion_is_light = game % 2 == 0;
        let record = play_game(
            if champion_is_light { &champion } else { &opponent },
            if champion_is_light { &opponent } else { &champion },
            MatchOptions {
                seed: seed.wrapping_add(game),
                max_plies,
                opening_random_plies,
            },
        );
        match record.winner {
            None => draws += 1,
            Some(winner) if winner == if champion_is_light { Player::Light } else { Player::Dark } => wins += 1,
            Some(_) => losses += 1,
        }
        total_plies += record.moves.len() as u64;
        total_nodes += record.moves.iter().map(|movement| movement.nodes).sum::<u64>();
        book_hits += record.moves.iter().filter(|movement| movement.book_hit).count() as u64;
        if jsonl {
            println!("{}", record.to_json());
        }
        records.push(record);
    }
    let corpus = corpus_directory.as_ref().map(|directory| {
        write_corpus(directory, &records).unwrap_or_else(|error| fail(&format!("cannot write corpus: {error}")))
    });
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "{{\"schemaVersion\":2,\"engine\":\"rust\",\"agent\":\"{}\",\"opponent\":\"{}\",\"seed\":{},\"games\":{},\"wins\":{},\"losses\":{},\"draws\":{},\"plies\":{},\"nodes\":{},\"bookHits\":{},\"corpusGames\":{},\"corpusPositions\":{},\"seconds\":{:.6},\"gamesPerSecond\":{:.3}}}",
        champion.id(),
        opponent.id(),
        seed,
        games,
        wins,
        losses,
        draws,
        total_plies,
        total_nodes,
        book_hits,
        corpus.map_or(0, |summary| summary.games),
        corpus.map_or(0, |summary| summary.positions),
        elapsed,
        if elapsed > 0.0 { games as f64 / elapsed } else { 0.0 },
    );
}

fn with_optional_book(agent: Agent, book: &Option<Arc<StrategyBook>>) -> Agent {
    book.as_ref().map_or_else(|| agent.clone(), |book| agent.with_book(Arc::clone(book)))
}

fn fail(message: &str) -> ! {
    eprintln!("pathagon-selfplay: {message}");
    std::process::exit(2)
}

fn parse_args() -> HashMap<String, String> {
    let mut parsed = HashMap::new();
    let values: Vec<String> = env::args().skip(1).collect();
    let mut index = 0;
    while index < values.len() {
        let value = &values[index];
        if let Some(option) = value.strip_prefix("--") {
            if let Some((key, inline)) = option.split_once('=') {
                parsed.insert(key.to_owned(), inline.to_owned());
            } else if values.get(index + 1).is_some_and(|next| !next.starts_with("--")) {
                parsed.insert(option.to_owned(), values[index + 1].clone());
                index += 1;
            } else {
                parsed.insert(option.to_owned(), "true".to_owned());
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
    args.get(key).and_then(|value| value.parse().ok()).unwrap_or(fallback)
}
