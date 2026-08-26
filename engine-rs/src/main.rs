use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use pathagon_engine::corpus::{write_corpus, StrategyBook};
use pathagon_engine::learned::LearnedBook;
use pathagon_engine::search::{EvaluationWeights, SearchConfig};
use pathagon_engine::selfplay::{play_game, Agent, GameRecord, MatchOptions};
use pathagon_engine::Player;
#[cfg(feature = "inference")]
use pathagon_engine::inference::OnnxGnnPolicyValueModel;
#[cfg(feature = "inference")]
use pathagon_engine::puct::PuctConfig;

fn main() {
    let args = parse_args();
    let games = number(&args, "games", 20_u32);
    let seed = number(&args, "seed", 20_260_823_u32);
    let max_plies = number(&args, "max-plies", 180_u16);
    let opening_random_plies = number(&args, "opening-random-plies", 2_u16);
    let board_size = number(&args, "board-size", 7_u8);
    let reserve_per_player = number(&args, "reserve", board_size.saturating_mul(2));
    let depth = number(&args, "depth", 4_u8);
    let max_nodes = number(&args, "nodes", 90_000_u64);
    let beam_width = number(&args, "beam", 40_usize);
    let simulations = number(&args, "simulations", 64_u32);
    let cpuct = number(&args, "cpuct", 1.5_f32);
    let tactical_proof_horizon = args
        .get("tactical-proof-horizon")
        .and_then(|value| value.parse().ok());
    let opponent_name = args.get("opponent").map(String::as_str).unwrap_or("random");
    let jsonl = args.contains_key("jsonl");
    let progress_every = number(&args, "progress-every", (games / 20).max(1));
    let workers = number(&args, "workers", 1_usize).max(1);
    let corpus_directory = args.get("corpus").map(PathBuf::from);
    let learned_book = args.get("learned").map(PathBuf::from)
        .map(|path| LearnedBook::load(&path))
        .transpose()
        .unwrap_or_else(|error| fail(&format!("cannot load learned book: {error}")))
        .map(Arc::new);
    let learned_minimum_visits = number(&args, "learned-min-visits", 2_u32);
    let book = corpus_directory.as_ref()
        .map(|directory| StrategyBook::load(&directory.join("positions.tsv")))
        .transpose()
        .unwrap_or_else(|error| fail(&format!("cannot load strategy book: {error}")))
        .map(Arc::new);

    #[cfg(feature = "inference")]
    let neural_model = args.get("onnx").map(|path| {
        let bytes = fs::read(path).unwrap_or_else(|error| fail(&format!("cannot read ONNX model: {error}")));
        Arc::new(OnnxGnnPolicyValueModel::from_bytes(&bytes).unwrap_or_else(|error| fail(&format!("cannot load GNN ONNX model: {error}"))))
    });
    #[cfg(not(feature = "inference"))]
    if args.contains_key("onnx") {
        fail("--onnx requires the inference feature; rebuild with --features inference");
    }

    let config = SearchConfig {
        depth,
        max_nodes,
        beam_width,
        weights: EvaluationWeights::default(),
        tactical_proof_horizon,
    };
    #[cfg(feature = "inference")]
    let champion = if let Some(model) = neural_model.as_ref() {
        Agent::gnn(
            "qadv-arbiter-7x7-rust-policy-v0.1.0",
            PuctConfig { simulations, cpuct },
            Arc::clone(model),
        )
    } else {
        learned_book.as_ref().map_or_else(
            || with_optional_book(Agent::search("rust-pathfinder-v0.1.0", config), &book),
            |book| Agent::learned("rust-learned-tabular-v0.1.0", config, Arc::clone(book), learned_minimum_visits),
        )
    };
    #[cfg(not(feature = "inference"))]
    let champion = learned_book.as_ref().map_or_else(
        || with_optional_book(Agent::search("rust-pathfinder-v0.1.0", config), &book),
        |book| Agent::learned("rust-learned-tabular-v0.1.0", config, Arc::clone(book), learned_minimum_visits),
    );
    let opponent = if opponent_name == "neural" {
        #[cfg(feature = "inference")]
        {
            let model = neural_model.as_ref().unwrap_or_else(|| fail("--opponent neural requires --onnx <model>"));
            Agent::gnn(
                "qadv-arbiter-7x7-rust-policy-v0.1.0",
                PuctConfig { simulations, cpuct },
                Arc::clone(model),
            )
        }
        #[cfg(not(feature = "inference"))]
        fail("--opponent neural requires the inference feature; rebuild with --features inference")
    } else if opponent_name == "search" {
        with_optional_book(Agent::search(
            "rust-surveyor-v0.1.0",
            SearchConfig { depth: 2, max_nodes: 12_000, beam_width: 64, ..config },
        ), &book)
    } else if opponent_name == "learned" {
        let book = learned_book.as_ref().unwrap_or_else(|| fail("--opponent learned requires --learned <learned.tsv>"));
        Agent::learned("rust-learned-tabular-v0.1.0", SearchConfig { depth: 2, max_nodes: 12_000, beam_width: 64, ..config }, Arc::clone(book), learned_minimum_visits)
    } else if opponent_name == "lunatic" {
        Agent::lunatic("lunatic-v0.1.0")
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
    let worker_count = workers.min(games.max(1) as usize);
    let mut indexed_records: Vec<(u32, GameRecord)> = if worker_count == 1 {
        (0..games)
            .map(|game| (game, play_index(&champion, &opponent, game, seed, max_plies, opening_random_plies, board_size, reserve_per_player)))
            .collect()
    } else {
        std::thread::scope(|scope| {
            let handles = (0..worker_count)
                .map(|worker| {
                    let champion = champion.clone();
                    let opponent = opponent.clone();
                    scope.spawn(move || {
                        (worker as u32..games)
                            .step_by(worker_count)
                            .map(|game| (game, play_index(&champion, &opponent, game, seed, max_plies, opening_random_plies, board_size, reserve_per_player)))
                            .collect::<Vec<_>>()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .flat_map(|handle| handle.join().expect("Rust self-play worker panicked"))
                .collect()
        })
    };
    indexed_records.sort_by_key(|(game, _record)| *game);
    for (game, record) in indexed_records {
        let champion_is_light = game % 2 == 0;
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
        let completed = game + 1;
        if progress_every > 0 && (completed % progress_every == 0 || completed == games) {
            let elapsed = started.elapsed().as_secs_f64();
            eprintln!(
                "pathagon-selfplay: {completed}/{games} games ({:.0}%) wins={wins} losses={losses} draws={draws} elapsed={elapsed:.1}s games_per_second={:.3}",
                completed as f64 / games as f64 * 100.0,
                if elapsed > 0.0 { completed as f64 / elapsed } else { 0.0 },
            );
        }
    }
    let corpus = corpus_directory.as_ref().map(|directory| {
        write_corpus(directory, &records).unwrap_or_else(|error| fail(&format!("cannot write corpus: {error}")))
    });
    let elapsed = started.elapsed().as_secs_f64();
    let summary = format!(
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
    if jsonl {
        eprintln!("{summary}");
    } else {
        println!("{summary}");
    }
}

fn play_index(
    champion: &Agent,
    opponent: &Agent,
    game: u32,
    seed: u32,
    max_plies: u16,
    opening_random_plies: u16,
    board_size: u8,
    reserve_per_player: u8,
) -> GameRecord {
    let champion_is_light = game % 2 == 0;
    play_game(
        if champion_is_light { champion } else { opponent },
        if champion_is_light { opponent } else { champion },
        MatchOptions {
            seed: seed.wrapping_add(game),
            max_plies,
            opening_random_plies,
            board_size,
            reserve_per_player,
        },
    )
}

fn with_optional_book(agent: Agent, book: &Option<Arc<StrategyBook>>) -> Agent {
    if let Some(book) = book {
        agent.with_book(Arc::clone(book))
    } else {
        agent
    }
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
