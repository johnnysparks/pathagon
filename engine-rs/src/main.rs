use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use pathagon_engine::corpus::{write_corpus, StrategyBook};
#[cfg(feature = "inference")]
use pathagon_engine::inference::{OnnxGnnPolicyValueModel, OnnxQAdvModel};
use pathagon_engine::learned::LearnedBook;
use pathagon_engine::pathfinder::{PathfinderConfig, PathfinderGuide};
#[cfg(feature = "inference")]
use pathagon_engine::puct::PuctConfig;
use pathagon_engine::search::{EvaluationWeights, SearchConfig};
use pathagon_engine::selfplay::{play_game, Agent, GameRecord, MatchOptions};
#[cfg(feature = "inference")]
use pathagon_engine::selfplay::{GnnPlayConfig, QAdvPlayConfig};
use pathagon_engine::Player;

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
    let temperature_moves = number(&args, "temperature-moves", 8_u16);
    let policy_temperature = number(&args, "policy-temperature", 1.0_f32);
    let opening_moves = number(&args, "opening-moves", 0_u16);
    let opening_temperature = number(&args, "opening-temperature", 1.0_f32);
    let opening_randomness = number(&args, "opening-randomness", 0.0_f32);
    let pathfinder_guidance = number(&args, "pathfinder-guidance", 0.0_f32);
    let placement_guidance = number(&args, "placement-guidance", pathfinder_guidance);
    let pathfinder_temperature = number(&args, "pathfinder-temperature", 1.0_f32);
    let pathfinder_depth = number(&args, "pathfinder-depth", 2_u8);
    let pathfinder_beam = number(&args, "pathfinder-beam", 8_usize);
    let pathfinder_nodes = number(&args, "pathfinder-nodes", 1_000_u64);
    let qadv_weight = number(&args, "qadv-weight", 1.0_f32);
    let qadv_tree_seeds = !args.contains_key("no-qadv-tree-seeds");
    let sorter_top_k = number(&args, "sorter-top-k", 4_usize);
    let sorter_all_actions = args.contains_key("sorter-all-actions");
    let sorter_root_limit = number(&args, "sorter-root-limit", 0_usize);
    let sorter_min_margin = number(&args, "sorter-min-margin", 0.0_f32);
    let sorter_max_heuristic_gap = number(&args, "sorter-max-heuristic-gap", 0_i32);
    let tactical_simulations = number(&args, "tactical-simulations", 512_u32);
    let tactical_proof_nodes = number(&args, "tactical-proof-nodes", 50_000_u64);
    let guided = args.contains_key("guided")
        || args.contains_key("pathfinder-guidance")
        || args.contains_key("placement-guidance")
        || args.contains_key("opening-moves")
        || args.contains_key("opening-temperature")
        || args.contains_key("opening-randomness")
        || args.contains_key("policy-temperature")
        || args.contains_key("temperature-moves");
    let tactical_proof_horizon = args
        .get("tactical-proof-horizon")
        .and_then(|value| value.parse().ok());
    let opponent_name = args.get("opponent").map(String::as_str).unwrap_or("random");
    let jsonl = args.contains_key("jsonl");
    let progress_every = number(&args, "progress-every", (games / 20).max(1));
    let workers = number(&args, "workers", 1_usize).max(1);
    let corpus_directory = args.get("corpus").map(PathBuf::from);
    let learned_book = args
        .get("learned")
        .map(PathBuf::from)
        .map(|path| LearnedBook::load(&path))
        .transpose()
        .unwrap_or_else(|error| fail(&format!("cannot load learned book: {error}")))
        .map(Arc::new);
    let learned_minimum_visits = number(&args, "learned-min-visits", 2_u32);
    let book = corpus_directory
        .as_ref()
        .map(|directory| StrategyBook::load(&directory.join("positions.tsv")))
        .transpose()
        .unwrap_or_else(|error| fail(&format!("cannot load strategy book: {error}")))
        .map(Arc::new);

    #[cfg(feature = "inference")]
    let neural_model = args.get("onnx").map(|path| {
        let bytes = fs::read(path)
            .unwrap_or_else(|error| fail(&format!("cannot read ONNX model: {error}")));
        Arc::new(
            OnnxGnnPolicyValueModel::from_bytes(&bytes)
                .unwrap_or_else(|error| fail(&format!("cannot load GNN ONNX model: {error}"))),
        )
    });
    #[cfg(feature = "inference")]
    let qadv_model = args.get("qadv-onnx").map(|path| {
        let bytes = fs::read(path)
            .unwrap_or_else(|error| fail(&format!("cannot read QAdv ONNX model: {error}")));
        Arc::new(
            OnnxQAdvModel::from_bytes(&bytes)
                .unwrap_or_else(|error| fail(&format!("cannot load QAdv ONNX model: {error}"))),
        )
    });
    #[cfg(feature = "inference")]
    let sorter_model = args.get("sorter-onnx").map(|path| {
        let bytes = fs::read(path)
            .unwrap_or_else(|error| fail(&format!("cannot read sorter ONNX model: {error}")));
        Arc::new(
            OnnxGnnPolicyValueModel::from_bytes(&bytes)
                .unwrap_or_else(|error| fail(&format!("cannot load sorter ONNX model: {error}"))),
        )
    });
    #[cfg(feature = "inference")]
    let qadv_sorter_model = args.get("sorter-qadv-onnx").map(|path| {
        let bytes = fs::read(path)
            .unwrap_or_else(|error| fail(&format!("cannot read QAdv sorter ONNX model: {error}")));
        Arc::new(
            OnnxQAdvModel::from_bytes(&bytes).unwrap_or_else(|error| {
                fail(&format!("cannot load QAdv sorter ONNX model: {error}"))
            }),
        )
    });
    #[cfg(not(feature = "inference"))]
    if args.contains_key("onnx") {
        fail("--onnx requires the inference feature; rebuild with --features inference");
    }
    #[cfg(not(feature = "inference"))]
    if args.contains_key("qadv-onnx") {
        fail("--qadv-onnx requires the inference feature; rebuild with --features inference");
    }
    #[cfg(not(feature = "inference"))]
    if args.contains_key("sorter-onnx") {
        fail("--sorter-onnx requires the inference feature; rebuild with --features inference");
    }
    #[cfg(not(feature = "inference"))]
    if args.contains_key("sorter-qadv-onnx") {
        fail(
            "--sorter-qadv-onnx requires the inference feature; rebuild with --features inference",
        );
    }

    #[cfg(feature = "inference")]
    if args.contains_key("eval-only") {
        let state = evaluation_state(&args);
        let count = number(&args, "eval-count", 8_usize);
        if let Some(model) = qadv_model.as_ref() {
            let output = model
                .evaluate_qadv(state)
                .unwrap_or_else(|error| fail(&format!("QAdv evaluation failed: {error}")));
            println!(
                "{{\"legalActions\":{},\"policyFirst\":{},\"value\":{:.9},\"qFirst\":{}}}",
                state.legal_actions().len(),
                json_f32_prefix(&output.policy_logits, count),
                output.value,
                json_f32_prefix(&output.q_values, count),
            );
        } else if let Some(model) = neural_model.as_ref() {
            let output =
                pathagon_engine::inference::PolicyValueModel::evaluate(model.as_ref(), state)
                    .unwrap_or_else(|error| fail(&format!("GNN evaluation failed: {error}")));
            println!(
                "{{\"legalActions\":{},\"policyFirst\":{},\"value\":{:.9}}}",
                state.legal_actions().len(),
                json_f32_prefix(&output.policy_logits, count),
                output.value,
            );
        } else {
            fail("--eval-only requires --onnx or --qadv-onnx");
        }
        return;
    }

    if args.contains_key("pathfinder-only") {
        let state = evaluation_state(&args);
        let actions = state.legal_actions();
        let mut guide = PathfinderGuide::new(PathfinderConfig {
            depth: pathfinder_depth,
            beam_width: pathfinder_beam,
            max_nodes: pathfinder_nodes,
        })
        .unwrap_or_else(|error| fail(&format!("invalid Pathfinder config: {error}")));
        let scores = guide.score_actions(state, &actions);
        println!(
            "{{\"legalActions\":{},\"scoresFirst\":{}}}",
            actions.len(),
            json_f32_prefix(&scores, number(&args, "eval-count", 8_usize)),
        );
        return;
    }

    if args.contains_key("heuristic-only") {
        let state = evaluation_state(&args);
        let perspective = match args
            .get("perspective")
            .map(String::as_str)
            .unwrap_or("turn")
        {
            "light" => Player::Light,
            "dark" => Player::Dark,
            _ => state.turn,
        };
        let score =
            pathagon_engine::search::evaluate(state, perspective, EvaluationWeights::default());
        println!(
            "{{\"perspective\":\"{}\",\"score\":{score}}}",
            perspective.as_str()
        );
        return;
    }

    let config = SearchConfig {
        depth,
        max_nodes,
        beam_width,
        weights: EvaluationWeights::default(),
        tactical_proof_horizon,
    };
    #[cfg(feature = "inference")]
    let champion = if let Some(model) = qadv_sorter_model.as_ref() {
        Agent::qadv_sorter_with_pool(
            "rust-pathfinder-onnx-qadv-sorter-v0.1.0",
            config,
            sorter_top_k,
            sorter_all_actions,
            sorter_root_limit,
            sorter_min_margin,
            sorter_max_heuristic_gap,
            Arc::clone(model),
        )
    } else if let Some(model) = sorter_model.as_ref() {
        Agent::gnn_sorter_with_pool(
            "rust-pathfinder-onnx-sorter-v0.1.0",
            config,
            sorter_top_k,
            sorter_all_actions,
            sorter_root_limit,
            sorter_min_margin,
            sorter_max_heuristic_gap,
            Arc::clone(model),
        )
    } else if let Some(model) = qadv_model.as_ref() {
        Agent::qadv(
            "qadv-arbiter-7x7-rust-qadv-v0.1.0",
            QAdvPlayConfig {
                guided: GnnPlayConfig {
                    puct: PuctConfig {
                        simulations,
                        cpuct,
                        use_action_value_seeds: qadv_tree_seeds,
                    },
                    temperature_moves,
                    policy_temperature,
                    opening_moves,
                    opening_temperature,
                    opening_randomness,
                    pathfinder_guidance,
                    placement_guidance,
                    pathfinder_temperature,
                    pathfinder_depth,
                    pathfinder_beam,
                    pathfinder_nodes,
                },
                qadv_weight,
                tactical_simulations,
                tactical_capture_threshold: 2,
                tactical_proof_horizon,
                tactical_proof_nodes,
            },
            Arc::clone(model),
        )
    } else if let Some(model) = neural_model.as_ref() {
        if guided {
            Agent::gnn_guided(
                "qadv-arbiter-7x7-rust-policy-v0.1.0",
                GnnPlayConfig {
                    puct: PuctConfig {
                        simulations,
                        cpuct,
                        use_action_value_seeds: false,
                    },
                    temperature_moves,
                    policy_temperature,
                    opening_moves,
                    opening_temperature,
                    opening_randomness,
                    pathfinder_guidance,
                    placement_guidance,
                    pathfinder_temperature,
                    pathfinder_depth,
                    pathfinder_beam,
                    pathfinder_nodes,
                },
                Arc::clone(model),
            )
        } else {
            Agent::gnn(
                "qadv-arbiter-7x7-rust-policy-v0.1.0",
                PuctConfig {
                    simulations,
                    cpuct,
                    use_action_value_seeds: false,
                },
                Arc::clone(model),
            )
        }
    } else {
        learned_book.as_ref().map_or_else(
            || with_optional_book(Agent::search("rust-pathfinder-v0.1.0", config), &book),
            |book| {
                Agent::learned(
                    "rust-learned-tabular-v0.1.0",
                    config,
                    Arc::clone(book),
                    learned_minimum_visits,
                )
            },
        )
    };
    #[cfg(not(feature = "inference"))]
    let champion = learned_book.as_ref().map_or_else(
        || with_optional_book(Agent::search("rust-pathfinder-v0.1.0", config), &book),
        |book| {
            Agent::learned(
                "rust-learned-tabular-v0.1.0",
                config,
                Arc::clone(book),
                learned_minimum_visits,
            )
        },
    );
    let opponent = if opponent_name == "neural" {
        #[cfg(feature = "inference")]
        {
            if let Some(model) = qadv_model.as_ref() {
                Agent::qadv(
                    "qadv-arbiter-7x7-rust-qadv-v0.1.0",
                    QAdvPlayConfig {
                        guided: GnnPlayConfig {
                            puct: PuctConfig {
                                simulations,
                                cpuct,
                                use_action_value_seeds: qadv_tree_seeds,
                            },
                            temperature_moves,
                            policy_temperature,
                            opening_moves,
                            opening_temperature,
                            opening_randomness,
                            pathfinder_guidance,
                            placement_guidance,
                            pathfinder_temperature,
                            pathfinder_depth,
                            pathfinder_beam,
                            pathfinder_nodes,
                        },
                        qadv_weight,
                        tactical_simulations,
                        tactical_capture_threshold: 2,
                        tactical_proof_horizon,
                        tactical_proof_nodes,
                    },
                    Arc::clone(model),
                )
            } else {
                let model = neural_model.as_ref().unwrap_or_else(|| {
                    fail("--opponent neural requires --onnx <model> or --qadv-onnx <model>")
                });
                if guided {
                    Agent::gnn_guided(
                        "qadv-arbiter-7x7-rust-policy-v0.1.0",
                        GnnPlayConfig {
                            puct: PuctConfig {
                                simulations,
                                cpuct,
                                use_action_value_seeds: false,
                            },
                            temperature_moves,
                            policy_temperature,
                            opening_moves,
                            opening_temperature,
                            opening_randomness,
                            pathfinder_guidance,
                            placement_guidance,
                            pathfinder_temperature,
                            pathfinder_depth,
                            pathfinder_beam,
                            pathfinder_nodes,
                        },
                        Arc::clone(model),
                    )
                } else {
                    Agent::gnn(
                        "qadv-arbiter-7x7-rust-policy-v0.1.0",
                        PuctConfig {
                            simulations,
                            cpuct,
                            use_action_value_seeds: false,
                        },
                        Arc::clone(model),
                    )
                }
            }
        }
        #[cfg(not(feature = "inference"))]
        fail("--opponent neural requires the inference feature; rebuild with --features inference")
    } else if opponent_name == "qadv-sorter" {
        #[cfg(feature = "inference")]
        {
            let model = qadv_sorter_model.as_ref().unwrap_or_else(|| {
                fail("--opponent qadv-sorter requires --sorter-qadv-onnx <model>")
            });
            Agent::qadv_sorter_with_pool(
                "rust-pathfinder-onnx-qadv-sorter-v0.1.0",
                config,
                sorter_top_k,
                sorter_all_actions,
                sorter_root_limit,
                sorter_min_margin,
                sorter_max_heuristic_gap,
                Arc::clone(model),
            )
        }
        #[cfg(not(feature = "inference"))]
        fail("--opponent qadv-sorter requires the inference feature; rebuild with --features inference")
    } else if opponent_name == "sorter" {
        #[cfg(feature = "inference")]
        {
            let model = sorter_model
                .as_ref()
                .unwrap_or_else(|| fail("--opponent sorter requires --sorter-onnx <model>"));
            Agent::gnn_sorter_with_pool(
                "rust-pathfinder-onnx-sorter-v0.1.0",
                config,
                sorter_top_k,
                sorter_all_actions,
                sorter_root_limit,
                sorter_min_margin,
                sorter_max_heuristic_gap,
                Arc::clone(model),
            )
        }
        #[cfg(not(feature = "inference"))]
        fail("--opponent sorter requires the inference feature; rebuild with --features inference")
    } else if opponent_name == "deep-search" || opponent_name == "pathfinder" {
        with_optional_book(Agent::search("rust-pathfinder-v0.3.0", config), &book)
    } else if opponent_name == "search" {
        with_optional_book(
            Agent::search(
                "rust-surveyor-v0.1.0",
                SearchConfig {
                    depth: 2,
                    max_nodes: 12_000,
                    beam_width: 64,
                    ..config
                },
            ),
            &book,
        )
    } else if opponent_name == "learned" {
        let book = learned_book
            .as_ref()
            .unwrap_or_else(|| fail("--opponent learned requires --learned <learned.tsv>"));
        Agent::learned(
            "rust-learned-tabular-v0.1.0",
            SearchConfig {
                depth: 2,
                max_nodes: 12_000,
                beam_width: 64,
                ..config
            },
            Arc::clone(book),
            learned_minimum_visits,
        )
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
            .map(|game| {
                (
                    game,
                    play_index(
                        &champion,
                        &opponent,
                        game,
                        seed,
                        max_plies,
                        opening_random_plies,
                        board_size,
                        reserve_per_player,
                    ),
                )
            })
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
                            .map(|game| {
                                (
                                    game,
                                    play_index(
                                        &champion,
                                        &opponent,
                                        game,
                                        seed,
                                        max_plies,
                                        opening_random_plies,
                                        board_size,
                                        reserve_per_player,
                                    ),
                                )
                            })
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
            Some(winner)
                if winner
                    == if champion_is_light {
                        Player::Light
                    } else {
                        Player::Dark
                    } =>
            {
                wins += 1
            }
            Some(_) => losses += 1,
        }
        total_plies += record.moves.len() as u64;
        total_nodes += record
            .moves
            .iter()
            .map(|movement| movement.nodes)
            .sum::<u64>();
        book_hits += record
            .moves
            .iter()
            .filter(|movement| movement.book_hit)
            .count() as u64;
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
        write_corpus(directory, &records)
            .unwrap_or_else(|error| fail(&format!("cannot write corpus: {error}")))
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
        if champion_is_light {
            champion
        } else {
            opponent
        },
        if champion_is_light {
            opponent
        } else {
            champion
        },
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

fn evaluation_state(args: &HashMap<String, String>) -> pathagon_engine::GameState {
    let board_size = number(args, "board-size", 7_u8);
    let reserve = number(args, "reserve", board_size.saturating_mul(2));
    let max_plies = number(args, "max-plies", 196_u16);
    let config = pathagon_engine::BoardConfig::new(board_size, reserve)
        .and_then(|config| config.with_max_plies(max_plies))
        .unwrap_or_else(|error| fail(&format!("invalid evaluation board: {error}")));
    let mut state = pathagon_engine::GameState::with_config(config);
    if let Some(sequence) = args.get("eval-sequence") {
        for token in sequence.split(',').filter(|token| !token.is_empty()) {
            let action = if let Some(to) = token.strip_prefix('P') {
                pathagon_engine::Action::Place {
                    to: to.parse().unwrap_or_else(|_| {
                        fail(&format!("invalid placement in --eval-sequence: {token}"))
                    }),
                }
            } else if let Some((from, to)) = token
                .strip_prefix('R')
                .and_then(|token| token.split_once('>'))
            {
                pathagon_engine::Action::Relocate {
                    from: from.parse().unwrap_or_else(|_| {
                        fail(&format!("invalid relocation in --eval-sequence: {token}"))
                    }),
                    to: to.parse().unwrap_or_else(|_| {
                        fail(&format!("invalid relocation in --eval-sequence: {token}"))
                    }),
                }
            } else {
                fail(&format!("invalid action in --eval-sequence: {token}"));
            };
            if !state.legal_actions().contains(&action) {
                fail(&format!("illegal action in --eval-sequence: {token}"));
            }
            state = state.apply_legal(action).state;
        }
    }
    state
}

fn json_f32_prefix(values: &[f32], count: usize) -> String {
    let values = values.iter().take(count).map(|value| format!("{value:.9}"));
    format!("[{}]", values.collect::<Vec<_>>().join(","))
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
            } else if values
                .get(index + 1)
                .is_some_and(|next| !next.starts_with("--"))
            {
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
    args.get(key)
        .and_then(|value| value.parse().ok())
        .unwrap_or(fallback)
}
