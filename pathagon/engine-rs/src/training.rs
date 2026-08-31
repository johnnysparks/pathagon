//! Deterministic evaluator evolution with held-out promotion matches.

use std::fs;
use std::io;
use std::path::Path;

use serde::Deserialize;

use crate::corpus::{write_corpus, CorpusSummary};
use crate::search::{EvaluationWeights, SearchConfig};
use crate::selfplay::{play_game, Agent, GameRecord, MatchOptions, Mulberry32};
use crate::Player;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrainingConfig {
    pub generations: u8,
    pub population: u8,
    pub training_pairs: u16,
    pub evaluation_pairs: u16,
    pub seed: u32,
    pub mutation_per_mille: u16,
    pub promotion_rate_per_mille: u16,
    pub max_plies: u16,
    pub opening_random_plies: u16,
    pub tactical_filter: bool,
    pub search: SearchConfig,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            generations: 3,
            population: 6,
            training_pairs: 6,
            evaluation_pairs: 12,
            seed: 20_260_823,
            mutation_per_mille: 200,
            promotion_rate_per_mille: 550,
            max_plies: 120,
            opening_random_plies: 4,
            tactical_filter: false,
            search: SearchConfig {
                depth: 2,
                max_nodes: 12_000,
                beam_width: 49,
                weights: EvaluationWeights::default(),
                tactical_proof_horizon: None,
            },
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Champion {
    pub id: String,
    pub generation: u8,
    pub weights: EvaluationWeights,
}

impl Champion {
    pub fn baseline(weights: EvaluationWeights) -> Self {
        Self {
            id: "rust-handcrafted-g0".to_owned(),
            generation: 0,
            weights,
        }
    }

    pub fn from_manifest_file(path: &Path) -> io::Result<Self> {
        let contents = fs::read_to_string(path)?;
        Self::from_manifest_json(&contents)
    }

    pub fn from_manifest_json(contents: &str) -> io::Result<Self> {
        let document: InitialChampionDocument =
            serde_json::from_str(contents).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid evaluator manifest: {error}"),
                )
            })?;
        let weights = document
            .weights
            .or(document.evaluator_weights)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "evaluator manifest is missing weights or evaluatorWeights",
                )
            })?;
        Ok(Self {
            id: document
                .id
                .unwrap_or_else(|| "rust-initial-evaluator".to_owned()),
            generation: document.generation.unwrap_or(0),
            weights,
        })
    }
}

#[derive(Deserialize)]
struct InitialChampionDocument {
    id: Option<String>,
    generation: Option<u8>,
    weights: Option<EvaluationWeights>,
    #[serde(rename = "evaluatorWeights")]
    evaluator_weights: Option<EvaluationWeights>,
}

pub fn parse_weights_spec(spec: &str) -> Result<EvaluationWeights, String> {
    if spec.trim_start().starts_with('{') {
        return serde_json::from_str(spec)
            .map_err(|error| format!("invalid JSON weights: {error}"));
    }

    let mut weights = [None; 6];
    for assignment in spec.split(',').filter(|part| !part.trim().is_empty()) {
        let (key, value) = assignment.split_once('=').ok_or_else(|| {
            format!("invalid weight assignment {assignment:?}; expected key=value")
        })?;
        let index = match key.trim() {
            "path" => 0,
            "material" => 1,
            "capture" => 2,
            "structure" => 3,
            "threat" => 4,
            "edge" => 5,
            other => return Err(format!("unknown evaluator weight {other:?}")),
        };
        let value = value
            .trim()
            .parse::<i32>()
            .map_err(|error| format!("invalid value for {key:?}: {error}"))?;
        if value <= 0 {
            return Err(format!("evaluator weight {key:?} must be positive"));
        }
        weights[index] = Some(value);
    }

    let [path, material, capture, structure, threat, edge] = weights;
    Ok(EvaluationWeights {
        path: path.ok_or("missing path weight")?,
        material: material.ok_or("missing material weight")?,
        capture: capture.ok_or("missing capture weight")?,
        structure: structure.ok_or("missing structure weight")?,
        threat: threat.ok_or("missing threat weight")?,
        edge: edge.ok_or("missing edge weight")?,
    })
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MatchScore {
    pub games: u32,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
}

impl MatchScore {
    pub const fn net(self) -> i32 {
        self.wins as i32 - self.losses as i32
    }

    pub const fn points_rate_per_mille(self) -> u32 {
        if self.games == 0 {
            0
        } else {
            (self.wins * 2 + self.draws) * 1_000 / (self.games * 2)
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CandidateTrial {
    pub id: String,
    pub generation: u8,
    pub weights: EvaluationWeights,
    pub training: MatchScore,
    pub evaluation: Option<MatchScore>,
    pub promoted: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainingResult {
    pub config: TrainingConfig,
    pub initial: Champion,
    pub champion: Champion,
    pub trials: Vec<CandidateTrial>,
    pub training_records: Vec<GameRecord>,
    pub evaluation_records: Vec<GameRecord>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TrainingOutput {
    pub training: CorpusSummary,
    pub evaluation: CorpusSummary,
}

pub fn train(initial: Champion, config: TrainingConfig) -> TrainingResult {
    let mut random = Mulberry32::new(config.seed);
    let mut champion = initial.clone();
    let mut trials = Vec::new();
    let mut training_records = Vec::new();
    let mut evaluation_records = Vec::new();

    for generation_offset in 0..config.generations {
        let generation = initial
            .generation
            .saturating_add(generation_offset)
            .saturating_add(1);
        let incumbent = champion.clone();
        let mut generation_trials = Vec::new();
        for candidate_index in 0..config.population {
            let weights = mutate_weights(incumbent.weights, &mut random, config.mutation_per_mille);
            let id = candidate_id(generation, candidate_index, weights);
            let records = paired_series(
                &id,
                weights,
                &incumbent,
                config,
                config.seed.wrapping_add(generation_offset as u32 * 100_000),
                config.training_pairs,
            );
            let training = summarize(&records, &id);
            training_records.extend(records);
            generation_trials.push(CandidateTrial {
                id,
                generation,
                weights,
                training,
                evaluation: None,
                promoted: false,
            });
        }

        let best_index = generation_trials
            .iter()
            .enumerate()
            .max_by(|left, right| {
                left.1
                    .training
                    .net()
                    .cmp(&right.1.training.net())
                    .then_with(|| {
                        left.1
                            .training
                            .points_rate_per_mille()
                            .cmp(&right.1.training.points_rate_per_mille())
                    })
                    .then_with(|| right.1.id.cmp(&left.1.id))
            })
            .map(|(index, _)| index);

        if let Some(best_index) = best_index {
            let candidate = &generation_trials[best_index];
            let records = paired_series(
                &candidate.id,
                candidate.weights,
                &incumbent,
                config,
                config
                    .seed
                    .wrapping_add(0xA5A5_0000)
                    .wrapping_add(generation_offset as u32 * 100_000),
                config.evaluation_pairs,
            );
            let evaluation = summarize(&records, &candidate.id);
            evaluation_records.extend(records);
            let promoted = candidate.training.net() > 0
                && evaluation.net() > 0
                && evaluation.points_rate_per_mille() >= config.promotion_rate_per_mille as u32;
            generation_trials[best_index].evaluation = Some(evaluation);
            generation_trials[best_index].promoted = promoted;
            if promoted {
                champion = Champion {
                    id: generation_trials[best_index].id.clone(),
                    generation,
                    weights: generation_trials[best_index].weights,
                };
            }
        }
        trials.extend(generation_trials);
    }

    TrainingResult {
        config,
        initial,
        champion,
        trials,
        training_records,
        evaluation_records,
    }
}

pub fn write_training_output(
    directory: &Path,
    result: &TrainingResult,
) -> io::Result<TrainingOutput> {
    fs::create_dir_all(directory)?;
    let training = write_corpus(&directory.join("corpus/training"), &result.training_records)?;
    let evaluation = write_corpus(
        &directory.join("corpus/evaluation"),
        &result.evaluation_records,
    )?;
    fs::write(
        directory.join("champion.json"),
        champion_json(&result.champion),
    )?;
    fs::write(directory.join("report.json"), result.to_json())?;
    Ok(TrainingOutput {
        training,
        evaluation,
    })
}

impl TrainingResult {
    pub fn to_json(&self) -> String {
        let trials = self
            .trials
            .iter()
            .map(trial_json)
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "{{\"schemaVersion\":1,\"seed\":{},\"config\":{{\"generations\":{},\"population\":{},\"trainingPairs\":{},\"evaluationPairs\":{},\"mutationPerMille\":{},\"promotionRatePerMille\":{},\"maxPlies\":{},\"openingRandomPlies\":{},\"tacticalFilter\":{},\"search\":{}}},\"initial\":{},\"champion\":{},\"promotions\":{},\"trials\":[{}]}}\n",
            self.config.seed,
            self.config.generations,
            self.config.population,
            self.config.training_pairs,
            self.config.evaluation_pairs,
            self.config.mutation_per_mille,
            self.config.promotion_rate_per_mille,
            self.config.max_plies,
            self.config.opening_random_plies,
            self.config.tactical_filter,
            search_json(self.config.search),
            champion_json(&self.initial).trim(),
            champion_json(&self.champion).trim(),
            self.trials.iter().filter(|trial| trial.promoted).count(),
            trials,
        )
    }
}

fn paired_series(
    candidate_id: &str,
    candidate_weights: EvaluationWeights,
    incumbent: &Champion,
    config: TrainingConfig,
    seed: u32,
    pairs: u16,
) -> Vec<GameRecord> {
    let candidate_config = SearchConfig {
        weights: candidate_weights,
        ..config.search
    };
    let incumbent_config = SearchConfig {
        weights: incumbent.weights,
        ..config.search
    };
    let candidate = if config.tactical_filter {
        Agent::search_tactical_filter(candidate_id, candidate_config)
    } else {
        Agent::search(candidate_id, candidate_config)
    };
    let incumbent_agent = if config.tactical_filter {
        Agent::search_tactical_filter(&incumbent.id, incumbent_config)
    } else {
        Agent::search(&incumbent.id, incumbent_config)
    };
    let mut records = Vec::with_capacity(pairs as usize * 2);
    for pair in 0..pairs {
        let options = MatchOptions {
            seed: seed.wrapping_add(pair as u32),
            max_plies: config.max_plies,
            opening_random_plies: config.opening_random_plies,
            ..MatchOptions::default()
        };
        records.push(play_game(&candidate, &incumbent_agent, options));
        records.push(play_game(&incumbent_agent, &candidate, options));
    }
    records
}

fn summarize(records: &[GameRecord], candidate_id: &str) -> MatchScore {
    let mut score = MatchScore {
        games: records.len() as u32,
        ..MatchScore::default()
    };
    for record in records {
        let candidate_player = if record.light_agent == candidate_id {
            Player::Light
        } else {
            Player::Dark
        };
        match record.winner {
            Some(winner) if winner == candidate_player => score.wins += 1,
            Some(_) => score.losses += 1,
            None => score.draws += 1,
        }
    }
    score
}

fn mutate_weights(
    weights: EvaluationWeights,
    random: &mut Mulberry32,
    scale: u16,
) -> EvaluationWeights {
    EvaluationWeights {
        path: mutate_weight(weights.path, random, scale),
        material: mutate_weight(weights.material, random, scale),
        capture: mutate_weight(weights.capture, random, scale),
        structure: mutate_weight(weights.structure, random, scale),
        threat: mutate_weight(weights.threat, random, scale),
        edge: mutate_weight(weights.edge, random, scale),
    }
}

fn mutate_weight(value: i32, random: &mut Mulberry32, scale: u16) -> i32 {
    let centered = (random.next_u32() % 2_001) as i64 - 1_000;
    let multiplier = 1_000_000_i64 + centered * scale as i64;
    ((value as i64 * multiplier + 500_000) / 1_000_000).max(1) as i32
}

fn candidate_id(generation: u8, index: u8, weights: EvaluationWeights) -> String {
    format!(
        "rust-evo-g{generation}-c{index}-{}-{}-{}-{}-{}-{}",
        weights.path,
        weights.material,
        weights.capture,
        weights.structure,
        weights.threat,
        weights.edge,
    )
}

fn score_json(score: MatchScore) -> String {
    format!(
        "{{\"games\":{},\"wins\":{},\"losses\":{},\"draws\":{},\"pointsRatePerMille\":{}}}",
        score.games,
        score.wins,
        score.losses,
        score.draws,
        score.points_rate_per_mille(),
    )
}

fn weights_json(weights: EvaluationWeights) -> String {
    format!(
        "{{\"path\":{},\"material\":{},\"capture\":{},\"structure\":{},\"threat\":{},\"edge\":{}}}",
        weights.path,
        weights.material,
        weights.capture,
        weights.structure,
        weights.threat,
        weights.edge,
    )
}

fn search_json(search: SearchConfig) -> String {
    let tactical_proof_horizon = search
        .tactical_proof_horizon
        .map_or_else(|| "null".to_owned(), |horizon| horizon.to_string());
    format!(
        "{{\"depth\":{},\"maxNodes\":{},\"beamWidth\":{},\"weights\":{},\"tacticalProofHorizon\":{}}}",
        search.depth,
        search.max_nodes,
        search.beam_width,
        weights_json(search.weights),
        tactical_proof_horizon,
    )
}

fn champion_json(champion: &Champion) -> String {
    format!(
        "{{\"schemaVersion\":1,\"id\":\"{}\",\"generation\":{},\"weights\":{}}}\n",
        champion.id,
        champion.generation,
        weights_json(champion.weights),
    )
}

fn trial_json(trial: &CandidateTrial) -> String {
    let evaluation = trial.evaluation.map_or("null".to_owned(), score_json);
    format!(
        "{{\"id\":\"{}\",\"generation\":{},\"weights\":{},\"training\":{},\"evaluation\":{},\"promoted\":{}}}",
        trial.id,
        trial.generation,
        weights_json(trial.weights),
        score_json(trial.training),
        evaluation,
        trial.promoted,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static TEMP_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn temp_path(label: &str) -> std::path::PathBuf {
        let number = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "pathagon-training-{label}-{}-{number}",
            std::process::id()
        ))
    }

    #[test]
    fn training_is_reproducible_and_uses_held_out_games() {
        let config = TrainingConfig {
            generations: 1,
            population: 2,
            training_pairs: 1,
            evaluation_pairs: 1,
            max_plies: 50,
            opening_random_plies: 4,
            search: SearchConfig {
                depth: 1,
                max_nodes: 500,
                beam_width: 20,
                ..SearchConfig::default()
            },
            ..TrainingConfig::default()
        };
        let initial = Champion::baseline(config.search.weights);
        let first = train(initial.clone(), config);
        let second = train(initial, config);
        assert_eq!(first, second);
        assert_eq!(first.trials.len(), 2);
        assert_eq!(first.training_records.len(), 4);
        assert_eq!(first.evaluation_records.len(), 2);
        assert_eq!(
            first
                .trials
                .iter()
                .filter(|trial| trial.evaluation.is_some())
                .count(),
            1
        );
    }

    #[test]
    fn initial_manifest_supports_opponent_and_training_shapes() {
        let opponent = Champion::from_manifest_json(
            r#"{"id":"pathfinder-v0.5.0-trained-evaluator","generation":2,"evaluatorWeights":{"path":241,"material":112,"capture":887,"structure":40,"threat":154,"edge":74}}"#,
        )
        .expect("supported opponent manifest parses");
        assert_eq!(opponent.id, "pathfinder-v0.5.0-trained-evaluator");
        assert_eq!(opponent.generation, 2);
        assert_eq!(opponent.weights.capture, 887);

        let training = Champion::from_manifest_json(
            r#"{"id":"rust-evo-g2-c0","weights":{"path":242,"material":113,"capture":888,"structure":41,"threat":155,"edge":75}}"#,
        )
        .expect("training champion parses");
        assert_eq!(training.generation, 0);
        assert_eq!(training.weights.path, 242);
    }

    #[test]
    fn explicit_weight_specs_are_complete_and_positive() {
        assert_eq!(
            parse_weights_spec("path=241,material=112,capture=887,structure=40,threat=154,edge=74")
                .expect("named weights parse"),
            EvaluationWeights {
                path: 241,
                material: 112,
                capture: 887,
                structure: 40,
                threat: 154,
                edge: 74,
            }
        );
        assert!(parse_weights_spec(
            "path=0,material=112,capture=887,structure=40,threat=154,edge=74"
        )
        .is_err());
        assert!(parse_weights_spec("path=241").is_err());
    }

    #[test]
    fn manifest_and_weight_parsers_cover_json_aliases_and_errors() {
        let baseline = Champion::baseline(EvaluationWeights::default());
        assert_eq!(baseline.id, "rust-handcrafted-g0");
        assert_eq!(baseline.generation, 0);
        let from_json = Champion::from_manifest_json(
            r#"{"weights":{"path":1,"material":2,"capture":3,"structure":4,"threat":5,"edge":6}}"#,
        )
        .unwrap();
        assert_eq!(from_json.id, "rust-initial-evaluator");
        assert_eq!(from_json.weights.edge, 6);
        assert!(Champion::from_manifest_json("not json").is_err());
        assert!(Champion::from_manifest_json(r#"{"id":"missing"}"#).is_err());
        assert!(
            Champion::from_manifest_file(Path::new("/definitely/missing/manifest.json")).is_err()
        );
        let path = temp_path("manifest");
        fs::write(&path, r#"{"generation":3,"weights":{"path":1,"material":1,"capture":1,"structure":1,"threat":1,"edge":1}}"#).unwrap();
        assert_eq!(Champion::from_manifest_file(&path).unwrap().generation, 3);
        let _ = fs::remove_file(path);

        assert!(parse_weights_spec("{").is_err());
        assert!(parse_weights_spec("path=1,bad").is_err());
        assert!(parse_weights_spec("unknown=1").is_err());
        assert!(
            parse_weights_spec("path=x,material=1,capture=1,structure=1,threat=1,edge=1").is_err()
        );
        assert!(
            parse_weights_spec("path=-1,material=1,capture=1,structure=1,threat=1,edge=1").is_err()
        );
        assert!(parse_weights_spec("material=1,capture=1,structure=1,threat=1,edge=1").is_err());
        assert_eq!(
            parse_weights_spec("  path=1,material=2,capture=3,structure=4,threat=5,edge=6,")
                .unwrap()
                .path,
            1
        );
        assert!(parse_weights_spec(r#"{"path":"bad"}"#).is_err());
    }

    #[test]
    fn scoring_mutation_and_serialization_helpers_cover_empty_and_populated_results() {
        let empty = MatchScore::default();
        assert_eq!(empty.net(), 0);
        assert_eq!(empty.points_rate_per_mille(), 0);
        let score = MatchScore {
            games: 4,
            wins: 2,
            losses: 1,
            draws: 1,
        };
        assert_eq!(score.net(), 1);
        assert_eq!(score.points_rate_per_mille(), 625);
        let mut random = Mulberry32::new(4);
        assert!(mutate_weight(0, &mut random, 200) >= 1);
        assert!(mutate_weight(10, &mut random, 0) >= 1);
        let weights = EvaluationWeights {
            path: 1,
            material: 2,
            capture: 3,
            structure: 4,
            threat: 5,
            edge: 6,
        };
        assert_eq!(candidate_id(2, 3, weights), "rust-evo-g2-c3-1-2-3-4-5-6");
        assert!(score_json(score).contains("pointsRatePerMille"));
        assert_eq!(
            weights_json(weights),
            r#"{"path":1,"material":2,"capture":3,"structure":4,"threat":5,"edge":6}"#
        );
        let config = SearchConfig {
            tactical_proof_horizon: Some(3),
            ..SearchConfig::default()
        };
        assert!(search_json(config).contains("tacticalProofHorizon"));
        assert!(champion_json(&Champion::baseline(weights)).contains("rust-handcrafted-g0"));

        let result = TrainingResult {
            config: TrainingConfig::default(),
            initial: Champion::baseline(weights),
            champion: Champion {
                id: "winner".to_owned(),
                generation: 2,
                weights,
            },
            trials: vec![
                CandidateTrial {
                    id: "candidate".to_owned(),
                    generation: 1,
                    weights,
                    training: score,
                    evaluation: None,
                    promoted: false,
                },
                CandidateTrial {
                    id: "promoted".to_owned(),
                    generation: 2,
                    weights,
                    training: score,
                    evaluation: Some(score),
                    promoted: true,
                },
            ],
            training_records: Vec::new(),
            evaluation_records: Vec::new(),
        };
        let json = result.to_json();
        assert!(json.contains("\"promotions\":1"));
        assert!(json.contains("\"evaluation\":null"));
        assert!(json.contains("\"evaluation\":{"));
    }

    #[test]
    fn training_handles_empty_generations_tactical_filter_and_output_files() {
        let mut config = TrainingConfig {
            generations: 0,
            population: 0,
            training_pairs: 0,
            evaluation_pairs: 0,
            ..TrainingConfig::default()
        };
        let initial = Champion::baseline(config.search.weights);
        let empty = train(initial.clone(), config);
        assert_eq!(empty.champion, initial);
        assert!(empty.trials.is_empty());

        config.generations = 1;
        config.population = 1;
        config.training_pairs = 1;
        config.evaluation_pairs = 1;
        config.max_plies = 2;
        config.opening_random_plies = 0;
        config.tactical_filter = true;
        config.search = SearchConfig {
            depth: 1,
            max_nodes: 32,
            beam_width: 4,
            ..SearchConfig::default()
        };
        let result = train(initial, config);
        assert_eq!(result.trials.len(), 1);
        assert_eq!(result.training_records.len(), 2);
        assert_eq!(result.evaluation_records.len(), 2);

        let directory = temp_path("output");
        let output = write_training_output(&directory, &empty).unwrap();
        assert_eq!(output.training.games, 0);
        assert_eq!(output.evaluation.games, 0);
        assert!(directory.join("champion.json").exists());
        assert!(directory.join("report.json").exists());
        let _ = fs::remove_dir_all(directory);
    }

    #[test]
    fn summarize_classifies_candidate_from_both_sides_and_draws() {
        let make = |light_agent: &str, dark_agent: &str, winner: Option<Player>| -> GameRecord {
            GameRecord {
                seed: 1,
                max_plies: 1,
                board_size: 7,
                reserve_per_player: 14,
                light_agent: light_agent.to_owned(),
                dark_agent: dark_agent.to_owned(),
                light_specification: "{}".to_owned(),
                dark_specification: "{}".to_owned(),
                winner,
                reason: crate::selfplay::TerminationReason::MaxPlies,
                moves: Vec::new(),
            }
        };
        let records = vec![
            make("candidate", "incumbent", Some(Player::Light)),
            make("incumbent", "candidate", Some(Player::Light)),
            make("candidate", "incumbent", None),
        ];
        let score = summarize(&records, "candidate");
        assert_eq!(
            score,
            MatchScore {
                games: 3,
                wins: 1,
                losses: 1,
                draws: 1
            }
        );
    }
}
