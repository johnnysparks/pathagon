//! Deterministic evaluator evolution with held-out promotion matches.

use std::fs;
use std::io;
use std::path::Path;

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
}
