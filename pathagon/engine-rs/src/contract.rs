//! Typed validation for the shared JSON interchange contract.
//!
//! The normative field names and bounds live in
//! `contracts/pathagon-contract-v1.schema.json`; this module keeps the native
//! engine's boundary typed and checks the same invariants before a record is
//! accepted.

use serde::{Deserialize, Serialize};

pub const CONTRACT_VERSION: u8 = 1;
pub const RULES_VERSION: &str = "pathagon-rules-v1";
pub const ROOT_Q_SOURCE: &str = "mcts-root-q-v1";

#[derive(Clone, Debug, PartialEq)]
pub struct RootQTargets {
    pub action_values: Vec<f32>,
    pub action_visits: Vec<u32>,
}

impl RootQTargets {
    pub fn new(action_values: Vec<f32>, action_visits: Vec<u32>) -> Result<Self, String> {
        let targets = Self {
            action_values,
            action_visits,
        };
        targets.validate()?;
        Ok(targets)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.action_values.is_empty() || self.action_values.len() != self.action_visits.len() {
            return Err("root-Q values and visits must be non-empty and aligned".to_owned());
        }
        if self
            .action_values
            .iter()
            .any(|value| !value.is_finite() || !(-1.0..=1.0).contains(value))
        {
            return Err("root-Q action value outside -1..1".to_owned());
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ContractPlayer {
    Light,
    Dark,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct GameConfig {
    #[serde(rename = "rulesVersion")]
    pub rules_version: String,
    #[serde(rename = "boardSize")]
    pub board_size: u8,
    #[serde(rename = "reservePerPlayer")]
    pub reserve_per_player: u8,
    #[serde(rename = "maxPlies")]
    pub max_plies: u16,
    #[serde(rename = "repetitionLimit")]
    pub repetition_limit: u8,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind")]
pub enum ContractAction {
    #[serde(rename = "place")]
    Place { to: u8 },
    #[serde(rename = "relocate")]
    Relocate { from: u8, to: u8 },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct Position {
    #[serde(rename = "contractVersion")]
    pub contract_version: u8,
    pub config: GameConfig,
    pub board: Vec<Option<ContractPlayer>>,
    pub reserve: PlayerNumbers,
    pub turn: ContractPlayer,
    pub forbidden: Vec<u8>,
    #[serde(rename = "lastRelocatedTo")]
    pub last_relocated_to: PlayerSquares,
    pub winner: Option<ContractPlayer>,
    pub ply: u16,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PlayerNumbers {
    pub light: u16,
    pub dark: u16,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PlayerSquares {
    pub light: Option<u8>,
    pub dark: Option<u8>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EngineMetadata {
    pub id: String,
    pub runtime: String,
    pub version: String,
    #[serde(rename = "rulesVersion")]
    pub rules_version: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EvaluatorWeights {
    pub path: i32,
    pub material: i32,
    pub capture: i32,
    pub structure: i32,
    pub threat: i32,
    pub edge: i32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AgentManifest {
    #[serde(rename = "manifestVersion")]
    pub manifest_version: u8,
    pub runtime: String,
    #[serde(rename = "rulesVersion")]
    pub rules_version: String,
    #[serde(rename = "evaluatorWeights")]
    pub evaluator_weights: EvaluatorWeights,
    pub depth: u32,
    #[serde(rename = "nodeBudget")]
    pub node_budget: u64,
    pub beam: u32,
    #[serde(rename = "modelHash")]
    pub model_hash: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AgentSpecification {
    pub id: String,
    pub name: String,
    pub version: String,
    pub kind: String,
    #[serde(rename = "engineId")]
    pub engine_id: String,
    pub manifest: AgentManifest,
    pub parameters: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ContractMove {
    pub ply: u16,
    pub player: ContractPlayer,
    pub action: ContractAction,
    pub captured: Vec<u8>,
    pub nodes: u64,
    #[serde(rename = "completedDepth")]
    pub completed_depth: u8,
    #[serde(rename = "tableHits")]
    pub table_hits: u64,
    #[serde(default)]
    pub policy: Option<Vec<f32>>,
    #[serde(
        rename = "actionValues",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub action_values: Option<Vec<f32>>,
    #[serde(
        rename = "actionVisits",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub action_visits: Option<Vec<u32>>,
    #[serde(
        rename = "actionValueSource",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub action_value_source: Option<String>,
    pub score: Option<i32>,
    #[serde(rename = "bookHit")]
    pub book_hit: Option<bool>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PlayerAgents {
    pub light: String,
    pub dark: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AgentSpecifications {
    pub light: AgentSpecification,
    pub dark: AgentSpecification,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ReplayRecord {
    #[serde(rename = "contractVersion")]
    pub contract_version: u8,
    pub seed: u64,
    pub config: GameConfig,
    pub engine: EngineMetadata,
    pub agents: PlayerAgents,
    #[serde(rename = "agentSpecifications")]
    pub agent_specifications: AgentSpecifications,
    #[serde(rename = "initialPosition", default)]
    pub initial_position: Option<Position>,
    #[serde(default)]
    pub provenance: Option<serde_json::Value>,
    pub winner: Option<ContractPlayer>,
    pub result: String,
    pub reason: String,
    pub plies: u16,
    pub moves: Vec<ContractMove>,
}

impl GameConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.rules_version != RULES_VERSION {
            return Err("unsupported rules version".to_owned());
        }
        if !(3..=8).contains(&self.board_size) {
            return Err("board size outside 3..8".to_owned());
        }
        if self.reserve_per_player == 0 || self.reserve_per_player > 64 {
            return Err("reserve outside 1..64".to_owned());
        }
        if self.max_plies == 0 || self.max_plies > 4096 {
            return Err("maximum plies outside 1..4096".to_owned());
        }
        if self.repetition_limit != 3 {
            return Err("repetition limit must be 3".to_owned());
        }
        Ok(())
    }

    pub fn cells(&self) -> u8 {
        self.board_size * self.board_size
    }
}

impl ContractAction {
    fn validate(&self, cells: u8) -> Result<(), String> {
        let valid = |square: u8| square < cells;
        match self {
            Self::Place { to } if valid(*to) => Ok(()),
            Self::Relocate { from, to } if valid(*from) && valid(*to) => Ok(()),
            _ => Err("action square outside configured board".to_owned()),
        }
    }
}

impl Position {
    pub fn validate(&self) -> Result<(), String> {
        if self.contract_version != CONTRACT_VERSION {
            return Err("unsupported position contract version".to_owned());
        }
        self.config.validate()?;
        let cells = self.config.cells() as usize;
        if self.board.len() != cells {
            return Err("position board length mismatch".to_owned());
        }
        validate_squares(&self.forbidden, self.config.cells(), "forbidden")?;
        for square in [self.last_relocated_to.light, self.last_relocated_to.dark]
            .into_iter()
            .flatten()
        {
            if square >= self.config.cells() {
                return Err("relocation marker outside board".to_owned());
            }
        }
        if self.ply > self.config.max_plies {
            return Err("position ply exceeds maximum".to_owned());
        }
        Ok(())
    }
}

impl EngineMetadata {
    pub fn validate(&self) -> Result<(), String> {
        if self.id.is_empty()
            || self.id.len() > 128
            || self.version.is_empty()
            || self.version.len() > 32
        {
            return Err("invalid engine metadata fields".to_owned());
        }
        if !matches!(self.runtime.as_str(), "typescript" | "rust" | "python")
            || self.rules_version != RULES_VERSION
        {
            return Err("invalid engine metadata".to_owned());
        }
        Ok(())
    }
}

impl AgentSpecification {
    pub fn validate(&self) -> Result<(), String> {
        if self.id.is_empty()
            || self.name.is_empty()
            || self.version.is_empty()
            || self.engine_id.is_empty()
        {
            return Err("invalid agent specification fields".to_owned());
        }
        if !matches!(
            self.kind.as_str(),
            "random" | "heuristic" | "search" | "learned" | "puct"
        ) {
            return Err("invalid agent kind".to_owned());
        }
        self.manifest.validate()?;
        Ok(())
    }
}

impl AgentManifest {
    pub fn validate(&self) -> Result<(), String> {
        if self.manifest_version != 1
            || !matches!(self.runtime.as_str(), "typescript" | "rust" | "python")
            || self.rules_version != RULES_VERSION
        {
            return Err("invalid agent manifest metadata".to_owned());
        }
        if let Some(hash) = &self.model_hash {
            let digest = hash.strip_prefix("sha256:").unwrap_or("");
            if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                return Err("invalid agent model hash".to_owned());
            }
        }
        Ok(())
    }
}

impl ReplayRecord {
    pub fn validate(&self) -> Result<(), String> {
        if self.contract_version != CONTRACT_VERSION {
            return Err("unsupported replay contract version".to_owned());
        }
        if self.seed > u32::MAX as u64 {
            return Err("seed outside u32".to_owned());
        }
        self.config.validate()?;
        if let Some(initial) = &self.initial_position {
            initial.validate()?;
            if initial.config.board_size != self.config.board_size
                || initial.config.reserve_per_player != self.config.reserve_per_player
                || initial.config.max_plies != self.config.max_plies
            {
                return Err("initial position configuration does not match replay".to_owned());
            }
            if initial.winner.is_some() {
                return Err("initial position must be non-terminal".to_owned());
            }
        }
        self.engine.validate()?;
        self.agent_specifications.light.validate()?;
        self.agent_specifications.dark.validate()?;
        if self.agents.light != self.agent_specifications.light.id
            || self.agents.dark != self.agent_specifications.dark.id
        {
            return Err("agent ID does not match specification".to_owned());
        }
        if self.result != if self.winner.is_some() { "win" } else { "draw" } {
            return Err("result does not match winner".to_owned());
        }
        if !matches!(
            self.reason.as_str(),
            "path" | "threefold-repetition" | "max-plies" | "no-legal-action"
        ) {
            return Err("invalid termination reason".to_owned());
        }
        if self.plies > self.config.max_plies || self.moves.len() != self.plies as usize {
            return Err("replay plies do not match moves".to_owned());
        }
        for (index, movement) in self.moves.iter().enumerate() {
            if movement.ply != index as u16 + 1 {
                return Err("move ply is not sequential".to_owned());
            }
            movement.action.validate(self.config.cells())?;
            validate_squares(&movement.captured, self.config.cells(), "captured")?;
            if let Some(policy) = &movement.policy {
                if policy.is_empty()
                    || policy.iter().any(|probability| {
                        !probability.is_finite() || *probability < 0.0 || *probability > 1.0
                    })
                    || policy.iter().sum::<f32>() <= 0.0
                {
                    return Err("invalid move policy".to_owned());
                }
            }
            validate_root_q_fields(
                &movement.action_values,
                &movement.action_visits,
                &movement.action_value_source,
            )?;
        }
        Ok(())
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let record: Self = serde_json::from_str(text).map_err(|error| error.to_string())?;
        record.validate()?;
        Ok(record)
    }
}

fn validate_root_q_fields(
    values: &Option<Vec<f32>>,
    visits: &Option<Vec<u32>>,
    source: &Option<String>,
) -> Result<(), String> {
    match (values, visits, source) {
        (None, None, None) => Ok(()),
        (Some(values), Some(visits), Some(source)) if source == ROOT_Q_SOURCE => {
            RootQTargets::new(values.clone(), visits.clone()).map(|_| ())
        }
        _ => Err(
            "root-Q fields must include aligned values, visits, and the supported source"
                .to_owned(),
        ),
    }
}

fn validate_squares(squares: &[u8], cells: u8, label: &str) -> Result<(), String> {
    for (index, square) in squares.iter().enumerate() {
        if *square >= cells || squares[..index].contains(square) {
            return Err(format!("invalid {label} square"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> GameConfig {
        GameConfig {
            rules_version: RULES_VERSION.to_owned(),
            board_size: 3,
            reserve_per_player: 2,
            max_plies: 30,
            repetition_limit: 3,
        }
    }

    fn position() -> Position {
        Position {
            contract_version: CONTRACT_VERSION,
            config: config(),
            board: vec![None; 9],
            reserve: PlayerNumbers { light: 2, dark: 2 },
            turn: ContractPlayer::Light,
            forbidden: Vec::new(),
            last_relocated_to: PlayerSquares {
                light: None,
                dark: None,
            },
            winner: None,
            ply: 0,
        }
    }

    fn metadata() -> EngineMetadata {
        EngineMetadata {
            id: "rust-bitboard".to_owned(),
            runtime: "rust".to_owned(),
            version: "1.0.0".to_owned(),
            rules_version: RULES_VERSION.to_owned(),
        }
    }

    fn manifest() -> AgentManifest {
        AgentManifest {
            manifest_version: 1,
            runtime: "rust".to_owned(),
            rules_version: RULES_VERSION.to_owned(),
            evaluator_weights: EvaluatorWeights {
                path: 1,
                material: 2,
                capture: 3,
                structure: 4,
                threat: 5,
                edge: 6,
            },
            depth: 4,
            node_budget: 256_000,
            beam: 150,
            model_hash: None,
        }
    }

    fn specification(id: &str) -> AgentSpecification {
        AgentSpecification {
            id: id.to_owned(),
            name: format!("{id} agent"),
            version: "1.0.0".to_owned(),
            kind: "search".to_owned(),
            engine_id: "rust-bitboard".to_owned(),
            manifest: manifest(),
            parameters: None,
        }
    }

    fn replay() -> ReplayRecord {
        ReplayRecord {
            contract_version: CONTRACT_VERSION,
            seed: 7,
            config: config(),
            engine: metadata(),
            agents: PlayerAgents {
                light: "light".to_owned(),
                dark: "dark".to_owned(),
            },
            agent_specifications: AgentSpecifications {
                light: specification("light"),
                dark: specification("dark"),
            },
            initial_position: None,
            provenance: None,
            winner: None,
            result: "draw".to_owned(),
            reason: "max-plies".to_owned(),
            plies: 0,
            moves: Vec::new(),
        }
    }

    fn movement(action: ContractAction) -> ContractMove {
        ContractMove {
            ply: 1,
            player: ContractPlayer::Light,
            action,
            captured: Vec::new(),
            nodes: 10,
            completed_depth: 2,
            table_hits: 1,
            policy: None,
            action_values: None,
            action_visits: None,
            action_value_source: None,
            score: Some(3),
            book_hit: Some(false),
        }
    }

    #[test]
    fn root_q_targets_validate_shape_and_range() {
        assert!(RootQTargets::new(vec![], vec![]).is_err());
        assert!(RootQTargets::new(vec![0.0], vec![]).is_err());
        assert!(RootQTargets::new(vec![-1.0, 1.0], vec![1, 2]).is_ok());
        assert!(RootQTargets::new(vec![f32::NAN], vec![1]).is_err());
        assert!(RootQTargets::new(vec![f32::INFINITY], vec![1]).is_err());
        assert!(RootQTargets::new(vec![1.01], vec![1]).is_err());
        assert!(RootQTargets::new(vec![-1.01], vec![1]).is_err());
    }

    #[test]
    fn game_config_and_action_validation_cover_bounds() {
        let mut invalid = config();
        invalid.rules_version = "other-rules".to_owned();
        assert_eq!(invalid.validate().unwrap_err(), "unsupported rules version");
        invalid = config();
        invalid.board_size = 2;
        assert!(invalid.validate().is_err());
        invalid.board_size = 9;
        assert!(invalid.validate().is_err());
        invalid = config();
        invalid.reserve_per_player = 0;
        assert!(invalid.validate().is_err());
        invalid.reserve_per_player = 65;
        assert!(invalid.validate().is_err());
        invalid = config();
        invalid.max_plies = 0;
        assert!(invalid.validate().is_err());
        invalid.max_plies = 4097;
        assert!(invalid.validate().is_err());
        invalid = config();
        invalid.repetition_limit = 2;
        assert!(invalid.validate().is_err());

        assert!(ContractAction::Place { to: 8 }.validate(9).is_ok());
        assert!(ContractAction::Place { to: 9 }.validate(9).is_err());
        assert!(ContractAction::Relocate { from: 0, to: 8 }
            .validate(9)
            .is_ok());
        assert!(ContractAction::Relocate { from: 9, to: 0 }
            .validate(9)
            .is_err());
        assert!(ContractAction::Relocate { from: 0, to: 9 }
            .validate(9)
            .is_err());
    }

    #[test]
    fn position_validation_rejects_each_boundary_invariant() {
        let mut invalid = position();
        invalid.contract_version = 2;
        assert!(invalid.validate().is_err());

        invalid = position();
        invalid.board.pop();
        assert!(invalid.validate().is_err());
        invalid = position();
        invalid.config.rules_version = "bad".to_owned();
        assert!(invalid.validate().is_err());
        invalid = position();
        invalid.forbidden = vec![0, 0];
        assert!(invalid.validate().is_err());
        invalid.forbidden = vec![9];
        assert!(invalid.validate().is_err());
        invalid = position();
        invalid.last_relocated_to.light = Some(9);
        assert!(invalid.validate().is_err());
        invalid = position();
        invalid.ply = invalid.config.max_plies + 1;
        assert!(invalid.validate().is_err());

        let valid = position();
        assert!(valid.validate().is_ok());
    }

    #[test]
    fn metadata_agent_and_manifest_validation_rejects_bad_values() {
        let mut invalid = metadata();
        invalid.id.clear();
        assert!(invalid.validate().is_err());
        invalid = metadata();
        invalid.version.clear();
        assert!(invalid.validate().is_err());
        invalid = metadata();
        invalid.runtime = "go".to_owned();
        assert!(invalid.validate().is_err());
        invalid = metadata();
        invalid.rules_version = "bad".to_owned();
        assert!(invalid.validate().is_err());
        invalid = metadata();
        invalid.id = "x".repeat(129);
        assert!(invalid.validate().is_err());
        invalid = metadata();
        invalid.version = "x".repeat(33);
        assert!(invalid.validate().is_err());

        let mut agent = specification("agent");
        assert!(agent.validate().is_ok());
        agent.id.clear();
        assert!(agent.validate().is_err());
        agent = specification("agent");
        agent.name.clear();
        assert!(agent.validate().is_err());
        agent = specification("agent");
        agent.version.clear();
        assert!(agent.validate().is_err());
        agent = specification("agent");
        agent.engine_id.clear();
        assert!(agent.validate().is_err());
        agent = specification("agent");
        agent.kind = "random".to_owned();
        assert!(agent.validate().is_ok());
        agent.kind = "invalid".to_owned();
        assert!(agent.validate().is_err());

        let mut agent_manifest = manifest();
        agent_manifest.manifest_version = 2;
        assert!(agent_manifest.validate().is_err());
        agent_manifest = manifest();
        agent_manifest.runtime = "go".to_owned();
        assert!(agent_manifest.validate().is_err());
        agent_manifest = manifest();
        agent_manifest.rules_version = "bad".to_owned();
        assert!(agent_manifest.validate().is_err());
        agent_manifest = manifest();
        agent_manifest.model_hash = Some("sha256:bad".to_owned());
        assert!(agent_manifest.validate().is_err());
        agent_manifest.model_hash = Some(format!("sha256:{}", "a".repeat(64)));
        assert!(agent_manifest.validate().is_ok());
        agent_manifest.model_hash = Some("a".repeat(64));
        assert!(agent_manifest.validate().is_err());
    }

    #[test]
    fn replay_validation_covers_metadata_moves_policy_and_root_q() {
        let mut invalid = replay();
        invalid.contract_version = 2;
        assert!(invalid.validate().is_err());
        invalid = replay();
        invalid.seed = u32::MAX as u64 + 1;
        assert!(invalid.validate().is_err());
        invalid = replay();
        invalid.initial_position = Some(position());
        invalid.initial_position.as_mut().unwrap().config.board_size = 4;
        assert!(invalid.validate().is_err());
        invalid = replay();
        invalid.initial_position = Some(position());
        invalid.initial_position.as_mut().unwrap().winner = Some(ContractPlayer::Light);
        assert!(invalid.validate().is_err());
        invalid = replay();
        invalid.engine.runtime = "go".to_owned();
        assert!(invalid.validate().is_err());
        invalid = replay();
        invalid.agents.light = "wrong".to_owned();
        assert!(invalid.validate().is_err());
        invalid = replay();
        invalid.winner = Some(ContractPlayer::Light);
        assert!(invalid.validate().is_err());
        invalid = replay();
        invalid.reason = "unknown".to_owned();
        assert!(invalid.validate().is_err());
        invalid = replay();
        invalid.plies = 1;
        assert!(invalid.validate().is_err());

        let mut valid = replay();
        valid.moves = vec![movement(ContractAction::Place { to: 0 })];
        valid.plies = 1;
        valid.winner = Some(ContractPlayer::Light);
        valid.result = "win".to_owned();
        valid.moves[0].policy = Some(vec![0.25, 0.75]);
        valid.moves[0].action_values = Some(vec![-0.5, 0.5]);
        valid.moves[0].action_visits = Some(vec![2, 3]);
        valid.moves[0].action_value_source = Some(ROOT_Q_SOURCE.to_owned());
        assert!(valid.validate().is_ok());

        let mut bad_move = valid.clone();
        bad_move.moves[0].ply = 2;
        assert!(bad_move.validate().is_err());
        bad_move = valid.clone();
        bad_move.moves[0].action = ContractAction::Place { to: 9 };
        assert!(bad_move.validate().is_err());
        bad_move = valid.clone();
        bad_move.moves[0].captured = vec![0, 0];
        assert!(bad_move.validate().is_err());
        for policy in [
            Vec::new(),
            vec![f32::NAN],
            vec![-0.1],
            vec![1.1],
            vec![0.0, 0.0],
        ] {
            bad_move = valid.clone();
            bad_move.moves[0].policy = Some(policy);
            assert!(bad_move.validate().is_err());
        }
        for (values, visits, source) in [
            (Some(vec![0.0]), None, Some(ROOT_Q_SOURCE.to_owned())),
            (Some(vec![0.0]), Some(vec![1]), None),
            (Some(vec![0.0]), Some(vec![1]), Some("other".to_owned())),
            (
                Some(vec![2.0]),
                Some(vec![1]),
                Some(ROOT_Q_SOURCE.to_owned()),
            ),
        ] {
            bad_move = valid.clone();
            bad_move.moves[0].action_values = values;
            bad_move.moves[0].action_visits = visits;
            bad_move.moves[0].action_value_source = source;
            assert!(bad_move.validate().is_err());
        }

        let json = r#"{"contractVersion":1,"seed":1,"config":{"rulesVersion":"pathagon-rules-v1","boardSize":3,"reservePerPlayer":2,"maxPlies":30,"repetitionLimit":3},"engine":{"id":"engine","runtime":"rust","version":"1","rulesVersion":"pathagon-rules-v1"},"agents":{"light":"light","dark":"dark"},"agentSpecifications":{"light":{"id":"light","name":"light","version":"1","kind":"search","engineId":"rust-bitboard","manifest":{"manifestVersion":1,"runtime":"rust","rulesVersion":"pathagon-rules-v1","evaluatorWeights":{"path":0,"material":0,"capture":0,"structure":0,"threat":0,"edge":0},"depth":1,"nodeBudget":1,"beam":1,"modelHash":null},"parameters":null},"dark":{"id":"dark","name":"dark","version":"1","kind":"search","engineId":"rust-bitboard","manifest":{"manifestVersion":1,"runtime":"rust","rulesVersion":"pathagon-rules-v1","evaluatorWeights":{"path":0,"material":0,"capture":0,"structure":0,"threat":0,"edge":0},"depth":1,"nodeBudget":1,"beam":1,"modelHash":null},"parameters":null}},"winner":null,"result":"draw","reason":"max-plies","plies":0,"moves":[]}"#;
        assert!(ReplayRecord::from_json(json).is_ok());
        assert!(ReplayRecord::from_json("not json").is_err());
    }
}
