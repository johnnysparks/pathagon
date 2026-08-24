//! Typed validation for the shared JSON interchange contract.
//!
//! The normative field names and bounds live in
//! `contracts/pathagon-contract-v1.schema.json`; this module keeps the native
//! engine's boundary typed and checks the same invariants before a record is
//! accepted.

use serde::{Deserialize, Serialize};

pub const CONTRACT_VERSION: u8 = 1;
pub const RULES_VERSION: &str = "pathagon-rules-v1";

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

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
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

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct ReplayRecord {
    #[serde(rename = "contractVersion")]
    pub contract_version: u8,
    pub seed: u64,
    pub config: GameConfig,
    pub engine: EngineMetadata,
    pub agents: PlayerAgents,
    #[serde(rename = "agentSpecifications")]
    pub agent_specifications: AgentSpecifications,
    pub winner: Option<ContractPlayer>,
    pub result: String,
    pub reason: String,
    pub plies: u16,
    pub moves: Vec<ContractMove>,
}

impl GameConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.rules_version != RULES_VERSION { return Err("unsupported rules version".to_owned()); }
        if !(3..=8).contains(&self.board_size) { return Err("board size outside 3..8".to_owned()); }
        if self.reserve_per_player == 0 || self.reserve_per_player > 64 { return Err("reserve outside 1..64".to_owned()); }
        if self.max_plies == 0 || self.max_plies > 4096 { return Err("maximum plies outside 1..4096".to_owned()); }
        if self.repetition_limit != 3 { return Err("repetition limit must be 3".to_owned()); }
        Ok(())
    }

    pub fn cells(&self) -> u8 { self.board_size * self.board_size }
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
        if self.contract_version != CONTRACT_VERSION { return Err("unsupported position contract version".to_owned()); }
        self.config.validate()?;
        let cells = self.config.cells() as usize;
        if self.board.len() != cells { return Err("position board length mismatch".to_owned()); }
        validate_squares(&self.forbidden, self.config.cells(), "forbidden")?;
        for square in [self.last_relocated_to.light, self.last_relocated_to.dark].into_iter().flatten() {
            if square >= self.config.cells() { return Err("relocation marker outside board".to_owned()); }
        }
        if self.ply > self.config.max_plies { return Err("position ply exceeds maximum".to_owned()); }
        Ok(())
    }
}

impl EngineMetadata {
    pub fn validate(&self) -> Result<(), String> {
        if self.id.is_empty() || self.id.len() > 128 || self.version.is_empty() || self.version.len() > 32 { return Err("invalid engine metadata fields".to_owned()); }
        if !matches!(self.runtime.as_str(), "typescript" | "rust" | "python") || self.rules_version != RULES_VERSION { return Err("invalid engine metadata".to_owned()); }
        Ok(())
    }
}

impl AgentSpecification {
    pub fn validate(&self) -> Result<(), String> {
        if self.id.is_empty() || self.name.is_empty() || self.version.is_empty() || self.engine_id.is_empty() { return Err("invalid agent specification fields".to_owned()); }
        if !matches!(self.kind.as_str(), "random" | "heuristic" | "search" | "learned" | "puct") { return Err("invalid agent kind".to_owned()); }
        self.manifest.validate()?;
        Ok(())
    }
}

impl AgentManifest {
    pub fn validate(&self) -> Result<(), String> {
        if self.manifest_version != 1 || !matches!(self.runtime.as_str(), "typescript" | "rust" | "python") || self.rules_version != RULES_VERSION {
            return Err("invalid agent manifest metadata".to_owned());
        }
        if let Some(hash) = &self.model_hash {
            let digest = hash.strip_prefix("sha256:").unwrap_or("");
            if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) { return Err("invalid agent model hash".to_owned()); }
        }
        Ok(())
    }
}

impl ReplayRecord {
    pub fn validate(&self) -> Result<(), String> {
        if self.contract_version != CONTRACT_VERSION { return Err("unsupported replay contract version".to_owned()); }
        if self.seed > u32::MAX as u64 { return Err("seed outside u32".to_owned()); }
        self.config.validate()?;
        self.engine.validate()?;
        self.agent_specifications.light.validate()?;
        self.agent_specifications.dark.validate()?;
        if self.agents.light != self.agent_specifications.light.id || self.agents.dark != self.agent_specifications.dark.id { return Err("agent ID does not match specification".to_owned()); }
        if self.result != if self.winner.is_some() { "win" } else { "draw" } { return Err("result does not match winner".to_owned()); }
        if !matches!(self.reason.as_str(), "path" | "threefold-repetition" | "max-plies" | "no-legal-action") { return Err("invalid termination reason".to_owned()); }
        if self.plies > self.config.max_plies || self.moves.len() != self.plies as usize { return Err("replay plies do not match moves".to_owned()); }
        for (index, movement) in self.moves.iter().enumerate() {
            if movement.ply != index as u16 + 1 { return Err("move ply is not sequential".to_owned()); }
            movement.action.validate(self.config.cells())?;
            validate_squares(&movement.captured, self.config.cells(), "captured")?;
        }
        Ok(())
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let record: Self = serde_json::from_str(text).map_err(|error| error.to_string())?;
        record.validate()?;
        Ok(record)
    }
}

fn validate_squares(squares: &[u8], cells: u8, label: &str) -> Result<(), String> {
    for (index, square) in squares.iter().enumerate() {
        if *square >= cells || squares[..index].contains(square) { return Err(format!("invalid {label} square")); }
    }
    Ok(())
}
