//! Disk-facing retrograde propagation for exact endgame graphs.
//!
//! State generation is intentionally a separate concern. A generator supplies
//! D4-canonical keys and every legal child edge; this module then performs
//! monotonic W/D/L propagation. Missing edges or incomplete enumeration never
//! become draws. A closed unresolved region is a draw only when every node in
//! that region is marked complete.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::ground_truth::{GroundTruthOutcome, GroundTruthValue};
use serde::{Deserialize, Serialize};

const COMPACT_VALUE_MAGIC: &[u8; 8] = b"PGTBV01\0";
const COMPACT_ACTION_MAGIC: &[u8; 8] = b"PGTBA01\0";
const COMPACT_GRAPH_MAGIC: &[u8; 8] = b"PGGRF01\0";
const COMPACT_HEADER_BYTES: usize = 20;
const COMPACT_NONE_DISTANCE: u16 = u16::MAX;
const COMPACT_NONE_OUTCOME: u8 = u8::MAX;
const ACTION_ALPHABET: &[u8; 64] =
    b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RetrogradeNode {
    pub key: String,
    #[serde(default)]
    pub children: Vec<String>,
    /// Set true only after the generator has enumerated every legal action.
    #[serde(default)]
    pub complete: bool,
    /// Terminal result from this node's side-to-move perspective.
    #[serde(default)]
    pub terminal: Option<String>,
    /// Exact value imported from an already solved inner ring. This is a
    /// seed, not a claim that the position itself is a rule terminal.
    #[serde(default)]
    pub seed: Option<RetrogradeValue>,
    /// Optional action labels aligned with the legal child edges.
    #[serde(default)]
    pub actions: Vec<RetrogradeEdge>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RetrogradeEdge {
    pub action: String,
    pub child: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RetrogradeActionValue {
    pub action: String,
    pub outcome: GroundTruthOutcome,
    pub distance: Option<u16>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RetrogradeValue {
    pub outcome: GroundTruthOutcome,
    pub distance: Option<u16>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct RetrogradeStats {
    pub nodes: usize,
    pub edges: usize,
    pub rounds: usize,
    pub solved: usize,
    pub draws: usize,
    pub unknown: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RetrogradeOutput {
    pub schema_version: u8,
    pub table_family: String,
    pub values: BTreeMap<String, RetrogradeValue>,
    pub stats: RetrogradeStats,
    #[serde(default)]
    pub action_values: BTreeMap<String, Vec<RetrogradeActionValue>>,
    #[serde(default)]
    pub optimal_actions_complete: BTreeMap<String, bool>,
    #[serde(default)]
    pub optimal_actions: BTreeMap<String, Vec<String>>,
    pub provenance: RetrogradeProvenance,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RetrogradeProvenance {
    pub solver_version: String,
    pub rules_version: String,
    pub proof_lineage: String,
    pub node_count: usize,
    pub edge_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FrontierCheckpoint {
    pub schema_version: u8,
    pub table_family: String,
    pub input_nodes: usize,
    pub input_edges: usize,
    pub rounds: usize,
    pub solved: usize,
    pub draws: usize,
    pub unknown: usize,
    pub complete_graph: bool,
    /// Values already proven at checkpoint time.  An absent key remains
    /// unknown and is safe to revisit after a restart.
    #[serde(default)]
    pub values: BTreeMap<String, RetrogradeValue>,
}

#[derive(Clone, Debug, Default)]
pub struct RetrogradeGraph {
    nodes: BTreeMap<String, RetrogradeNode>,
}

impl RetrogradeGraph {
    pub fn insert(&mut self, node: RetrogradeNode) -> Result<(), String> {
        if node.key.is_empty() {
            return Err("retrograde node key cannot be empty".to_owned());
        }
        if node.children.iter().any(|child| child.is_empty()) {
            return Err(format!(
                "retrograde node {} has an empty child key",
                node.key
            ));
        }
        let action_children = node
            .actions
            .iter()
            .map(|edge| edge.child.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        if let Some(terminal) = node.terminal.as_deref() {
            if parse_outcome(terminal).is_none() {
                return Err(format!(
                    "retrograde node {} has an invalid terminal outcome",
                    node.key
                ));
            }
        }
        if node.seed.is_some_and(|value| !value.outcome.is_known()) {
            return Err(format!(
                "retrograde node {} has an unknown seed value",
                node.key
            ));
        }
        if node
            .actions
            .iter()
            .any(|edge| edge.action.is_empty() || edge.child.is_empty())
        {
            return Err(format!(
                "retrograde node {} has an empty action edge",
                node.key
            ));
        }
        let action_labels = node
            .actions
            .iter()
            .map(|edge| edge.action.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        if action_labels.len() != node.actions.len() {
            return Err(format!(
                "retrograde node {} has duplicate action edges",
                node.key
            ));
        }
        let children = node
            .children
            .iter()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
        if children.len() != node.children.len() {
            return Err(format!(
                "retrograde node {} has duplicate child edges",
                node.key
            ));
        }
        if !node.actions.is_empty() && action_children != children {
            return Err(format!(
                "retrograde node {} action edges must align with children",
                node.key
            ));
        }
        if let Some(previous) = self.nodes.insert(node.key.clone(), node.clone()) {
            if previous != node {
                return Err(format!("contradictory retrograde node {}", node.key));
            }
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn edge_count(&self) -> usize {
        self.nodes.values().map(|node| node.children.len()).sum()
    }

    pub fn solve(&self) -> (BTreeMap<String, RetrogradeValue>, RetrogradeStats) {
        self.solve_parallel_from_seed(&BTreeMap::new(), 1)
            .expect("empty retrograde seed is valid")
    }

    /// Continue propagation from a previously persisted exact-value seed.
    /// Seeds are validated against this graph so a checkpoint from a
    /// different frontier cannot silently contaminate the result.
    pub fn solve_from_seed(
        &self,
        seed: &BTreeMap<String, RetrogradeValue>,
    ) -> Result<(BTreeMap<String, RetrogradeValue>, RetrogradeStats), String> {
        self.solve_parallel_from_seed(seed, 1)
    }

    /// Solve with independent workers over deterministic key ranges. Every
    /// worker reads the same immutable previous-round snapshot; updates are
    /// sorted and merged by the coordinator, so parallelism cannot change
    /// the result or hide a contradiction.
    pub fn solve_parallel_from_seed(
        &self,
        seed: &BTreeMap<String, RetrogradeValue>,
        workers: usize,
    ) -> Result<(BTreeMap<String, RetrogradeValue>, RetrogradeStats), String> {
        let workers = workers.max(1).min(self.nodes.len().max(1));
        let mut values = BTreeMap::new();
        for (key, value) in seed {
            if !self.nodes.contains_key(key) {
                return Err(format!("checkpoint contains unknown node {key}"));
            }
            if !value.outcome.is_known() {
                return Err(format!("checkpoint contains unknown value for node {key}"));
            }
            values.insert(key.clone(), *value);
        }
        for (key, node) in &self.nodes {
            if let Some(seed_value) = node.seed {
                if let Some(previous) = values.get(key) {
                    if previous != &seed_value {
                        return Err(format!("conflicting seed value for node {key}"));
                    }
                } else {
                    values.insert(key.clone(), seed_value);
                }
            }
        }
        for (key, node) in &self.nodes {
            let Some(terminal) = node.terminal.as_deref().and_then(parse_outcome) else {
                continue;
            };
            if !terminal.is_known() {
                continue;
            }
            let terminal_value = RetrogradeValue {
                outcome: terminal,
                distance: Some(0),
            };
            if let Some(previous) = values.get(key) {
                if previous != &terminal_value {
                    return Err(format!(
                        "checkpoint contradicts terminal value for node {key}"
                    ));
                }
            } else {
                values.insert(key.clone(), terminal_value);
            }
        }
        let mut rounds = 0;
        loop {
            rounds += 1;
            let node_keys = self.nodes.keys().map(String::as_str).collect::<Vec<_>>();
            let snapshot = &values;
            let mut updates = std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(workers);
                for range in node_keys.chunks(node_keys.len().div_ceil(workers).max(1)) {
                    handles.push(scope.spawn(move || {
                        range
                            .iter()
                            .filter_map(|key| {
                                if snapshot.contains_key(*key) {
                                    return None;
                                }
                                let node = self.nodes.get(*key)?;
                                self.resolve_node(node, snapshot)
                                    .map(|value| ((*key).to_owned(), value))
                            })
                            .collect::<Vec<_>>()
                    }));
                }
                handles
                    .into_iter()
                    .flat_map(|handle| handle.join().expect("retrograde worker panicked"))
                    .collect::<Vec<_>>()
            });
            updates.sort_by(|left, right| left.0.cmp(&right.0));
            if updates.is_empty() {
                break;
            }
            for (key, value) in updates {
                values.insert(key, value);
            }
        }

        for key in self.nodes.keys().cloned().collect::<Vec<_>>() {
            if !values.contains_key(&key) && self.closed_unresolved_region(&key, &values) {
                values.insert(
                    key,
                    RetrogradeValue {
                        outcome: GroundTruthOutcome::Draw,
                        distance: None,
                    },
                );
            }
        }
        let solved = values
            .values()
            .filter(|value| value.outcome.is_known())
            .count();
        let draws = values
            .values()
            .filter(|value| value.outcome == GroundTruthOutcome::Draw)
            .count();
        let unknown = self.nodes.len().saturating_sub(values.len());
        Ok((
            values,
            RetrogradeStats {
                nodes: self.nodes.len(),
                edges: self.edge_count(),
                rounds,
                solved,
                draws,
                unknown,
            },
        ))
    }

    fn closed_unresolved_region(
        &self,
        start: &str,
        values: &BTreeMap<String, RetrogradeValue>,
    ) -> bool {
        let mut stack = vec![start.to_owned()];
        let mut visited = std::collections::BTreeSet::new();
        while let Some(key) = stack.pop() {
            if !visited.insert(key.clone()) {
                continue;
            }
            let Some(node) = self.nodes.get(&key) else {
                return false;
            };
            if !node.complete {
                return false;
            }
            for child in &node.children {
                if !self.nodes.contains_key(child) {
                    return false;
                }
                if let Some(value) = values.get(child) {
                    if value.outcome != GroundTruthOutcome::Draw {
                        // A closed movement cycle with a known win/loss exit
                        // is not an exact draw: the unresolved branch still
                        // needs proof before the parent can be classified.
                        return false;
                    }
                } else {
                    stack.push(child.clone());
                }
            }
        }
        true
    }

    fn resolve_node(
        &self,
        node: &RetrogradeNode,
        values: &BTreeMap<String, RetrogradeValue>,
    ) -> Option<RetrogradeValue> {
        if !node.complete {
            return None;
        }
        if node.children.is_empty() {
            return Some(RetrogradeValue {
                outcome: GroundTruthOutcome::Draw,
                distance: None,
            });
        }
        let mut child_values = Vec::with_capacity(node.children.len());
        for child in &node.children {
            child_values.push(values.get(child)?);
        }
        let parent_values = child_values
            .iter()
            .map(|value| GroundTruthValue {
                outcome: value.outcome.negate(),
                distance: value.distance.map(|distance| distance.saturating_add(1)),
            })
            .collect::<Vec<_>>();
        if let Some(distance) = parent_values
            .iter()
            .filter(|value| value.outcome == GroundTruthOutcome::Win)
            .filter_map(|value| value.distance)
            .min()
        {
            return Some(RetrogradeValue {
                outcome: GroundTruthOutcome::Win,
                distance: Some(distance),
            });
        }
        if parent_values
            .iter()
            .all(|value| value.outcome == GroundTruthOutcome::Loss)
        {
            return Some(RetrogradeValue {
                outcome: GroundTruthOutcome::Loss,
                distance: parent_values
                    .iter()
                    .filter_map(|value| value.distance)
                    .max(),
            });
        }
        if parent_values.iter().all(|value| value.outcome.is_known()) {
            Some(RetrogradeValue {
                outcome: GroundTruthOutcome::Draw,
                distance: None,
            })
        } else {
            None
        }
    }

    pub fn write_checkpoint(
        &self,
        path: impl AsRef<Path>,
        rounds: usize,
        stats: RetrogradeStats,
        values: &BTreeMap<String, RetrogradeValue>,
    ) -> io::Result<()> {
        let path = path.as_ref();
        let values_path = path.with_extension("values.bin");
        write_compact_values(&values_path, values, key_width(values)?)?;
        let checkpoint = FrontierCheckpoint {
            schema_version: 1,
            table_family: "pathagon-retrograde-frontier-v1".to_owned(),
            input_nodes: self.len(),
            input_edges: self.edge_count(),
            rounds,
            solved: stats.solved,
            draws: stats.draws,
            unknown: stats.unknown,
            complete_graph: self.nodes.values().all(|node| node.complete),
            values: BTreeMap::new(),
        };
        let values_name = values_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "checkpoint values path must have a UTF-8 file name",
                )
            })?;
        let metadata = serde_json::json!({
            "schema_version": checkpoint.schema_version,
            "table_family": checkpoint.table_family,
            "format": "compact-checkpoint-v1",
            "input_nodes": checkpoint.input_nodes,
            "input_edges": checkpoint.input_edges,
            "rounds": checkpoint.rounds,
            "solved": checkpoint.solved,
            "draws": checkpoint.draws,
            "unknown": checkpoint.unknown,
            "complete_graph": checkpoint.complete_graph,
            "values_path": values_name,
            "values_format": "compact-value-v1",
        });
        atomic_json_write(path, &metadata)
    }

    pub fn write_values(
        &self,
        path: impl AsRef<Path>,
        values: &BTreeMap<String, RetrogradeValue>,
        stats: RetrogradeStats,
    ) -> io::Result<()> {
        let action_values = self.action_values(values);
        let mut optimal_actions_complete = BTreeMap::new();
        let mut optimal_actions = BTreeMap::new();
        for (key, node) in &self.nodes {
            let root_value = values.get(key).copied();
            let actions_complete = node.actions.iter().all(|edge| {
                values
                    .get(&edge.child)
                    .is_some_and(|value| value.outcome.is_known())
            });
            let complete = node.complete
                && root_value.is_some_and(|value| value.outcome.is_known())
                && actions_complete;
            optimal_actions_complete.insert(key.clone(), complete);
            if complete {
                let root_outcome = root_value.expect("complete root has a value").outcome;
                optimal_actions.insert(
                    key.clone(),
                    action_values
                        .get(key)
                        .into_iter()
                        .flatten()
                        .filter(|action| action.outcome == root_outcome)
                        .map(|action| action.action.clone())
                        .collect(),
                );
            }
        }
        let output = RetrogradeOutput {
            schema_version: 1,
            table_family: "pathagon-retrograde-wdl-v1".to_owned(),
            values: values.clone(),
            stats,
            action_values,
            optimal_actions_complete,
            optimal_actions,
            provenance: self.provenance(stats),
        };
        atomic_json_write(path.as_ref(), &output)
    }

    /// Write exact W/D/L values as fixed-width binary records. Unknown nodes
    /// are omitted, so absence of a key remains the compact unknown value.
    /// Records are sorted by canonical key for binary search.
    pub fn write_compact_values(
        &self,
        path: impl AsRef<Path>,
        values: &BTreeMap<String, RetrogradeValue>,
    ) -> io::Result<()> {
        write_compact_values(path.as_ref(), values, key_width(values)?)
    }

    /// Write per-action W/D/L/Unknown labels and the complete-optimal-set bit
    /// in a compact sidecar. Action strings are length-prefixed so this also
    /// supports non-board-specific research labels.
    pub fn write_compact_actions(
        &self,
        path: impl AsRef<Path>,
        values: &BTreeMap<String, RetrogradeValue>,
    ) -> io::Result<()> {
        let action_values = self.action_values(values);
        let key_bytes = if action_values.is_empty() {
            key_width(values)?
        } else {
            key_width_from_keys(action_values.keys())?
        };
        let rows = action_values.len();
        atomic_binary_write(path.as_ref(), |writer| {
            writer.write_all(COMPACT_ACTION_MAGIC)?;
            writer.write_all(&[key_bytes as u8, 0, 0, 0])?;
            writer.write_all(&(rows as u64).to_le_bytes())?;
            for (key, actions) in &action_values {
                let key_bytes = decode_hex_key(key, key_bytes)?;
                writer.write_all(&key_bytes)?;
                let complete = self
                    .nodes
                    .get(key)
                    .is_some_and(|node| self.action_set_complete(key, node, values));
                writer.write_all(&[u8::from(complete)])?;
                if actions.len() > u32::MAX as usize {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "compact action row contains too many actions",
                    ));
                }
                writer.write_all(&(actions.len() as u32).to_le_bytes())?;
                let mut actions = actions.clone();
                actions
                    .sort_by_key(|action| compact_action_code(&action.action).unwrap_or(u16::MAX));
                for action in &actions {
                    let action_bytes = action.action.as_bytes();
                    if action_bytes.len() > usize::from(u16::MAX) {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidInput,
                            "compact action label is too long",
                        ));
                    }
                    writer.write_all(&(action_bytes.len() as u16).to_le_bytes())?;
                    writer.write_all(action_bytes)?;
                    writer.write_all(&[outcome_byte(action.outcome)])?;
                    writer.write_all(&distance_bytes(action.distance)?)?;
                }
            }
            Ok(())
        })
    }

    /// Write the legal-edge graph without repeating JSON field names or
    /// hexadecimal key text. Children are sorted once per node and actions
    /// are encoded as two base64-alphabet bytes plus a local child index.
    /// The result is deterministic and can be read back by `read_nodes`.
    pub fn write_compact_graph(&self, path: impl AsRef<Path>) -> io::Result<()> {
        let key_bytes = key_width_from_keys(self.nodes.keys())?;
        if key_bytes > usize::from(u8::MAX) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "compact graph key width exceeds u8",
            ));
        }
        atomic_binary_write(path.as_ref(), |writer| {
            writer.write_all(COMPACT_GRAPH_MAGIC)?;
            writer.write_all(&[key_bytes as u8, 0, 0, 0])?;
            writer.write_all(&(self.nodes.len() as u64).to_le_bytes())?;
            for (key, node) in &self.nodes {
                writer.write_all(&decode_hex_key(key, key_bytes)?)?;
                let terminal = node
                    .terminal
                    .as_deref()
                    .map(|value| {
                        parse_outcome(value).map(outcome_byte).ok_or_else(|| {
                            io::Error::new(
                                io::ErrorKind::InvalidInput,
                                format!("retrograde node {key} has an invalid terminal outcome"),
                            )
                        })
                    })
                    .transpose()?
                    .unwrap_or(COMPACT_NONE_OUTCOME);
                let seed = node.seed.map(|value| outcome_byte(value.outcome));
                let mut flags = u8::from(node.complete);
                if node.terminal.is_some() {
                    flags |= 1 << 1;
                }
                if node.seed.is_some() {
                    flags |= 1 << 2;
                }
                writer.write_all(&[flags, terminal, seed.unwrap_or(COMPACT_NONE_OUTCOME)])?;
                writer.write_all(&distance_bytes(node.seed.and_then(|value| value.distance))?)?;

                let mut children = node.children.clone();
                children.sort();
                if children.len() > u32::MAX as usize {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "compact graph node contains too many children",
                    ));
                }
                let child_indices = children
                    .iter()
                    .enumerate()
                    .map(|(index, child)| (child.as_str(), index as u32))
                    .collect::<BTreeMap<_, _>>();
                let mut actions = node.actions.clone();
                actions.sort_by_key(|edge| compact_action_code(&edge.action).unwrap_or(u16::MAX));
                if actions.len() > u32::MAX as usize {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "compact graph node contains too many actions",
                    ));
                }
                writer.write_all(&(children.len() as u32).to_le_bytes())?;
                writer.write_all(&(actions.len() as u32).to_le_bytes())?;
                for child in &children {
                    writer.write_all(&decode_hex_key(child, key_bytes)?)?;
                }
                for edge in actions {
                    let child_index = child_indices.get(edge.child.as_str()).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidInput,
                            format!(
                                "retrograde action {} points outside its child set",
                                edge.action
                            ),
                        )
                    })?;
                    writer.write_all(&compact_action_code(&edge.action)?.to_le_bytes())?;
                    writer.write_all(&child_index.to_le_bytes())?;
                }
            }
            Ok(())
        })
    }

    /// Write a deterministic, graph-only JSONL evidence export. This is the
    /// reversible inspection path for PGGRF01; generator-specific fields that
    /// were intentionally omitted from the compact graph are not recreated.
    pub fn write_jsonl(&self, path: impl AsRef<Path>) -> io::Result<()> {
        atomic_binary_write(path.as_ref(), |writer| {
            for node in self.nodes.values() {
                serde_json::to_writer(&mut *writer, node)
                    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
                writer.write_all(b"\n")?;
            }
            Ok(())
        })
    }

    pub fn provenance(&self, stats: RetrogradeStats) -> RetrogradeProvenance {
        RetrogradeProvenance {
            solver_version: "pathagon-retrograde-v1".to_owned(),
            rules_version: "pathagon-rules-v1".to_owned(),
            proof_lineage: "complete-forward-legal-edges-plus-exact-inner-seeds".to_owned(),
            node_count: stats.nodes,
            edge_count: stats.edges,
        }
    }

    fn action_set_complete(
        &self,
        key: &str,
        node: &RetrogradeNode,
        values: &BTreeMap<String, RetrogradeValue>,
    ) -> bool {
        node.complete
            && values
                .get(key)
                .is_some_and(|value| value.outcome.is_known())
            && node.actions.iter().all(|edge| {
                values
                    .get(&edge.child)
                    .is_some_and(|value| value.outcome.is_known())
            })
    }

    fn action_values(
        &self,
        values: &BTreeMap<String, RetrogradeValue>,
    ) -> BTreeMap<String, Vec<RetrogradeActionValue>> {
        self.nodes
            .iter()
            .filter(|(_, node)| !node.actions.is_empty())
            .map(|(key, node)| {
                let actions = node
                    .actions
                    .iter()
                    .map(|edge| {
                        let (outcome, distance) = values
                            .get(&edge.child)
                            .map(|child| {
                                (
                                    child.outcome.negate(),
                                    child.distance.map(|distance| distance.saturating_add(1)),
                                )
                            })
                            .unwrap_or((GroundTruthOutcome::Unknown, None));
                        RetrogradeActionValue {
                            action: edge.action.clone(),
                            outcome,
                            distance,
                        }
                    })
                    .collect();
                (key.clone(), actions)
            })
            .collect()
    }

    /// Persist deterministic value shards.  The shard function is stable
    /// across processes and platforms, so independent workers can merge these
    /// files without depending on a randomized hash-map iteration order.
    pub fn write_value_shards(
        &self,
        directory: impl AsRef<Path>,
        values: &BTreeMap<String, RetrogradeValue>,
        stats: RetrogradeStats,
        shard_count: usize,
    ) -> io::Result<()> {
        if shard_count == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "shard count must be positive",
            ));
        }
        let directory = directory.as_ref();
        fs::create_dir_all(directory)?;
        let mut shards = vec![BTreeMap::new(); shard_count];
        for (key, value) in values {
            let index = stable_shard(key.as_bytes(), shard_count);
            shards[index].insert(key.clone(), *value);
        }
        let key_bytes = key_width(values)?;
        for (index, shard) in shards.iter().enumerate() {
            let path = directory.join(format!("shard-{index:05}.bin"));
            write_compact_values(&path, shard, key_bytes)?;
        }
        let manifest = serde_json::json!({
            "schemaVersion": 1,
            "tableFamily": "pathagon-retrograde-wdl-v1",
            "format": "compact-value-v1",
            "record": {
                "keyBytes": key_bytes,
                "valueBytes": 3,
                "distanceSentinel": u16::MAX,
            },
            "solverVersion": "pathagon-retrograde-v1",
            "rulesVersion": "pathagon-rules-v1",
            "proofLineage": "complete-forward-legal-edges-plus-exact-inner-seeds",
            "shardCount": shard_count,
            "nodes": stats.nodes,
            "edges": stats.edges,
            "solved": stats.solved,
            "draws": stats.draws,
            "unknown": stats.unknown,
            "shards": (0..shard_count)
                .map(|index| format!("shard-{index:05}.bin"))
                .collect::<Vec<_>>(),
        });
        atomic_json_write(&directory.join("manifest.json"), &manifest)
    }
}

fn key_width(values: &BTreeMap<String, RetrogradeValue>) -> io::Result<usize> {
    key_width_from_keys(values.keys())
}

fn key_width_from_keys<'a, I>(keys: I) -> io::Result<usize>
where
    I: IntoIterator<Item = &'a String>,
{
    let keys = keys.into_iter().collect::<Vec<_>>();
    let Some(first) = keys.first() else {
        return Ok(0);
    };
    let first_len = first.len();
    if first_len % 2 != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "retrograde key must contain an even number of hex digits",
        ));
    }
    let width = first_len / 2;
    if width > usize::from(u8::MAX) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "retrograde key is too wide for compact format",
        ));
    }
    for key in keys {
        if key.len() != first_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "retrograde keys have inconsistent widths",
            ));
        }
    }
    Ok(width)
}

fn decode_hex_key(key: &str, width: usize) -> io::Result<Vec<u8>> {
    if key.len() != width * 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("retrograde key {key} has the wrong width"),
        ));
    }
    let bytes = key.as_bytes();
    let mut output = Vec::with_capacity(width);
    for index in (0..bytes.len()).step_by(2) {
        let high = hex_value(bytes[index]).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("retrograde key {key} contains non-hex digits"),
            )
        })?;
        let low = hex_value(bytes[index + 1]).expect("validated key pair");
        output.push((high << 4) | low);
    }
    Ok(output)
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn hex_key(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn outcome_byte(outcome: GroundTruthOutcome) -> u8 {
    match outcome {
        GroundTruthOutcome::Loss => 0,
        GroundTruthOutcome::Draw => 1,
        GroundTruthOutcome::Win => 2,
        GroundTruthOutcome::Unknown => 3,
    }
}

fn outcome_from_byte(byte: u8) -> Option<GroundTruthOutcome> {
    match byte {
        0 => Some(GroundTruthOutcome::Loss),
        1 => Some(GroundTruthOutcome::Draw),
        2 => Some(GroundTruthOutcome::Win),
        3 => Some(GroundTruthOutcome::Unknown),
        _ => None,
    }
}

fn distance_bytes(distance: Option<u16>) -> io::Result<[u8; 2]> {
    if distance == Some(COMPACT_NONE_DISTANCE) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "compact distance cannot equal the missing-distance sentinel",
        ));
    }
    Ok(distance.unwrap_or(COMPACT_NONE_DISTANCE).to_le_bytes())
}

fn compact_action_code(action: &str) -> io::Result<u16> {
    let bytes = action.as_bytes();
    if bytes.len() != 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("compact graph action {action:?} must be exactly two corpus characters"),
        ));
    }
    let high = ACTION_ALPHABET
        .iter()
        .position(|candidate| *candidate == bytes[0])
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("compact graph action {action:?} contains an unsupported character"),
            )
        })?;
    let low = ACTION_ALPHABET
        .iter()
        .position(|candidate| *candidate == bytes[1])
        .expect("validated action alphabet");
    Ok(((high as u16) << 6) | low as u16)
}

fn compact_action_from_code(code: u16) -> Option<String> {
    if code >= 64 * 64 {
        return None;
    }
    let high = ACTION_ALPHABET[(code >> 6) as usize] as char;
    let low = ACTION_ALPHABET[(code & 0x3f) as usize] as char;
    Some(format!("{high}{low}"))
}

fn compact_error(path: &Path, message: impl Into<String>) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("{}: {}", path.display(), message.into()),
    )
}

fn compact_take<'a>(
    bytes: &'a [u8],
    offset: &mut usize,
    length: usize,
    path: &Path,
    field: &str,
) -> io::Result<&'a [u8]> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| compact_error(path, format!("compact graph {field} overflows")))?;
    if end > bytes.len() {
        return Err(compact_error(
            path,
            format!("compact graph is truncated in {field}"),
        ));
    }
    let value = &bytes[*offset..end];
    *offset = end;
    Ok(value)
}

/// Read a PGGRF01 compact graph directly.
pub fn read_compact_graph(path: impl AsRef<Path>) -> io::Result<RetrogradeGraph> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    read_compact_graph_bytes(path, &bytes)
}

fn read_compact_graph_bytes(path: &Path, bytes: &[u8]) -> io::Result<RetrogradeGraph> {
    if bytes.len() < COMPACT_HEADER_BYTES || bytes[..8] != *COMPACT_GRAPH_MAGIC {
        return Err(compact_error(path, "invalid compact graph header"));
    }
    let key_bytes = bytes[8] as usize;
    if bytes[9..12] != [0, 0, 0] {
        return Err(compact_error(path, "unsupported compact graph flags"));
    }
    let rows = usize::try_from(u64::from_le_bytes(
        bytes[12..20]
            .try_into()
            .expect("eight-byte graph row count"),
    ))
    .map_err(|_| compact_error(path, "compact graph row count is too large"))?;
    if rows > 0 && key_bytes == 0 {
        return Err(compact_error(
            path,
            "non-empty compact graph has a zero-width key",
        ));
    }

    let mut graph = RetrogradeGraph::default();
    let mut offset = COMPACT_HEADER_BYTES;
    let mut previous_key = None;
    for row in 0..rows {
        let key_bytes_value = compact_take(bytes, &mut offset, key_bytes, path, "node key")?;
        if previous_key
            .as_ref()
            .is_some_and(|previous: &Vec<u8>| previous.as_slice() >= key_bytes_value)
        {
            return Err(compact_error(
                path,
                format!("compact graph node keys are not strictly sorted at row {row}"),
            ));
        }
        let key = hex_key(key_bytes_value);
        previous_key = Some(key_bytes_value.to_vec());

        let flags = compact_take(bytes, &mut offset, 1, path, "node flags")?[0];
        if flags & !0b111 != 0 {
            return Err(compact_error(
                path,
                format!("compact graph row {row} has unsupported node flags"),
            ));
        }
        let terminal_byte = compact_take(bytes, &mut offset, 1, path, "terminal outcome")?[0];
        let seed_byte = compact_take(bytes, &mut offset, 1, path, "seed outcome")?[0];
        let seed_distance = u16::from_le_bytes(
            compact_take(bytes, &mut offset, 2, path, "seed distance")?
                .try_into()
                .expect("two-byte seed distance"),
        );
        let terminal = if flags & (1 << 1) != 0 {
            Some(
                outcome_from_byte(terminal_byte)
                    .ok_or_else(|| compact_error(path, "invalid compact terminal outcome"))?,
            )
        } else {
            if terminal_byte != COMPACT_NONE_OUTCOME {
                return Err(compact_error(
                    path,
                    format!("compact graph row {row} has an unmarked terminal outcome"),
                ));
            }
            None
        };
        let seed = if flags & (1 << 2) != 0 {
            let outcome = outcome_from_byte(seed_byte)
                .ok_or_else(|| compact_error(path, "invalid compact seed outcome"))?;
            if !outcome.is_known() {
                return Err(compact_error(
                    path,
                    format!("compact graph row {row} has an unknown seed outcome"),
                ));
            }
            Some(RetrogradeValue {
                outcome,
                distance: (seed_distance != COMPACT_NONE_DISTANCE).then_some(seed_distance),
            })
        } else {
            if seed_byte != COMPACT_NONE_OUTCOME || seed_distance != COMPACT_NONE_DISTANCE {
                return Err(compact_error(
                    path,
                    format!("compact graph row {row} has unmarked seed metadata"),
                ));
            }
            None
        };

        let child_count = u32::from_le_bytes(
            compact_take(bytes, &mut offset, 4, path, "child count")?
                .try_into()
                .expect("four-byte child count"),
        ) as usize;
        let action_count = u32::from_le_bytes(
            compact_take(bytes, &mut offset, 4, path, "action count")?
                .try_into()
                .expect("four-byte action count"),
        ) as usize;
        let mut children = Vec::with_capacity(child_count);
        let mut previous_child = None;
        for _ in 0..child_count {
            let child_bytes = compact_take(bytes, &mut offset, key_bytes, path, "child key")?;
            if previous_child
                .as_ref()
                .is_some_and(|previous: &Vec<u8>| previous.as_slice() >= child_bytes)
            {
                return Err(compact_error(
                    path,
                    format!("compact graph row {row} child keys are not strictly sorted"),
                ));
            }
            children.push(hex_key(child_bytes));
            previous_child = Some(child_bytes.to_vec());
        }
        let mut actions = Vec::with_capacity(action_count);
        let mut previous_action = None;
        for _ in 0..action_count {
            let code = u16::from_le_bytes(
                compact_take(bytes, &mut offset, 2, path, "action code")?
                    .try_into()
                    .expect("two-byte action code"),
            );
            if previous_action.is_some_and(|previous| code <= previous) {
                return Err(compact_error(
                    path,
                    format!("compact graph row {row} action codes are not strictly sorted"),
                ));
            }
            previous_action = Some(code);
            let child_index = u32::from_le_bytes(
                compact_take(bytes, &mut offset, 4, path, "action child index")?
                    .try_into()
                    .expect("four-byte action child index"),
            ) as usize;
            let child = children.get(child_index).ok_or_else(|| {
                compact_error(
                    path,
                    format!("compact graph row {row} action points outside child list"),
                )
            })?;
            let action = compact_action_from_code(code)
                .ok_or_else(|| compact_error(path, "invalid compact action code"))?;
            actions.push(RetrogradeEdge {
                action,
                child: child.clone(),
            });
        }

        graph
            .insert(RetrogradeNode {
                key,
                children,
                complete: flags & 1 != 0,
                terminal: terminal.map(|outcome| match outcome {
                    GroundTruthOutcome::Loss => "loss".to_owned(),
                    GroundTruthOutcome::Draw => "draw".to_owned(),
                    GroundTruthOutcome::Win => "win".to_owned(),
                    GroundTruthOutcome::Unknown => "unknown".to_owned(),
                }),
                seed,
                actions,
            })
            .map_err(|error| compact_error(path, error))?;
    }
    if offset != bytes.len() {
        return Err(compact_error(path, "compact graph contains trailing bytes"));
    }
    Ok(graph)
}

fn atomic_binary_write<F>(path: &Path, write: F) -> io::Result<()>
where
    F: FnOnce(&mut BufWriter<File>) -> io::Result<()>,
{
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("tmp");
    let file = File::create(&temporary)?;
    let mut writer = BufWriter::new(file);
    write(&mut writer)?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    fs::rename(temporary, path)
}

fn write_compact_values(
    path: &Path,
    values: &BTreeMap<String, RetrogradeValue>,
    key_bytes: usize,
) -> io::Result<()> {
    if key_bytes > usize::from(u8::MAX) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "compact key width exceeds u8",
        ));
    }
    atomic_binary_write(path, |writer| {
        writer.write_all(COMPACT_VALUE_MAGIC)?;
        writer.write_all(&[key_bytes as u8, 0, 0, 0])?;
        writer.write_all(&(values.len() as u64).to_le_bytes())?;
        let mut previous = None;
        for (key, value) in values {
            let key_bytes = decode_hex_key(key, key_bytes)?;
            if previous
                .as_ref()
                .is_some_and(|old: &Vec<u8>| *old >= key_bytes)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "compact value keys must be strictly sorted",
                ));
            }
            if !value.outcome.is_known() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "compact value files cannot store unknown rows",
                ));
            }
            writer.write_all(&key_bytes)?;
            writer.write_all(&[outcome_byte(value.outcome)])?;
            writer.write_all(&distance_bytes(value.distance)?)?;
            previous = Some(key_bytes);
        }
        Ok(())
    })
}

/// Read a compact exact-value file, accepting only sorted W/D/L rows.
pub fn read_compact_values(
    path: impl AsRef<Path>,
) -> io::Result<BTreeMap<String, RetrogradeValue>> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    if bytes.len() < COMPACT_HEADER_BYTES || bytes[..8] != *COMPACT_VALUE_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: invalid compact value header", path.display()),
        ));
    }
    let key_bytes = bytes[8] as usize;
    if bytes[9..12] != [0, 0, 0] {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: unsupported compact value flags", path.display()),
        ));
    }
    let rows = u64::from_le_bytes(bytes[12..20].try_into().expect("eight bytes")) as usize;
    let row_bytes = key_bytes.checked_add(3).ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "compact value row is too wide")
    })?;
    let expected = COMPACT_HEADER_BYTES
        .checked_add(rows.checked_mul(row_bytes).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "compact value file is too large",
            )
        })?)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "compact value file is too large",
            )
        })?;
    if bytes.len() != expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: compact value file has the wrong size", path.display()),
        ));
    }
    let mut values = BTreeMap::new();
    let mut offset = COMPACT_HEADER_BYTES;
    let mut previous = None;
    for _ in 0..rows {
        let key = bytes[offset..offset + key_bytes].to_vec();
        offset += key_bytes;
        if previous.as_ref().is_some_and(|old: &Vec<u8>| *old >= key) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: compact value keys are not sorted", path.display()),
            ));
        }
        let outcome = outcome_from_byte(bytes[offset]).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: compact value has an invalid outcome", path.display()),
            )
        })?;
        offset += 1;
        if !outcome.is_known() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: compact value contains unknown row", path.display()),
            ));
        }
        let distance = u16::from_le_bytes(bytes[offset..offset + 2].try_into().expect("two bytes"));
        offset += 2;
        values.insert(
            hex_key(&key),
            RetrogradeValue {
                outcome,
                distance: (distance != COMPACT_NONE_DISTANCE).then_some(distance),
            },
        );
        previous = Some(key);
    }
    Ok(values)
}

fn stable_shard(key: &[u8], shard_count: usize) -> usize {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in key {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    (hash % shard_count as u64) as usize
}

/// Read a deterministic shard directory and reject stale, misplaced, or
/// contradictory values before they can enter a merged table.
pub fn read_value_shards(
    directory: impl AsRef<Path>,
) -> io::Result<BTreeMap<String, RetrogradeValue>> {
    let directory = directory.as_ref();
    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(directory.join("manifest.json"))?)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    let shard_count = manifest["shardCount"].as_u64().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "shard manifest has no shardCount",
        )
    })? as usize;
    if shard_count == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "shard manifest has an invalid shardCount",
        ));
    }
    let shard_paths = manifest
        .get("shards")
        .and_then(serde_json::Value::as_array)
        .map(|paths| {
            paths
                .iter()
                .map(|path| {
                    path.as_str().map(ToOwned::to_owned).ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            "shard manifest contains a non-string path",
                        )
                    })
                })
                .collect::<io::Result<Vec<_>>>()
        })
        .transpose()?
        .unwrap_or_else(|| {
            (0..shard_count)
                .map(|index| format!("shard-{index:05}.json"))
                .collect()
        });
    if shard_paths.len() != shard_count {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "shard manifest path count does not match shardCount",
        ));
    }
    let mut merged = BTreeMap::new();
    for (index, relative_path) in shard_paths.iter().enumerate() {
        let path = directory.join(relative_path);
        let shard = if path.extension().and_then(|extension| extension.to_str()) == Some("bin") {
            read_compact_values(&path)?
        } else {
            serde_json::from_str(&fs::read_to_string(&path)?)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?
        };
        for (key, value) in shard {
            if stable_shard(key.as_bytes(), shard_count) != index {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("key {key} is in the wrong shard"),
                ));
            }
            if let Some(previous) = merged.insert(key.clone(), value) {
                if previous != value {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("contradictory merged value for key {key}"),
                    ));
                }
            }
        }
    }
    Ok(merged)
}

pub fn write_merged_values(
    path: impl AsRef<Path>,
    values: &BTreeMap<String, RetrogradeValue>,
) -> io::Result<()> {
    if path
        .as_ref()
        .extension()
        .and_then(|extension| extension.to_str())
        == Some("bin")
    {
        write_compact_values(path.as_ref(), values, key_width(values)?)
    } else {
        atomic_json_write(path.as_ref(), values)
    }
}

pub fn read_checkpoint(path: impl AsRef<Path>) -> io::Result<FrontierCheckpoint> {
    let path = path.as_ref();
    let source = fs::read_to_string(path)?;
    let manifest: serde_json::Value = serde_json::from_str(&source).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {error}", path.display()),
        )
    })?;
    let compact_values_path = manifest
        .get("values_path")
        .or_else(|| manifest.get("valuesPath"))
        .and_then(serde_json::Value::as_str);
    if let Some(values_path) = compact_values_path {
        let values_path = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new(""))
            .join(values_path);
        let values = read_compact_values(&values_path)?;
        let mut checkpoint: FrontierCheckpoint =
            serde_json::from_value(manifest).map_err(|error| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("{}: {error}", path.display()),
                )
            })?;
        checkpoint.values = values;
        return Ok(checkpoint);
    }
    serde_json::from_value(manifest).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {error}", path.display()),
        )
    })
}

pub fn read_nodes(path: impl AsRef<Path>) -> io::Result<RetrogradeGraph> {
    let path = path.as_ref();
    let bytes = fs::read(path)?;
    if bytes.len() >= COMPACT_GRAPH_MAGIC.len()
        && bytes[..COMPACT_GRAPH_MAGIC.len()] == *COMPACT_GRAPH_MAGIC
    {
        return read_compact_graph_bytes(path, &bytes);
    }
    let source = String::from_utf8(bytes).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "{}: graph is neither compact binary nor UTF-8 JSONL: {error}",
                path.display()
            ),
        )
    })?;
    let mut graph = RetrogradeGraph::default();
    for (line_number, line) in source.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let node: RetrogradeNode = serde_json::from_str(line).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}:{}: {error}", path.display(), line_number + 1),
            )
        })?;
        graph.insert(node).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}:{}: {error}", path.display(), line_number + 1),
            )
        })?;
    }
    Ok(graph)
}

fn parse_outcome(value: &str) -> Option<GroundTruthOutcome> {
    match value {
        "loss" => Some(GroundTruthOutcome::Loss),
        "draw" => Some(GroundTruthOutcome::Draw),
        "win" => Some(GroundTruthOutcome::Win),
        "unknown" => Some(GroundTruthOutcome::Unknown),
        _ => None,
    }
}

fn atomic_json_write<T: Serialize>(path: &Path, value: &T) -> io::Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("tmp");
    let file = File::create(&temporary)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(temporary, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(key: &str, children: &[&str]) -> RetrogradeNode {
        RetrogradeNode {
            key: key.to_owned(),
            children: children.iter().map(|child| (*child).to_owned()).collect(),
            complete: true,
            terminal: None,
            seed: None,
            actions: Vec::new(),
        }
    }

    #[test]
    fn forced_win_distance_propagates_from_terminal_loss() {
        let mut graph = RetrogradeGraph::default();
        graph.insert(node("root", &["terminal"])).unwrap();
        graph
            .insert(RetrogradeNode {
                key: "terminal".to_owned(),
                children: Vec::new(),
                complete: true,
                terminal: Some("loss".to_owned()),
                seed: None,
                actions: Vec::new(),
            })
            .unwrap();
        let (values, _) = graph.solve();
        assert_eq!(values["root"].outcome, GroundTruthOutcome::Win);
        assert_eq!(values["root"].distance, Some(1));
    }

    #[test]
    fn incomplete_cycle_remains_unknown_but_closed_cycle_is_draw() {
        let mut incomplete = RetrogradeGraph::default();
        incomplete
            .insert(RetrogradeNode {
                complete: false,
                ..node("a", &["a"])
            })
            .unwrap();
        assert!(!incomplete.solve().0.contains_key("a"));

        let mut closed = RetrogradeGraph::default();
        closed.insert(node("a", &["b"])).unwrap();
        closed.insert(node("b", &["a"])).unwrap();
        assert_eq!(closed.solve().0["a"].outcome, GroundTruthOutcome::Draw);

        closed
            .insert(RetrogradeNode {
                complete: false,
                ..node("incomplete", &["incomplete"])
            })
            .unwrap();
        assert_eq!(closed.solve().0["a"].outcome, GroundTruthOutcome::Draw);
        assert!(!closed.solve().0.contains_key("incomplete"));

        let mut open_exit = RetrogradeGraph::default();
        open_exit.insert(node("a", &["b", "win"])).unwrap();
        open_exit.insert(node("b", &["b"])).unwrap();
        open_exit
            .insert(RetrogradeNode {
                key: "win".to_owned(),
                children: Vec::new(),
                complete: true,
                terminal: Some("win".to_owned()),
                seed: None,
                actions: Vec::new(),
            })
            .unwrap();
        let values = open_exit.solve().0;
        assert_eq!(values["b"].outcome, GroundTruthOutcome::Draw);
        assert!(!values.contains_key("a"));
    }

    #[test]
    fn checkpoint_values_resume_and_shard_deterministically() {
        let mut graph = RetrogradeGraph::default();
        graph.insert(node("0000", &["1111"])).unwrap();
        graph
            .insert(RetrogradeNode {
                key: "1111".to_owned(),
                children: Vec::new(),
                complete: true,
                terminal: Some("loss".to_owned()),
                seed: None,
                actions: Vec::new(),
            })
            .unwrap();
        let (values, stats) = graph.solve();
        let parallel = graph.solve_parallel_from_seed(&BTreeMap::new(), 2).unwrap();
        assert_eq!(parallel.0, values);
        assert_eq!(parallel.1, stats);
        let resumed = graph.solve_from_seed(&values).unwrap();
        assert_eq!(resumed.0, values);
        assert_eq!(resumed.1.solved, stats.solved);

        let directory =
            std::env::temp_dir().join(format!("pathagon-tablebase-test-{}", std::process::id()));
        graph
            .write_value_shards(&directory, &values, stats, 2)
            .unwrap();
        assert_eq!(read_value_shards(&directory).unwrap(), values);
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(directory.join("manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(manifest["shardCount"], 2);
        assert_eq!(manifest["format"], "compact-value-v1");
        assert!(directory.join("shard-00000.bin").exists());
        assert!(directory.join("shard-00001.bin").exists());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn checkpoint_uses_compact_values_and_round_trips() {
        let mut graph = RetrogradeGraph::default();
        graph.insert(node("0000", &[])).unwrap();
        let (values, stats) = graph.solve();
        let path = std::env::temp_dir().join(format!(
            "pathagon-checkpoint-{}-{}.json",
            std::process::id(),
            stats.nodes
        ));
        graph
            .write_checkpoint(&path, stats.rounds, stats, &values)
            .unwrap();
        let restored = read_checkpoint(&path).unwrap();
        assert_eq!(restored.values, values);
        let metadata: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(metadata["format"], "compact-checkpoint-v1");
        assert_eq!(
            metadata["values_path"],
            path.with_extension("values.bin")
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
        );
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("values.bin"));
    }

    #[test]
    fn legacy_inline_checkpoint_remains_readable() {
        let mut values = BTreeMap::new();
        values.insert(
            "0000".to_owned(),
            RetrogradeValue {
                outcome: GroundTruthOutcome::Win,
                distance: Some(2),
            },
        );
        let checkpoint = FrontierCheckpoint {
            schema_version: 1,
            table_family: "pathagon-retrograde-frontier-v1".to_owned(),
            input_nodes: 1,
            input_edges: 0,
            rounds: 2,
            solved: 1,
            draws: 0,
            unknown: 0,
            complete_graph: true,
            values: values.clone(),
        };
        let path = std::env::temp_dir().join(format!(
            "pathagon-legacy-checkpoint-{}.json",
            std::process::id()
        ));
        std::fs::write(&path, serde_json::to_vec(&checkpoint).unwrap()).unwrap();
        assert_eq!(read_checkpoint(&path).unwrap().values, values);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn contradictory_checkpoint_is_rejected() {
        let mut graph = RetrogradeGraph::default();
        graph
            .insert(RetrogradeNode {
                key: "terminal".to_owned(),
                children: Vec::new(),
                complete: true,
                terminal: Some("loss".to_owned()),
                seed: None,
                actions: Vec::new(),
            })
            .unwrap();
        let mut seed = BTreeMap::new();
        seed.insert(
            "terminal".to_owned(),
            RetrogradeValue {
                outcome: GroundTruthOutcome::Win,
                distance: Some(0),
            },
        );
        assert!(graph.solve_from_seed(&seed).is_err());
    }

    #[test]
    fn imported_inner_ring_seed_propagates_to_parent() {
        let mut graph = RetrogradeGraph::default();
        graph
            .insert(RetrogradeNode {
                key: "parent".to_owned(),
                children: vec!["inner".to_owned()],
                complete: true,
                terminal: None,
                seed: None,
                actions: vec![RetrogradeEdge {
                    action: "move".to_owned(),
                    child: "inner".to_owned(),
                }],
            })
            .unwrap();
        graph
            .insert(RetrogradeNode {
                key: "inner".to_owned(),
                children: Vec::new(),
                complete: false,
                terminal: None,
                seed: Some(RetrogradeValue {
                    outcome: GroundTruthOutcome::Loss,
                    distance: Some(1),
                }),
                actions: Vec::new(),
            })
            .unwrap();
        let (values, _) = graph.solve();
        assert_eq!(values["parent"].outcome, GroundTruthOutcome::Win);
        assert_eq!(values["parent"].distance, Some(2));
    }

    #[test]
    fn distinct_actions_may_share_a_canonical_child() {
        let mut graph = RetrogradeGraph::default();
        graph
            .insert(RetrogradeNode {
                key: "0000".to_owned(),
                children: vec!["1111".to_owned()],
                complete: true,
                terminal: None,
                seed: None,
                actions: vec![
                    RetrogradeEdge {
                        action: "a".to_owned(),
                        child: "1111".to_owned(),
                    },
                    RetrogradeEdge {
                        action: "b".to_owned(),
                        child: "1111".to_owned(),
                    },
                ],
            })
            .unwrap();
    }

    #[test]
    fn compact_output_round_trips_exact_values_and_complete_actions() {
        let mut graph = RetrogradeGraph::default();
        graph
            .insert(RetrogradeNode {
                key: "0000".to_owned(),
                children: vec!["1111".to_owned()],
                complete: true,
                terminal: None,
                seed: None,
                actions: vec![RetrogradeEdge {
                    action: "move".to_owned(),
                    child: "1111".to_owned(),
                }],
            })
            .unwrap();
        graph
            .insert(RetrogradeNode {
                key: "1111".to_owned(),
                children: Vec::new(),
                complete: true,
                terminal: Some("loss".to_owned()),
                seed: None,
                actions: Vec::new(),
            })
            .unwrap();
        let (values, stats) = graph.solve();
        let root = std::env::temp_dir().join(format!(
            "pathagon-tablebase-compact-{}-{}",
            std::process::id(),
            stats.nodes
        ));
        let value_path = root.with_extension("bin");
        let action_path = root.with_extension("actions.bin");
        let json_path = root.with_extension("json");
        graph.write_compact_values(&value_path, &values).unwrap();
        graph.write_compact_actions(&action_path, &values).unwrap();
        graph.write_values(&json_path, &values, stats).unwrap();
        assert_eq!(read_compact_values(&value_path).unwrap(), values);
        let output: RetrogradeOutput =
            serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap()).unwrap();
        assert_eq!(output.optimal_actions_complete["0000"], true);
        assert_eq!(output.optimal_actions["0000"], vec!["move"]);
        assert_eq!(
            output.action_values["0000"][0].outcome,
            GroundTruthOutcome::Win
        );
        let _ = std::fs::remove_file(value_path);
        let _ = std::fs::remove_file(action_path);
        let _ = std::fs::remove_file(json_path);
    }

    #[test]
    fn compact_graph_round_trips_edges_and_solver_values() {
        let mut graph = RetrogradeGraph::default();
        graph
            .insert(RetrogradeNode {
                key: "0000".to_owned(),
                children: vec!["1111".to_owned()],
                complete: true,
                terminal: None,
                seed: None,
                actions: vec![RetrogradeEdge {
                    action: "00".to_owned(),
                    child: "1111".to_owned(),
                }],
            })
            .unwrap();
        graph
            .insert(RetrogradeNode {
                key: "1111".to_owned(),
                children: Vec::new(),
                complete: true,
                terminal: Some("loss".to_owned()),
                seed: None,
                actions: Vec::new(),
            })
            .unwrap();
        let expected = graph.solve();
        let path = std::env::temp_dir().join(format!(
            "pathagon-graph-compact-{}-{}.bin",
            std::process::id(),
            expected.1.nodes
        ));
        graph.write_compact_graph(&path).unwrap();
        let decoded = read_compact_graph(&path).unwrap();
        assert_eq!(decoded.len(), graph.len());
        assert_eq!(decoded.edge_count(), graph.edge_count());
        assert_eq!(decoded.solve(), expected);
        let autodetected = read_nodes(&path).unwrap();
        assert_eq!(autodetected.solve(), expected);
        let jsonl_path = path.with_extension("jsonl");
        decoded.write_jsonl(&jsonl_path).unwrap();
        assert_eq!(read_nodes(&jsonl_path).unwrap().solve(), expected);
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_file(jsonl_path);
    }
}
