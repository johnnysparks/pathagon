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
        let action_children = node
            .actions
            .iter()
            .map(|edge| edge.child.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        let children = node
            .children
            .iter()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>();
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
        self.solve_from_seed(&BTreeMap::new())
            .expect("empty retrograde seed is valid")
    }

    /// Continue propagation from a previously persisted exact-value seed.
    /// Seeds are validated against this graph so a checkpoint from a
    /// different frontier cannot silently contaminate the result.
    pub fn solve_from_seed(
        &self,
        seed: &BTreeMap<String, RetrogradeValue>,
    ) -> Result<(BTreeMap<String, RetrogradeValue>, RetrogradeStats), String> {
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
            let mut updates = Vec::new();
            for (key, node) in &self.nodes {
                if values.contains_key(key) {
                    continue;
                }
                let Some(value) = self.resolve_node(node, &values) else {
                    continue;
                };
                updates.push((key.clone(), value));
            }
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
                if !values.contains_key(child) {
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
        if !node.complete || node.children.is_empty() && !node.complete {
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
            values: values.clone(),
        };
        atomic_json_write(path.as_ref(), &checkpoint)
    }

    pub fn write_values(
        &self,
        path: impl AsRef<Path>,
        values: &BTreeMap<String, RetrogradeValue>,
        stats: RetrogradeStats,
    ) -> io::Result<()> {
        let output = RetrogradeOutput {
            schema_version: 1,
            table_family: "pathagon-retrograde-wdl-v1".to_owned(),
            values: values.clone(),
            stats,
            action_values: self.action_values(values),
        };
        atomic_json_write(path.as_ref(), &output)
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
        for (index, shard) in shards.iter().enumerate() {
            let path = directory.join(format!("shard-{index:05}.json"));
            atomic_json_write(&path, shard)?;
        }
        let manifest = serde_json::json!({
            "schemaVersion": 1,
            "tableFamily": "pathagon-retrograde-wdl-v1",
            "shardCount": shard_count,
            "nodes": stats.nodes,
            "edges": stats.edges,
            "solved": stats.solved,
            "draws": stats.draws,
            "unknown": stats.unknown,
            "shards": (0..shard_count)
                .map(|index| format!("shard-{index:05}.json"))
                .collect::<Vec<_>>(),
        });
        atomic_json_write(&directory.join("manifest.json"), &manifest)
    }
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
    let mut merged = BTreeMap::new();
    for index in 0..shard_count {
        let path = directory.join(format!("shard-{index:05}.json"));
        let shard: BTreeMap<String, RetrogradeValue> =
            serde_json::from_str(&fs::read_to_string(&path)?)
                .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
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
    atomic_json_write(path.as_ref(), values)
}

pub fn read_checkpoint(path: impl AsRef<Path>) -> io::Result<FrontierCheckpoint> {
    let source = fs::read_to_string(path.as_ref())?;
    serde_json::from_str(&source).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{}: {error}", path.as_ref().display()),
        )
    })
}

pub fn read_nodes(path: impl AsRef<Path>) -> io::Result<RetrogradeGraph> {
    let source = fs::read_to_string(path.as_ref())?;
    let mut graph = RetrogradeGraph::default();
    for (line_number, line) in source.lines().enumerate() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let node: RetrogradeNode = serde_json::from_str(line).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}:{}: {error}", path.as_ref().display(), line_number + 1),
            )
        })?;
        graph.insert(node).map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}:{}: {error}", path.as_ref().display(), line_number + 1),
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
    }

    #[test]
    fn checkpoint_values_resume_and_shard_deterministically() {
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
        let (values, stats) = graph.solve();
        let resumed = graph.solve_from_seed(&values).unwrap();
        assert_eq!(resumed.0, values);
        assert_eq!(resumed.1.solved, stats.solved);

        let directory =
            std::env::temp_dir().join(format!("pathagon-tablebase-test-{}", std::process::id()));
        graph
            .write_value_shards(&directory, &values, stats, 2)
            .unwrap();
        let manifest: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(directory.join("manifest.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(manifest["shardCount"], 2);
        assert!(directory.join("shard-00000.json").exists());
        assert!(directory.join("shard-00001.json").exists());
        let _ = std::fs::remove_dir_all(directory);
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
}
